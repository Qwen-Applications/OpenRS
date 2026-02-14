"""
rewardbench_v2.py - RewardBench V2 评测脚本

功能：
1. 处理 RewardBench V2 数据集 (1vsN 比较)
2. 过滤 "Tie" subset
3. 路由评测逻辑：
   - Math / Factuality -> Verifiable (有 GT 时先 verifiable，无法判定则 fallback to pairwise)
   - Precise IF -> Pointwise
   - Chat / Safety / Other -> Pairwise
4. 计算胜率: Win / (Win + Loss)

使用方式：
python rewardbench_v2.py --input data/rewardbench_v2/rewardbench_v2.jsonl --output results/rbv2_results.jsonl
"""

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from evaluator import evaluate_pair, evaluate_precise_if
from tools import load_jsonl, save_jsonl, parse_pair_score
from robust_utils import safe_json_dumps_robust as safe_json_dumps

logger = logging.getLogger(__name__)
file_lock = threading.Lock()

def process_single_item(
    data: Dict[str, Any],
    temperature: float = 0.0,
    min_score: float = 0.0,
    output_dir: str = "./results",
    annotation: str = "",
) -> Dict[str, Any]:
    """处理单条数据 (1 vs N)，并按阶段分流写入文件"""
    subset = data.get('subset', 'unknown')
    question_id = data.get('id', 'unknown')
    
    # 2.1 抛弃 Tie subset
    if subset == 'Tie':
        return {'skip': True}

    try:
        query = data.get('prompt')
        chosen_list = data.get('chosen', [])
        rejected_list = data.get('rejected', [])
        ground_truth = data.get('ground_truth') # Math/Factuality 需要
        constraints = data.get('constraints', []) # 指令遵循预提取结果

        # 规范化 chosen/rejected 为列表
        if isinstance(chosen_list, str): chosen_list = [chosen_list]
        if isinstance(rejected_list, str): rejected_list = [rejected_list]
        
        # 按照 RewardBench 定义，chosen 是列表但通常只有一个正确答案
        # 我们取第一个 chosen 与所有 rejected 比较
        if not chosen_list:
            raise ValueError('No chosen response')
        chosen_response = chosen_list[0]
        
        if not rejected_list:
            raise ValueError('No rejected responses')

        # 评测策略分流
        is_pointwise = bool(constraints)  # 有 constraints → Precise IF 逻辑
        
        comparisons = []
        results_for_sample = [] # 1(Win), 0(Tie), -1(Loss)
        comparison_verdicts = [] # 保存每次比较的原始 verdict
        
        for rejected_response in rejected_list:
            comparison_result = 0
            detail = {}
            single_verdict = 'error'
            
            try:
                if constraints:
                    # 有 constraints → Precise IF 评测
                    # evaluate_precise_if 内部已包含 fallback to pairwise 逻辑
                    res = evaluate_precise_if(
                        query=query,
                        chosen=chosen_response,
                        rejected=rejected_response,
                        constraints=constraints,
                        temperature=temperature
                    )
                    detail = res
                    final_verdict_val = res.get('final_verdict')
                    
                    if final_verdict_val == 'good':
                        comparison_result = 1
                        single_verdict = 'precise_if_good'
                    elif final_verdict_val == 'bad':
                        comparison_result = -1
                        single_verdict = 'precise_if_bad'
                    else:
                        comparison_result = 0
                        single_verdict = 'precise_if_same'
                else:
                    # Math/Factuality 使用 evaluate_pair (传入 GT)
                    # Chat/Safety 等 使用 evaluate_pair (GT=None)
                    # evaluate_pair 会自动处理: 如果 GT 存在且能分出胜负则返回; 否则(或GT为空)跑 Pairwise
                    
                    # 如果是 Math/Factuality, 尝试传入 GT
                    gt_to_use = ground_truth if subset in ['Math', 'Factuality'] else None
                    
                    # 调用 evaluate_pair
                    # query_type 传入 subset 以便选择专用 Pairwise Prompt (如果存在对应 .md)
                    res = evaluate_pair(
                        query=query, 
                        chosen=chosen_response, 
                        rejected=rejected_response, 
                        ground_truth=gt_to_use,
                        query_type=subset, # 尝试匹配 prompts/pairwise_prompts/{subset}.md
                        temperature=temperature,
                        min_score=min_score
                    )
                    
                    detail = res
                    single_verdict = res.get('final_verdict', 'error')
                    
                    # 转换 verdict 为 1/0/-1
                    if single_verdict in ['verifiable_good', 'good']:
                        comparison_result = 1
                    elif single_verdict in ['verifiable_bad', 'bad']:
                        comparison_result = -1
                    else:
                        comparison_result = 0

            except Exception as e:
                comparison_result = 0
                detail = {'error': str(e)}
                single_verdict = 'error'
            
            results_for_sample.append(comparison_result)
            comparison_verdicts.append(single_verdict)
            comparisons.append({
                'rejected_response': rejected_response,
                'result': comparison_result,
                'verdict': single_verdict,
                'detail': detail
            })

        # 聚合结果 (1 vs N)
        # Win: All 1 (Chosen 击败所有 Rejected)
        # Loss: Any -1 (Chosen 输给任何一个 Rejected)
        # Tie: 否则 (Chosen >= 所有 Rejected, 但至少有一个 Tie)
        
        final_verdict = 'Tie'
        if all(r == 1 for r in results_for_sample):
            final_verdict = 'Win'
        elif any(r == -1 for r in results_for_sample):
            final_verdict = 'Loss'
        else:
            final_verdict = 'Tie'
            
        data['eval_comparisons'] = comparisons
        data['final_verdict'] = final_verdict
        
        # 提取关键信息用于分流
        # 根据评测类型决定分流文件
        has_verifiable = any(v.startswith('verifiable_') for v in comparison_verdicts)
        
        # 根据 final_verdict 和评测类型来决定分流文件
        if final_verdict == 'Win':
            if is_pointwise:
                target_file = f"{output_dir}/pointwise_good_cases_{annotation}.jsonl"
            elif has_verifiable:
                target_file = f"{output_dir}/verifiable_good_cases_{annotation}.jsonl"
            else:
                target_file = f"{output_dir}/pairwise_good_cases_{annotation}.jsonl"
        elif final_verdict == 'Loss':
            if is_pointwise:
                target_file = f"{output_dir}/pointwise_bad_cases_{annotation}.jsonl"
            elif has_verifiable:
                target_file = f"{output_dir}/verifiable_bad_cases_{annotation}.jsonl"
            else:
                target_file = f"{output_dir}/pairwise_bad_cases_{annotation}.jsonl"
        else:  # Tie
            if is_pointwise:
                target_file = f"{output_dir}/pointwise_same_cases_{annotation}.jsonl"
            else:
                target_file = f"{output_dir}/pairwise_same_cases_{annotation}.jsonl"
        
        # 写入分流文件和汇总文件
        json_str = safe_json_dumps(data) + '\n'
        with file_lock:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json_str)
            with open(f"{output_dir}/all_results_{annotation}.jsonl", "a", encoding="utf-8") as f:
                f.write(json_str)
        
        return data
        
    except Exception as e:
        error_msg = f"处理失败: {e}"
        logger.error("%s: %s", question_id, error_msg)
        
        data['error'] = error_msg
        data['final_verdict'] = 'error'
        
        with file_lock:
            with open(f"{output_dir}/error_cases_{annotation}.jsonl", "a", encoding="utf-8") as f:
                f.write(safe_json_dumps(data) + '\n')
        
        return data

def main():
    parser = argparse.ArgumentParser(description="RewardBench V2 Evaluation Script")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./results", help="输出目录")
    parser.add_argument("--annotation", default="rbv2", help="输出文件名标注")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subset", type=str, nargs="+", default=None, help="Process specific subset(s) only")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotation = args.annotation
    all_results_file = f"{output_dir}/all_results_{annotation}.jsonl"
    
    logger.info("Loading data: %s", args.input)
    try:
        all_data = load_jsonl(args.input)
    except Exception:
        with open(args.input, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f if line.strip()]

    logger.info("Total raw data: %d", len(all_data))
    
    # 过滤 Tie subset
    all_data = [d for d in all_data if d.get('subset') != 'Tie']
    logger.info("After removing 'Tie' subset: %d", len(all_data))
    
    if args.subset:
        target_subsets = set(args.subset)
        all_data = [d for d in all_data if d.get('subset') in target_subsets]
        logger.info("Filtering subsets %s: %d", args.subset, len(all_data))
        
    # Resume
    done_ids = set()
    if not args.no_resume and os.path.exists(all_results_file):
        with open(all_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'id' in d: done_ids.add(d['id'])
                except: pass
        if done_ids:
            logger.info("Resuming: %d already done.", len(done_ids))
            all_data = [d for d in all_data if d.get('id') not in done_ids]
    else:
        # Clean up old files when starting fresh
        old_files = [
            f"{output_dir}/all_results_{annotation}.jsonl",
            f"{output_dir}/verifiable_good_cases_{annotation}.jsonl",
            f"{output_dir}/verifiable_bad_cases_{annotation}.jsonl",
            f"{output_dir}/pairwise_good_cases_{annotation}.jsonl",
            f"{output_dir}/pairwise_bad_cases_{annotation}.jsonl",
            f"{output_dir}/pairwise_same_cases_{annotation}.jsonl",
            f"{output_dir}/pointwise_good_cases_{annotation}.jsonl",
            f"{output_dir}/pointwise_bad_cases_{annotation}.jsonl",
            f"{output_dir}/pointwise_same_cases_{annotation}.jsonl",
            f"{output_dir}/error_cases_{annotation}.jsonl",
        ]
        for old_file in old_files:
            if os.path.exists(old_file):
                os.remove(old_file)
            
    if args.limit > 0:
        all_data = all_data[:args.limit]

    if not all_data:
        logger.info("Nothing to process.")
        return

    logger.info("Processing %d items...", len(all_data))
    
    func = partial(
        process_single_item, 
        temperature=args.temperature,
        output_dir=output_dir,
        annotation=annotation
    )
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(func, d): d for d in all_data}
        
        for future in tqdm(as_completed(futures), total=len(all_data)):
            try:
                res = future.result()
                if 'skip' in res: continue
                results.append(res)
            except Exception as e:
                logger.error("Error: %s", e)

    # Statistics
    # Reload full if resumed to get complete stats
    if not args.no_resume and os.path.exists(all_results_file):
        all_results = []
        with open(all_results_file, 'r', encoding='utf-8') as f:
             for line in f:
                try:
                    all_results.append(json.loads(line))
                except: pass
        results = all_results

    subset_stats = {}
    for res in results:
        subset = res.get('subset', 'unknown')
        verdict = res.get('final_verdict')
        
        if subset not in subset_stats:
            subset_stats[subset] = {'Win': 0, 'Tie': 0, 'Loss': 0}
        
        if verdict in subset_stats[subset]:
            subset_stats[subset][verdict] += 1

    # 保存统计 JSON
    summary = {'by_subset': subset_stats}
    total_win, total_loss, total_tie = 0, 0, 0
    for stats in subset_stats.values():
        total_win += stats['Win']
        total_loss += stats['Loss']
        total_tie += stats['Tie']
    summary['overall'] = {
        'Win': total_win, 'Loss': total_loss, 'Tie': total_tie,
        'win_rate': total_win / (total_win + total_loss) if (total_win + total_loss) > 0 else 0.0,
    }
    with open(f"{output_dir}/summary_{annotation}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印易读报告
    print("\n")
    print("=" * 70)
    print("  RewardBench V2 评测报告")
    print("=" * 70)
    
    print(f"\n  {'Subset':<20s} {'Win':>6s} {'Loss':>6s} {'Tie':>6s} {'Total':>6s} {'Win Rate':>10s}")
    print("  " + "-" * 60)
    
    for subset in sorted(subset_stats.keys()):
        stats = subset_stats[subset]
        win = stats['Win']
        loss = stats['Loss']
        tie = stats['Tie']
        total = win + loss + tie
        denom = win + loss
        rate = (win / denom) if denom > 0 else 0.0
        print(f"  {subset:<20s} {win:>6d} {loss:>6d} {tie:>6d} {total:>6d} {rate:>9.2%}")

    print("  " + "-" * 60)
    overall_total = total_win + total_loss + total_tie
    overall_rate = total_win / (total_win + total_loss) if (total_win + total_loss) > 0 else 0.0
    print(f"  {'Overall':<20s} {total_win:>6d} {total_loss:>6d} {total_tie:>6d} {overall_total:>6d} {overall_rate:>9.2%}")
    
    print("\n" + "=" * 70)
    print(f"  结果目录: {output_dir}")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
