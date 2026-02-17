"""
judgebench_and_ppe.py - JudgeBench and PPE Dataset Evaluation Script

Features:
1. Load JudgeBench / PPE format data
2. Call common evaluation interface from evaluator.py
3. Execute post-processing (routing, statistics, etc.)

Data format requirements:
{
    "question_id": "...",
    "prompt": "...",
    "chosen": "...",
    "rejected": "...",
    "query_type": "...",
    "ground_truth": "..."  # Optional
}
"""

import argparse
import concurrent.futures
import json
import logging
import os
import threading
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from evaluator import evaluate_pair, compute_metrics_from_verdicts
from tools import load_jsonl, save_jsonl, parse_pair_score
from robust_utils import safe_json_dumps_robust as safe_json_dumps

logger = logging.getLogger(__name__)

# File lock
file_lock = threading.Lock()


def process_single_item(
    data: Dict[str, Any],
    temperature: float = 0.0,
    annotation: str = "",
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """
    Process single item
    
    Args:
        data: data item
        temperature: generation temperature
        annotation: annotation (for output filename)
        output_dir: output directory
        
    Returns:
        Processed data
    """
    question_id = data.get('question_id', 'unknown')
    
    try:
        query = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        query_type = data.get('query_type', 'general')
        ground_truth = data.get('ground_truth')
        
        # Call common evaluation interface
        eval_result = evaluate_pair(
            query=query,
            chosen=chosen,
            rejected=rejected,
            ground_truth=ground_truth,
            query_type=query_type,
            temperature=temperature,
        )
        
        # Merge results to original data
        data['eval_result'] = eval_result
        data['final_verdict'] = eval_result.get('final_verdict', 'error')
        
        # Extract pairwise results (compatible with original format)
        if 'pairwise_forward' in eval_result:
            data['pairwise_judge_chosen_rejected'] = eval_result['pairwise_forward'].get('raw_result')
        if 'pairwise_backward' in eval_result:
            data['pairwise_judge_rejected_chosen'] = eval_result['pairwise_backward'].get('raw_result')
        
        # Extract verifiable results
        if 'verifiable' in eval_result:
            data['verifiable_judge_chosen'] = eval_result['verifiable'].get('chosen_result')
            data['verifiable_judge_rejected'] = eval_result['verifiable'].get('rejected_result')
        
        # Routing write
        verdict = eval_result.get('final_verdict', 'error')
        
        if verdict == 'verifiable_good':
            target_file = f"{output_dir}/verifiable_good_cases_{annotation}.jsonl"
        elif verdict == 'verifiable_bad':
            target_file = f"{output_dir}/verifiable_bad_cases_{annotation}.jsonl"
        elif verdict == 'good':
            target_file = f"{output_dir}/pairwise_good_cases_{annotation}.jsonl"
        elif verdict == 'bad':
            target_file = f"{output_dir}/pairwise_bad_cases_{annotation}.jsonl"
        elif verdict == 'same':
            target_file = f"{output_dir}/pairwise_same_cases_{annotation}.jsonl"
        else:
            target_file = f"{output_dir}/error_cases_{annotation}.jsonl"
        
        # Write routing files and summary file
        json_str = safe_json_dumps(data) + '\n'
        with file_lock:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json_str)
            with open(f"{output_dir}/all_results_{annotation}.jsonl", "a", encoding="utf-8") as f:
                f.write(json_str)
        
        return data
        
    except Exception as e:
        error_msg = f"Processing failed: {e}"
        logger.error("%s: %s", question_id, error_msg)
        
        data['error'] = error_msg
        data['final_verdict'] = 'error'
        
        with file_lock:
            with open(f"{output_dir}/error_cases_{annotation}.jsonl", "a", encoding="utf-8") as f:
                f.write(safe_json_dumps(data) + '\n')
        
        return data


def compute_score_from_files(
    output_dir: str,
    annotation: str,
) -> Dict[str, Any]:
    """
    Compute scores from output files (compatible with original compute_score logic)
    """
    verifiable_good_file = f"{output_dir}/verifiable_good_cases_{annotation}.jsonl"
    verifiable_bad_file = f"{output_dir}/verifiable_bad_cases_{annotation}.jsonl"
    pairwise_good_file = f"{output_dir}/pairwise_good_cases_{annotation}.jsonl"
    pairwise_bad_file = f"{output_dir}/pairwise_bad_cases_{annotation}.jsonl"
    pairwise_same_file = f"{output_dir}/pairwise_same_cases_{annotation}.jsonl"
    
    def count_lines(filepath):
        if not os.path.exists(filepath):
            return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    
    v_good = count_lines(verifiable_good_file)
    v_bad = count_lines(verifiable_bad_file)
    p_good = count_lines(pairwise_good_file)
    p_bad = count_lines(pairwise_bad_file)
    p_same = count_lines(pairwise_same_file)
    
    acc_num = v_good + p_good
    err_num = v_bad + p_bad
    same_num = p_same
    all_num = acc_num + err_num + same_num
    valid_num = acc_num + err_num
    
    return {
        'acc_num': acc_num,
        'same_num': same_num,
        'err_num': err_num,
        'all_num': all_num,
        'acc_rate': round(acc_num / valid_num, 4) if valid_num > 0 else 0,
        'same_rate': round(same_num / all_num, 4) if all_num > 0 else 0,
        'details': {
            'verifiable_good': v_good,
            'verifiable_bad': v_bad,
            'pairwise_good': p_good,
            'pairwise_bad': p_bad,
            'pairwise_same': p_same,
        }
    }


def compute_score_by_query_type(
    output_dir: str,
    annotation: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Statistics by query_type
    """
    all_results_file = f"{output_dir}/all_results_{annotation}.jsonl"
    if not os.path.exists(all_results_file):
        return {}
    
    # Group statistics by query_type
    stats_by_type: Dict[str, Dict[str, int]] = {}
    
    with open(all_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                query_type = data.get('query_type', 'unknown')
                verdict = data.get('final_verdict', 'error')
                
                if query_type not in stats_by_type:
                    stats_by_type[query_type] = {
                        'acc_num': 0, 'err_num': 0, 'same_num': 0
                    }
                
                if verdict in ('good', 'verifiable_good'):
                    stats_by_type[query_type]['acc_num'] += 1
                elif verdict in ('bad', 'verifiable_bad'):
                    stats_by_type[query_type]['err_num'] += 1
                elif verdict == 'same':
                    stats_by_type[query_type]['same_num'] += 1
            except:
                continue
    
    # Calculate ratios
    result = {}
    for query_type, stats in stats_by_type.items():
        all_num = stats['acc_num'] + stats['err_num'] + stats['same_num']
        valid_num = stats['acc_num'] + stats['err_num']
        result[query_type] = {
            **stats,
            'all_num': all_num,
            'acc_rate': round(stats['acc_num'] / valid_num, 4) if valid_num > 0 else 0,
            'same_rate': round(stats['same_num'] / all_num, 4) if all_num > 0 else 0,
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="JudgeBench/PPE Dataset Evaluation Script")
    
    parser.add_argument("--input", required=True, help="Input data file path")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--annotation", default="", help="Annotation (for output filename)")
    
    parser.add_argument("--workers", type=int, default=50, help="Concurrency")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items")
    
    parser.add_argument("--query-type", type=str, default=None, help="Process specific query_type only")
    parser.add_argument("--exclude-label-error", action="store_true", help="Exclude label_error data")
    parser.add_argument("--require-ground-truth", action="store_true", help="Process data with ground_truth only")
    
    parser.add_argument("--no-resume", action="store_true", help="Do not resume")
    parser.add_argument("--stats-only", action="store_true", help="Statistics only, no evaluation")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If statistics only
    if args.stats_only:
        logger.info("Calculating statistics...")
        overall = compute_score_from_files(args.output_dir, args.annotation)
        logger.info("Overall statistics: %s", overall)
        
        by_type = compute_score_by_query_type(args.output_dir, args.annotation)
        logger.info("Statistics by query_type:")
        for qt, stats in sorted(by_type.items()):
            logger.info("  %s: acc_rate=%s, same_rate=%s, all_num=%s", qt, stats['acc_rate'], stats['same_rate'], stats['all_num'])
        return
    
    # Load data
    logger.info("Load data: %s", args.input)
    all_data = load_jsonl(args.input)
    all_data = [d for d in all_data if d.get('label_error') is not True]
    logger.info("Total %d items", len(all_data))
    
    # Filter data
    if args.query_type:
        all_data = [d for d in all_data if d.get('query_type') == args.query_type]
        logger.info("Filtering query_type=%s: %d items", args.query_type, len(all_data))
    
    if args.exclude_label_error:
        all_data = [d for d in all_data if not d.get('label_error', False)]
        logger.info("Excluding label_error: %d items", len(all_data))
    
    if args.require_ground_truth:
        all_data = [d for d in all_data if d.get('ground_truth')]
        logger.info("Require ground_truth: %d items", len(all_data))
    
    # Resume
    if not args.no_resume:
        all_results_file = f"{args.output_dir}/all_results_{args.annotation}.jsonl"
        if os.path.exists(all_results_file):
            done_ids = set()
            with open(all_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get('question_id'):
                            done_ids.add(d['question_id'])
                    except:
                        pass
            if done_ids:
                logger.info("Resume: %d items completed", len(done_ids))
                all_data = [d for d in all_data if d.get('question_id') not in done_ids]
    else:
        # Clear output files
        for pattern in ['verifiable_good', 'verifiable_bad', 'pairwise_good', 'pairwise_bad', 'pairwise_same', 'error', 'all_results']:
            filepath = f"{args.output_dir}/{pattern}_cases_{args.annotation}.jsonl"
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Limit number of items
    if args.limit > 0:
        all_data = all_data[:args.limit]
    
    if not all_data:
        logger.info("No data to process")
    else:
        logger.info("Start processing %d items", len(all_data))
        logger.info("Concurrency: %d", args.workers)
        logger.info("Temperature: %s", args.temperature)
        
        # Build query type statistics
        query_types = Counter([d.get('query_type', 'unknown') for d in all_data])
        logger.info("Query type distribution: %s", dict(query_types))
        
        # Concurrent processing
        func = partial(
            process_single_item,
            temperature=args.temperature,
            annotation=args.annotation,
            output_dir=args.output_dir,
        )
        
        success_count = 0
        error_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(func, data): data for data in all_data}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_data), desc="Evaluation progress"):
                try:
                    result = future.result()
                    if result.get('final_verdict') != 'error':
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error("Task error: %s", e)
        
        logger.info("=" * 50)
        logger.info("Processing complete: Success=%d, Error=%d", success_count, error_count)
    
    # Compute statistics
    overall = compute_score_from_files(args.output_dir, args.annotation)
    by_type = compute_score_by_query_type(args.output_dir, args.annotation)
    
    # Save statistics
    summary = {
        'overall': overall,
        'by_query_type': by_type,
    }
    with open(f"{args.output_dir}/summary_{args.annotation}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print readable report
    _print_report(overall, by_type, args.output_dir)


def _print_report(overall: Dict, by_type: Dict, output_dir: str):
    """Print formatted results statistics report"""
    d = overall.get('details', {})
    
    print("\n")
    print("=" * 70)
    print("  JudgeBench / PPE Evaluation Report")
    print("=" * 70)
    
    # Overall Statistics
    print(f"\n  Total samples: {overall['all_num']}")
    print(f"  ├─ Chosen Better (Good):  {overall['acc_num']:>5d}   ({overall['acc_rate']:.2%})")
    print(f"  ├─ Rejected Better (Bad): {overall['err_num']:>5d}")
    print(f"  ├─ Same:         {overall['same_num']:>5d}   ({overall['same_rate']:.2%})")
    print(f"  └─ Accuracy (Win/Valid):         {overall['acc_rate']:.2%}")
    
    # Source Distribution
    print(f"\n  Source Distribution:")
    print(f"  ├─ Verifiable Good:  {d.get('verifiable_good', 0):>4d}    Verifiable Bad:  {d.get('verifiable_bad', 0):>4d}")
    print(f"  └─ Pairwise Good:    {d.get('pairwise_good', 0):>4d}    Pairwise Bad:    {d.get('pairwise_bad', 0):>4d}    Pairwise Same: {d.get('pairwise_same', 0):>4d}")
    
    # Statistics by query_type
    if by_type:
        print(f"\n  {'Query Type':<20s} {'Good':>6s} {'Bad':>6s} {'Same':>6s} {'Total':>6s} {'Acc Rate':>10s}")
        print("  " + "-" * 60)
        for qt, stats in sorted(by_type.items(), key=lambda x: -x[1]['all_num']):
            acc = stats['acc_num']
            err = stats['err_num']
            same = stats['same_num']
            total = stats['all_num']
            rate = stats['acc_rate']
            print(f"  {qt:<20s} {acc:>6d} {err:>6d} {same:>6d} {total:>6d} {rate:>9.2%}")
    
    print("\n" + "=" * 70)
    print(f"  Result directory: {output_dir}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
