"""
evaluator.py - 公共评测接口

提供统一的评测函数:
- evaluate_pointwise: 用于 RewardBench V2 中的 Precise IF
- evaluate_verifiable: 事实核查
- evaluate_pairwise: 两两比较
- evaluate_pair: 打包 evaluate_verifiable 和 evaluate_pairwise，统一评测入口

供各数据集脚本调用：
- judgebench_and_ppe.py
- rmbench.py
- rewardbench_v2.py
"""

import glob
import logging
from typing import Any, Dict, Optional

from tools import get_client_response, parse_pair_score, parse_json_result
from prompts.verifiable_prompts.ground_truth_check import ground_truth_check_prompt_template
from prompts.pairwise_prompts.common import pairwise_prompt_common_template
from evaluator_precise_if import evaluate_precise_if

logger = logging.getLogger(__name__)


def _load_pairwise_map() -> Dict[str, str]:
    """加载各 query_type 对应的 pairwise prompt"""

    def get_pairwise_prompt(pairwise_prompt_rubrics: str) -> str:
        sep_1 = '\n\n⚠️ **注意**：以下仅为参考框架，必须结合用户问题特性和 A、B 回答的特点进行筛选与重构，不可照搬！\n\n'
        sep_2 = '\n\n⚠️ **再次强调**：以上仅为参考框架，必须结合用户问题特性和 A、B 回答的特点进行筛选与重构，不可照搬！\n\n'
        if sep_1 not in pairwise_prompt_common_template or sep_2 not in pairwise_prompt_common_template:
            return pairwise_prompt_common_template
        pre_fix = pairwise_prompt_common_template.split(sep_1)[0]
        post_fix = pairwise_prompt_common_template.split(sep_1)[-1].split(sep_2)[-1]
        return pre_fix + sep_1 + pairwise_prompt_rubrics + sep_2 + post_fix

    pairwise_map = {}
    for _file in glob.glob('./prompts/pairwise_prompts/*.md'):
        key = _file.split('/')[-1].replace('.md', '')
        try:
            with open(_file, 'r', encoding='utf-8') as f:
                pairwise_map[key] = get_pairwise_prompt(f.read())
        except Exception:
            pass
    return pairwise_map


PAIRWISE_MAP = _load_pairwise_map()


# ============== 核心函数 ==============

def get_json_result(prompt: str, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    调用模型并解析 JSON 结果

    Args:
        prompt: 评测 prompt
        temperature: 生成温度

    Returns:
        解析后的 JSON 结果，失败返回 None
    """
    for _ in range(5):
        response = get_client_response(prompt, temperature=temperature)
        if response is None:
            continue
        json_result = parse_json_result(response)
        if not isinstance(json_result, dict):
            logger.warning('解析结果不是字典: %s，重试中...', type(json_result))
            continue

        if 'rubric_compares' in json_result:
            if isinstance(json_result['rubric_compares'], list):
                if all('score' in item for item in json_result['rubric_compares']):
                    return json_result
        if 'score' in json_result:
            return json_result
        logger.warning('JSON 格式错误，重试中...')
    return None


def evaluate_pointwise(
    query: str,
    chosen: str,
    rejected: str,
    temperature: float = 0.0,
    constraints: Optional[list] = None,
    subset: str = 'Precise IF',
) -> Dict[str, Any]:
    """
    Pointwise 评测（针对 Instruction Following / Focus）

    分别对 chosen 和 rejected 进行评分 (1/0/-1)，然后比较分数决定胜负。

    Args:
        query: 用户问题
        chosen: 被选择的回答
        rejected: 被拒绝的回答
        temperature: 生成温度
        constraints: 硬性指令列表 (Optional, for Precise IF)
        subset: 子集名称，用于选择 prompt ('Precise IF' or 'Focus')

    Returns:
        评测结果字典
    """
    if subset == 'Precise IF':
        return evaluate_precise_if(query, chosen, rejected, constraints, temperature)

    # Fallback for unknown subsets
    return {'error': f"Unknown subset: {subset}", 'verdict': 'error'}


def evaluate_verifiable(
    query: str,
    chosen: str,
    rejected: str,
    ground_truth: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Verifiable 评测（事实核查）

    Args:
        query: 用户问题
        chosen: 被选择的回答
        rejected: 被拒绝的回答
        ground_truth: 标准答案
        temperature: 生成温度

    Returns:
        包含 chosen_result, rejected_result, chosen_score, rejected_score, verdict 的字典
    """
    result = {}

    try:
        # 评测 chosen
        prompt_chosen = ground_truth_check_prompt_template.format(
            query=query,
            response=chosen,
            ground_truth=ground_truth
        )
        result_chosen = get_json_result(prompt_chosen, temperature=temperature)
        result['chosen_result'] = result_chosen

        # 评测 rejected
        prompt_rejected = ground_truth_check_prompt_template.format(
            query=query,
            response=rejected,
            ground_truth=ground_truth
        )
        result_rejected = get_json_result(prompt_rejected, temperature=temperature)
        result['rejected_result'] = result_rejected

        # 提取分数
        if result_chosen and result_rejected:
            score_chosen = result_chosen.get('score', 0)
            score_rejected = result_rejected.get('score', 0)
            result['chosen_score'] = score_chosen
            result['rejected_score'] = score_rejected

            if score_chosen > score_rejected:
                result['verdict'] = 'chosen_better'
            elif score_chosen < score_rejected:
                result['verdict'] = 'rejected_better'
            else:
                result['verdict'] = 'equal'
        else:
            result['verdict'] = 'error'

    except Exception as e:
        result['error'] = str(e)
        result['verdict'] = 'error'

    return result


def evaluate_pairwise(
    query: str,
    response_a: str,
    response_b: str,
    query_type: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Pairwise 评测（单次 A vs B）

    Args:
        query: 用户问题
        response_a: 回答 A
        response_b: 回答 B
        query_type: 问题类型（用于选择特定 prompt）
        temperature: 生成温度

    Returns:
        包含 raw_result, score, winner 的字典
    """
    result = {}

    try:
        # 选择 prompt 模板
        if query_type and query_type in PAIRWISE_MAP:
            prompt_template = PAIRWISE_MAP[query_type]
        else:
            prompt_template = pairwise_prompt_common_template

        # 构建 prompt
        prompt = prompt_template.format(
            query=query,
            response_a=response_a,
            response_b=response_b
        )

        # 调用模型
        json_result = get_json_result(prompt, temperature=temperature)

        if json_result is None:
            result['error'] = 'Failed to get valid JSON result'
            result['winner'] = None
            result['score'] = None
            return result

        result['raw_result'] = json_result

        # 解析分数
        score = parse_pair_score(json_result)
        result['score'] = score

        # 判定 winner
        if score > 0:
            result['winner'] = 'A'
        elif score < 0:
            result['winner'] = 'B'
        else:
            result['winner'] = 'Tie'

    except Exception as e:
        result['error'] = str(e)
        result['winner'] = None
        result['score'] = None

    return result


def evaluate_pair(
    query: str,
    chosen: str,
    rejected: str,
    ground_truth: Optional[str] = None,
    query_type: Optional[str] = None,
    temperature: float = 0.0,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    完整评测接口（Verifiable + Pairwise 双向）

    这是主要的公共接口，供各评测脚本调用。

    Args:
        query: 用户问题
        chosen: 被选择的回答（应该更好）
        rejected: 被拒绝的回答
        ground_truth: 标准答案（可选，有则先做事实核查）
        query_type: 问题类型（用于选择特定 pairwise prompt）
        temperature: 生成温度
        min_score: 判定阈值

    Returns:
        包含 verifiable, pairwise_forward, pairwise_backward, final_verdict 的字典
    """
    result = {}

    # Part 1: Verifiable (事实核查)
    if ground_truth:
        verifiable_result = evaluate_verifiable(
            query=query,
            chosen=chosen,
            rejected=rejected,
            ground_truth=ground_truth,
            temperature=temperature
        )
        result['verifiable'] = verifiable_result

        if verifiable_result.get('verdict') == 'chosen_better':
            result['final_verdict'] = 'verifiable_good'
            return result
        elif verifiable_result.get('verdict') == 'rejected_better':
            result['final_verdict'] = 'verifiable_bad'
            return result

    # Part 2: Pairwise (双向评测)
    # Forward: A=chosen, B=rejected
    pairwise_forward = evaluate_pairwise(
        query=query,
        response_a=chosen,
        response_b=rejected,
        query_type=query_type,
        temperature=temperature
    )
    result['pairwise_forward'] = pairwise_forward

    # Backward: A=rejected, B=chosen
    pairwise_backward = evaluate_pairwise(
        query=query,
        response_a=rejected,
        response_b=chosen,
        query_type=query_type,
        temperature=temperature
    )
    result['pairwise_backward'] = pairwise_backward

    # 判定最终结果
    score_f = pairwise_forward.get('score')
    score_b = pairwise_backward.get('score')

    if score_f is None or score_b is None:
        result['final_verdict'] = 'error'
    elif score_f > min_score and score_b <= -min_score:
        result['final_verdict'] = 'good'
    elif score_f <= -min_score and score_b > min_score:
        result['final_verdict'] = 'bad'
    else:
        result['final_verdict'] = 'same'

    return result


def compute_metrics_from_verdicts(verdicts: list, verifiable_good_count: int = 0, verifiable_bad_count: int = 0) -> Dict[str, Any]:
    """
    从判定结果列表计算指标

    Args:
        verdicts: final_verdict 结果列表
        verifiable_good_count: verifiable 阶段直接判定为 good 的数量
        verifiable_bad_count: verifiable 阶段直接判定为 bad 的数量

    Returns:
        包含 acc_num, same_num, err_num, all_num, acc_rate, same_rate, parse_failed 的字典
    """
    acc_num = verifiable_good_count
    err_num = verifiable_bad_count
    same_num = 0
    parse_failed = 0

    for verdict in verdicts:
        if verdict == 'good' or verdict == 'verifiable_good':
            acc_num += 1
        elif verdict == 'bad' or verdict == 'verifiable_bad':
            err_num += 1
        elif verdict == 'same':
            same_num += 1
        else:
            parse_failed += 1

    all_num = acc_num + err_num + same_num
    valid_num = acc_num + err_num

    return {
        'acc_num': acc_num,
        'same_num': same_num,
        'err_num': err_num,
        'all_num': all_num,
        'acc_rate': round(acc_num / valid_num, 4) if valid_num > 0 else 0,
        'same_rate': round(same_num / all_num, 4) if all_num > 0 else 0,
        'parse_failed': parse_failed,
    }
