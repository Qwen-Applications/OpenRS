"""
evaluator.py - Common Evaluation Interface

Provides unified evaluation functions:
- evaluate_pointwise: For Precise IF in RewardBench V2
- evaluate_verifiable: Fact checking
- evaluate_pairwise: Pairwise comparison
- evaluate_pair: Packages evaluate_verifiable and evaluate_pairwise, unified evaluation entry point

Called by dataset scripts:
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
    """Load pairwise prompts for each query_type"""

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


# ============== Core Functions ==============

def get_json_result(prompt: str, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Call model and parse JSON result

    Args:
        prompt: evaluation prompt
        temperature: generation temperature

    Returns:
        Parsed JSON result, None if failed
    """
    for _ in range(5):
        response = get_client_response(prompt, temperature=temperature)
        if response is None:
            continue
        json_result = parse_json_result(response)
        if not isinstance(json_result, dict):
            logger.warning('Parsed result is not a dict: %s, retrying...', type(json_result))
            continue

        if 'rubric_compares' in json_result:
            if isinstance(json_result['rubric_compares'], list):
                if all('score' in item for item in json_result['rubric_compares']):
                    return json_result
        if 'score' in json_result:
            return json_result
        logger.warning('JSON format error, retrying...')
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
    Pointwise evaluation (for Instruction Following / Focus)

    Score chosen and rejected separately (1/0/-1), then compare scores to determine winner.

    Args:
        query: User question
        chosen: Chosen response
        rejected: Rejected response
        temperature: Generation temperature
        constraints: List of hard constraints (Optional, for Precise IF)
        subset: Subset name, used to select prompt ('Precise IF' or 'Focus')

    Returns:
        Evaluation result dictionary
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
    Verifiable evaluation (Fact Checking)

    Args:
        query: User question
        chosen: Chosen response
        rejected: Rejected response
        ground_truth: Ground truth
        temperature: Generation temperature

    Returns:
        Dictionary containing chosen_result, rejected_result, chosen_score, rejected_score, verdict
    """
    result = {}

    try:
        # Evaluate chosen
        prompt_chosen = ground_truth_check_prompt_template.format(
            query=query,
            response=chosen,
            ground_truth=ground_truth
        )
        result_chosen = get_json_result(prompt_chosen, temperature=temperature)
        result['chosen_result'] = result_chosen

        # Evaluate rejected
        prompt_rejected = ground_truth_check_prompt_template.format(
            query=query,
            response=rejected,
            ground_truth=ground_truth
        )
        result_rejected = get_json_result(prompt_rejected, temperature=temperature)
        result['rejected_result'] = result_rejected

        # Extract scores
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
    Pairwise evaluation (Single A vs B)

    Args:
        query: User question
        response_a: Response A
        response_b: Response B
        query_type: Question type (used to select specific prompt)
        temperature: Generation temperature

    Returns:
        Dictionary containing raw_result, score, winner
    """
    result = {}

    try:
        # Select prompt template
        if query_type and query_type in PAIRWISE_MAP:
            prompt_template = PAIRWISE_MAP[query_type]
        else:
            prompt_template = pairwise_prompt_common_template

        # Build prompt
        prompt = prompt_template.format(
            query=query,
            response_a=response_a,
            response_b=response_b
        )

        # Call model
        json_result = get_json_result(prompt, temperature=temperature)

        if json_result is None:
            result['error'] = 'Failed to get valid JSON result'
            result['winner'] = None
            result['score'] = None
            return result

        result['raw_result'] = json_result

        # Parse score
        score = parse_pair_score(json_result)
        result['score'] = score

        # Determine winner
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
    Complete evaluation interface (Verifiable + Pairwise bidirectional)

    This is the main common interface, called by evaluation scripts.

    Args:
        query: User question
        chosen: Chosen response (should be better)
        rejected: Rejected response
        ground_truth: Ground truth (optional, if exists, do fact checking first)
        query_type: Question type (used to select specific pairwise prompt)
        temperature: Generation temperature
        min_score: Judgment threshold

    Returns:
        Dictionary containing verifiable, pairwise_forward, pairwise_backward, final_verdict
    """
    result = {}

    # Part 1: Verifiable (Fact Checking)
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

    # Part 2: Pairwise (Bidirectional Evaluation)
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

    # Determine final result
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
    Compute metrics from verdict list

    Args:
        verdicts: List of final_verdict results
        verifiable_good_count: Count of good verdicts directly from verifiable stage
        verifiable_bad_count: Count of bad verdicts directly from verifiable stage

    Returns:
        Dictionary containing acc_num, same_num, err_num, all_num, acc_rate, same_rate, parse_failed
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
