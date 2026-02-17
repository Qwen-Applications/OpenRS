
import io
import time
import logging
from typing import Any, Dict, Optional
from contextlib import redirect_stdout

from tools import get_client_response
from prompts.pointwise_prompts.precise_if import instruction_following_prompt, instruction_following_prompt_tools

logger = logging.getLogger(__name__)


def exec_python(response):
    """Extract and execute Python code from model response"""
    if '=====' not in response:
        return None

    python_code = response.split('=====')[-1].strip()
    if python_code.startswith('```python'):
        python_code = python_code[9:-3]
    elif python_code.startswith('```'):
        python_code = python_code[3:-3]

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            exec(python_code, {'__builtins__': __builtins__}, {})
        code_result = output_buffer.getvalue().strip()
        if code_result == 'True': return '是'
        if code_result == 'False': return '否'

        code_result_eval = eval(code_result)
        if code_result_eval is True: return '是'
        if code_result_eval is False: return '否'

    except Exception:
        pass

    return None


def evaluate_precise_if(
    query: str,
    chosen: str,
    rejected: str,
    constraints: Optional[list],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Precise IF evaluation logic

    1. Pre-check: Check instruction following for chosen and rejected separately (satisfy constraints).
    2. If pre-check results are inconsistent (one yes one no), determine winner directly.
    3. If results are consistent, downgrade to Pairwise comparison.
    """
    result = {}

    if not constraints or len(constraints) == 0:
        result['error'] = "No constraints provided"
        result['verdict'] = 'error'
        return result

    constraint = constraints[0]
    rule = constraint.get('rule', '')
    need_code = constraint.get('need_code', False)

    # Original query processing
    ori_query = query[:-len(rule)] if query.endswith(rule) else query

    constraint_results = {}
    constraint_judges = {}

    responses = {'chosen': chosen, 'rejected': rejected}

    for key, response in responses.items():
        if not need_code:
            prompt = instruction_following_prompt.format(query=ori_query, response=response, constraint=rule)

            res_text = None
            for _ in range(3):
                res = get_client_response(prompt, temperature=temperature)
                if res and res.strip().split()[-1] in ['是', '否']:
                    res_text = res.strip().split()[-1]
                    break
                time.sleep(1)

            constraint_results[key] = res_text if res_text else '否'
            constraint_judges[key] = res

        else:
            prompt = instruction_following_prompt_tools.format(query=ori_query, response=response, constraint=rule)

            res_text = None
            res = None
            for _ in range(3):
                res = get_client_response(prompt, temperature=temperature)
                code_res = exec_python(res)
                if code_res in ['是', '否']:
                    res_text = code_res
                    break
                time.sleep(1)

            constraint_results[key] = res_text if res_text else '否'
            constraint_judges[key] = res

    result['constraint_results'] = constraint_results
    result['constraint_judges'] = constraint_judges

    # Judgment logic
    c_res = constraint_results['chosen']
    r_res = constraint_results['rejected']

    if c_res == '是' and r_res == '否':
        result['verdict'] = 'chosen_better'
        result['final_verdict'] = 'good'
    elif c_res == '否' and r_res == '是':
        result['verdict'] = 'rejected_better'
        result['final_verdict'] = 'bad'
    else:
        # Consistent (both yes or both no), downgrade to Pairwise
        result['verdict'] = 'equal'

        # Local import to avoid circular dependency
        from evaluator import evaluate_pair

        pairwise_res = evaluate_pair(query, chosen, rejected, temperature=temperature)
        result['pairwise_res'] = pairwise_res
        result['final_verdict'] = pairwise_res.get('final_verdict', 'same')

    return result
