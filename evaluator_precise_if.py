
import io
import time
import logging
from typing import Any, Dict, Optional
from contextlib import redirect_stdout

from tools import get_client_response
from prompts.pointwise_prompts.precise_if import instruction_following_prompt, instruction_following_prompt_tools

logger = logging.getLogger(__name__)


def exec_python(response):
    """从模型响应中提取并执行Python代码"""
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
    Precise IF 评测逻辑

    1. 前置检查：分别对 chosen 和 rejected 进行指令遵循检查（是否满足 constraints）。
    2. 如果前置检查结果不一致（一是一否），直接判定胜负。
    3. 如果结果一致，降级为 Pairwise 比较。
    """
    result = {}

    if not constraints or len(constraints) == 0:
        result['error'] = "No constraints provided"
        result['verdict'] = 'error'
        return result

    constraint = constraints[0]
    rule = constraint.get('rule', '')
    need_code = constraint.get('need_code', False)

    # 原始 query 处理
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

    # 判定逻辑
    c_res = constraint_results['chosen']
    r_res = constraint_results['rejected']

    if c_res == '是' and r_res == '否':
        result['verdict'] = 'chosen_better'
        result['final_verdict'] = 'good'
    elif c_res == '否' and r_res == '是':
        result['verdict'] = 'rejected_better'
        result['final_verdict'] = 'bad'
    else:
        # 一致（都是或都否），降级为 Pairwise
        result['verdict'] = 'equal'

        # Local import to avoid circular dependency
        from evaluator import evaluate_pair

        pairwise_res = evaluate_pair(query, chosen, rejected, temperature=temperature)
        result['pairwise_res'] = pairwise_res
        result['final_verdict'] = pairwise_res.get('final_verdict', 'same')

    return result
