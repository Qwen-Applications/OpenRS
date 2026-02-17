
import os
import re
import json
import json5
import logging
from json_repair import repair_json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

# ============== OpenAI Client Setup ==============
# Configure via environment variables:
#   OPENAI_BASE_URL  - API base URL (default: http://localhost:8000/v1)
#   OPENAI_API_KEY   - API key (default: EMPTY)
#   OPENAI_MODEL_NAME - Model name (default: default)

openai_api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
model_path = os.environ.get("OPENAI_MODEL_NAME", "default")
client = OpenAI(api_key=api_key, base_url=openai_api_base)


# ============== File I/O ==============

def save_jsonl(datas, file_name):
    """Save JSONL file (Enhanced: supports Unicode fault tolerance)"""
    try:
        with open(file_name, 'w', encoding='utf-8') as fin:
            for data in datas:
                fin.write(json.dumps(data, ensure_ascii=False) + '\n')
    except (UnicodeEncodeError, Exception) as e:
        logger.warning("save_jsonl encountered error, using robust version: %s", e)
        from robust_utils import safe_save_jsonl
        return safe_save_jsonl(datas, file_name, mode='w')


def load_jsonl(file_path):
    """Load JSONL file (supports automatic repair of broken lines)"""
    try:
        from json_repair import repair_json
        has_repair = True
    except ImportError:
        has_repair = False
    data = []
    errors = []

    logger.info("Loading: %s", file_path)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    if has_repair:
                        try:
                            fixed_line = repair_json(line)
                            data.append(json.loads(fixed_line))
                            logger.warning("Line %d repaired automatically.", i)
                        except Exception:
                            errors.append(f"Line {i}: Repair failed")
                    else:
                        errors.append(f"Line {i}: JSON Error (json_repair module not found)")

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return []
    except Exception as e:
        logger.error("Error reading file: %s", e)
        return []
    if errors:
        logger.warning("Skipped %d invalid lines.", len(errors))
        if len(errors) > 0:
            logger.warning("Sample errors: %s", errors[:3])

    return data


# ============== JSON Parsing ==============

def parse_json_result(response) -> dict:
    """Parse JSON result from model response"""
    try:
        # Step 1: Extract Markdown code block
        matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if len(matches) == 0:
            raise ValueError("no json block found for pair rm")

        json_str = matches[-1]

        # Step 2: Try standard json first (fastest)
        try:
            return json.loads(json_str)
        except Exception:
            pass

        # Step 3: Try json5 (handle trailing commas, etc.)
        try:
            return json5.loads(json_str)
        except Exception:
            pass
        try:
            json_repaired = repair_json(json_str)
            return json5.loads(json_repaired)
        except Exception:
            raise ValueError("Repair failed")

    except Exception:
        # Fallback logic
        from robust_utils import parse_json_result_robust
        return parse_json_result_robust(response)



def parse_pair_score(json_result, min_score=0):
    """
    Parse score (Veto version):
    If there is a fatal error, return the total fatal score directly (ignoring all other dimensions).
    If there are no fatal errors, calculate the weighted average score of other dimensions.
    """
    try:
        weight_map = {'核心': 5, '重要': 2, '亮点': 1}
        epsilon = 1e-6

        if 'rubric_compares' not in json_result:
            return 0

        rubric_compares = json_result['rubric_compares']

        fatal_score_sum = 0      # Sum of scores specifically for fatal errors
        has_fatal_error = False  # Flag to indicate if a fatal error has occurred

        normal_weighted_sum = 0  # Weighted total score for normal dimensions
        normal_total_weight = 0  # Total weight for normal dimensions

        for item in rubric_compares:
            weight_type = item.get('type', '重要')
            raw_score = int(item['score'])

            if raw_score == 0:
                continue

            # Unified sign logic (Positive for A win, Negative for B win)
            abs_score = abs(raw_score)
            if item['chosen'] == 'A':
                final_score = abs_score
            elif item['chosen'] == 'B':
                final_score = -abs_score
            else:
                final_score = 0

            if weight_type == '硬伤':
                has_fatal_error = True
                fatal_score_sum += final_score

            elif weight_type in weight_map:
                w = weight_map[weight_type]
                normal_weighted_sum += final_score * w
                normal_total_weight += w

        if has_fatal_error:
            final_result = fatal_score_sum
        else:
            if normal_total_weight == 0:
                final_result = 0
            else:
                final_result = normal_weighted_sum / (normal_total_weight + epsilon)

        if abs(final_result) < min_score:
            return 0

        return final_result

    except Exception as e:
        logger.error("Error parsing score: %s", e)
        return 0


# ============== Model API ==============

def get_client_response(prompt, temperature=0.0, top_p=1.0, seed=1024):
    """Call OpenAI compatible API to get model response"""

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(10), retry_error_callback=lambda x: None)
    def _impl():
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                frequency_penalty=0,
                presence_penalty=0,
                max_tokens=8192,
            )
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            return str(response)
        except Exception as e:
            logger.error("OpenAI Client Error: %s", e)
            raise e
    return _impl()
