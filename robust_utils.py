"""
Robust utility functions: handling Unicode, JSON parsing, etc.
Ensuring maximum fault tolerance and stability.
"""

import json
import logging
import json5
import re
from json_repair import repair_json

logger = logging.getLogger(__name__)


def fix_surrogates_robust(obj):
    """
    Super robust Unicode surrogate repair.
    Handles all possible encoding issues.
    """
    if isinstance(obj, str):
        try:
            # Method 1: Try normal encoding first
            obj.encode('utf-8')
            return obj
        except UnicodeEncodeError:
            try:
                # Method 2: Use surrogatepass
                return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
            except Exception:
                try:
                    # Method 3: Use ignore to skip unencodable characters
                    return obj.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    # Method 4: Last resort, use ascii
                    return obj.encode('ascii', errors='ignore').decode('ascii', errors='ignore')
    elif isinstance(obj, dict):
        return {fix_surrogates_robust(k): fix_surrogates_robust(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_surrogates_robust(item) for item in obj]
    else:
        return obj


def safe_json_dumps_robust(data, ensure_ascii=False):
    """
    Super robust JSON serialization.
    Multi-layer fault tolerance to ensure success.
    """
    # Attempt 1: Normal serialization
    try:
        return json.dumps(data, ensure_ascii=ensure_ascii)
    except UnicodeEncodeError:
        pass
    
    # Attempt 2: Clean surrogates then serialize
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json.dumps(cleaned_data, ensure_ascii=ensure_ascii)
    except Exception:
        pass
    
    # Attempt 3: Use ensure_ascii=True (sacrifice readability for stability)
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json.dumps(cleaned_data, ensure_ascii=True)
    except Exception:
        pass
    
    # Attempt 4: Use json5 (looser parsing)
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json5.dumps(cleaned_data)
    except Exception:
        pass
    
    # Last resort: Return string representation
    return str(data)


def safe_file_write(filepath, content, mode='w', encoding='utf-8'):
    """
    Safe file write, handling all possible encoding issues.
    """
    try:
        # Attempt 1: Normal write
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
        return True
    except UnicodeEncodeError:
        pass
    
    try:
        # Attempt 2: Clean content then write
        cleaned_content = fix_surrogates_robust(content)
        with open(filepath, mode, encoding=encoding) as f:
            f.write(cleaned_content)
        return True
    except Exception:
        pass
    
    try:
        # Attempt 3: Use errors='replace'
        with open(filepath, mode, encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except Exception:
        pass
    
    try:
        # Attempt 4: Use errors='ignore'
        with open(filepath, mode, encoding=encoding, errors='ignore') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error("File write failed: %s, error: %s", filepath, e)
        return False


def parse_json_result_robust(response):
    """
    Super robust JSON parsing.
    Multiple strategies to ensure maximum success rate.
    """
    if isinstance(response, dict):
        return response
    
    if not isinstance(response, str):
        response = str(response)
    
    # Strategy 1: Extract ```json ... ``` code block
    matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if matches:
        json_str = matches[-1]
    else:
        # Strategy 2: Extract ``` ... ``` code block
        matches = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if matches:
            json_str = matches[-1]
        else:
            # Strategy 3: Use the entire response directly
            json_str = response
    
    # Clean potential leading/trailing whitespace and special characters
    json_str = json_str.strip()
    
    # Try multiple parsing methods
    parsers = [
        # Method 1: Standard json.loads
        lambda s: json.loads(s),
        # Method 2: json5.loads (looser)
        lambda s: json5.loads(s),
        # Method 3: repair_json + json.loads
        lambda s: json.loads(repair_json(s)),
        # Method 4: repair_json + json5.loads
        lambda s: json5.loads(repair_json(s)),
        # Method 5: Clean surrogates then parse
        lambda s: json.loads(fix_surrogates_robust(s)),
        # Method 6: Clean surrogates + repair_json
        lambda s: json.loads(repair_json(fix_surrogates_robust(s))),
    ]
    
    last_error = None
    for i, parser in enumerate(parsers):
        try:
            result = parser(json_str)
            if isinstance(result, dict):
                return result
        except Exception as e:
            last_error = e
            continue
    
    # All methods failed, raise detailed error
    raise ValueError(f"JSON parsing failed, tried {len(parsers)} methods. Last error: {last_error}\nFirst 500 chars of original response: {response[:500]}")


def validate_json_structure(json_result, required_keys=None):
    """
    Validate the integrity of JSON structure.
    """
    if not isinstance(json_result, dict):
        return False, "Result is not a dictionary"
    
    if required_keys:
        missing_keys = [key for key in required_keys if key not in json_result]
        if missing_keys:
            return False, f"Missing required fields: {missing_keys}"
    
    return True, "Validation passed"


    """
    Robust score parsing, increasing fault tolerance.
    """
    # Add 'Meltdown' weight (although score is already ±100, weight can still be added)
    # Increase 'Core' weight to 5 to outweigh multiple 'Important'(2) or 'Highlight'(1)
    weight_score_map = {'熔断': 100, '核心': 5, '重要': 2, '亮点': 1}
    epsilon = 1e-6
    
    # Validate basic structure
    if not isinstance(json_result, dict):
        raise ValueError(f"json_result is not a dict: {type(json_result)}")
    
    if 'rubric_compares' not in json_result:
        raise ValueError(f"Missing rubric_compares field. Available fields: {list(json_result.keys())}")
    
    rubric_compares = json_result['rubric_compares']
    
    if not isinstance(rubric_compares, list):
        raise ValueError(f"rubric_compares is not a list: {type(rubric_compares)}")
    
    if len(rubric_compares) == 0:
        raise ValueError("rubric_compares is an empty list")
    
    def validate_score(score, weight):
        """Validate validity of score and weight"""
        # Add 'Meltdown' type
        valid_weights = ('熔断', '核心', '重要', '亮点')
        # Allow ±100 as a veto score
        valid_scores = (1, 2, 0, -2, -1, 100, -100)
        return weight in valid_weights and score in valid_scores
    
    weighted_sum, all_weight = [], []
    parse_errors = []
    
    for idx, rubric_compare in enumerate(rubric_compares):
        try:
            # Get weight, default to "Important"
            weight = rubric_compare.get('type', '重要')
            
            # Get score, try multiple ways
            if 'score' not in rubric_compare:
                parse_errors.append(f"Item {idx} missing score field")
                continue
            
            try:
                score = int(rubric_compare['score'])
            except (ValueError, TypeError) as e:
                parse_errors.append(f"Item {idx} score cannot be converted to int: {rubric_compare['score']}")
                continue
            
            # Adjust score direction based on chosen field
            chosen = rubric_compare.get('chosen', 'S')
            if (chosen == 'A' and score < 0) or (chosen == 'B' and score > 0):
                score = -score
            
            # Skip 0 score
            if score == 0:
                continue
            
            # Validate score and weight
            if not validate_score(score, weight):
                parse_errors.append(f"Item {idx} score({score}) or weight({weight}) invalid")
                continue
            
            # Calculate weighted score
            weight_value = weight_score_map[weight]
            weighted_sum.append(weight_value * score)
            all_weight.append(weight_value)
            
        except Exception as e:
            parse_errors.append(f"Item {idx} parsing exception: {e}")
            continue
    
    # If all items failed parsing
    if len(weighted_sum) == 0:
        error_msg = f"All rubric_compare items failed parsing. Errors: {'; '.join(parse_errors)}"
        raise ValueError(error_msg)
    
    # Calculate weighted average score
    weighted_avg_score = sum(weighted_sum) / (sum(all_weight) + epsilon)
    
    # Apply minimum score threshold
    if abs(weighted_avg_score) < min_score:
        weighted_avg_score = 0
    
    return weighted_avg_score


def safe_load_jsonl(filepath):
    """
    Robust JSONL file loading.
    Skip corrupted lines, log errors.
    """
    results = []
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    results.append(data)
                except Exception as e:
                    # Try to repair
                    try:
                        data = json.loads(repair_json(line))
                        results.append(data)
                    except Exception as e2:
                        errors.append(f"Line {line_num}: {e}")
                        continue
    except Exception as e:
        logger.error("File read failed: %s, error: %s", filepath, e)
        return []
    
    if errors:
        logger.warning("%s has %d lines failed parsing", filepath, len(errors))
        if len(errors) <= 5:
            for err in errors:
                logger.warning("   %s", err)
    
    return results


def safe_save_jsonl(data_list, filepath, mode='w'):
    """
    Robust JSONL file saving.
    Ensure every data item is written successfully.
    """
    success_count = 0
    fail_count = 0
    
    try:
        with open(filepath, mode, encoding='utf-8', errors='replace') as f:
            for data in data_list:
                try:
                    json_str = safe_json_dumps_robust(data, ensure_ascii=False)
                    f.write(json_str + '\n')
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.warning("Data write failed: %s", e)
                    continue
    except Exception as e:
        logger.error("File open failed: %s, error: %s", filepath, e)
        return False
    
    if fail_count > 0:
        logger.warning("%s: Success %d, Failed %d", filepath, success_count, fail_count)
    
    return True
