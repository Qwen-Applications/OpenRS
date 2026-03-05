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
