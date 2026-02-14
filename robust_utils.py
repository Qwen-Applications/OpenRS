"""
鲁棒性工具函数：处理Unicode、JSON解析等问题
确保最大程度的容错性和稳定性
"""

import json
import logging
import json5
import re
from json_repair import repair_json

logger = logging.getLogger(__name__)


def fix_surrogates_robust(obj):
    """
    超级鲁棒的 Unicode surrogate 修复
    处理所有可能的编码问题
    """
    if isinstance(obj, str):
        try:
            # 方法1: 先尝试正常编码
            obj.encode('utf-8')
            return obj
        except UnicodeEncodeError:
            try:
                # 方法2: 使用 surrogatepass 处理
                return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
            except Exception:
                try:
                    # 方法3: 使用 ignore 忽略无法编码的字符
                    return obj.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    # 方法4: 最后的兜底，使用 ascii
                    return obj.encode('ascii', errors='ignore').decode('ascii', errors='ignore')
    elif isinstance(obj, dict):
        return {fix_surrogates_robust(k): fix_surrogates_robust(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_surrogates_robust(item) for item in obj]
    else:
        return obj


def safe_json_dumps_robust(data, ensure_ascii=False):
    """
    超级鲁棒的 JSON 序列化
    多层容错，确保一定能成功
    """
    # 第1次尝试：正常序列化
    try:
        return json.dumps(data, ensure_ascii=ensure_ascii)
    except UnicodeEncodeError:
        pass
    
    # 第2次尝试：清理 surrogate 后序列化
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json.dumps(cleaned_data, ensure_ascii=ensure_ascii)
    except Exception:
        pass
    
    # 第3次尝试：使用 ensure_ascii=True（牺牲可读性换取稳定性）
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json.dumps(cleaned_data, ensure_ascii=True)
    except Exception:
        pass
    
    # 第4次尝试：使用 json5（更宽松的解析）
    try:
        cleaned_data = fix_surrogates_robust(data)
        return json5.dumps(cleaned_data)
    except Exception:
        pass
    
    # 最后的兜底：返回字符串表示
    return str(data)


def safe_file_write(filepath, content, mode='w', encoding='utf-8'):
    """
    安全的文件写入，处理所有可能的编码问题
    """
    try:
        # 第1次尝试：正常写入
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
        return True
    except UnicodeEncodeError:
        pass
    
    try:
        # 第2次尝试：清理内容后写入
        cleaned_content = fix_surrogates_robust(content)
        with open(filepath, mode, encoding=encoding) as f:
            f.write(cleaned_content)
        return True
    except Exception:
        pass
    
    try:
        # 第3次尝试：使用 errors='replace'
        with open(filepath, mode, encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except Exception:
        pass
    
    try:
        # 第4次尝试：使用 errors='ignore'
        with open(filepath, mode, encoding=encoding, errors='ignore') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error("文件写入失败: %s, 错误: %s", filepath, e)
        return False


def parse_json_result_robust(response):
    """
    超级鲁棒的 JSON 解析
    多种策略，确保最大成功率
    """
    if isinstance(response, dict):
        return response
    
    if not isinstance(response, str):
        response = str(response)
    
    # 策略1: 提取 ```json ... ``` 代码块
    matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if matches:
        json_str = matches[-1]
    else:
        # 策略2: 提取 ``` ... ``` 代码块
        matches = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if matches:
            json_str = matches[-1]
        else:
            # 策略3: 直接使用整个响应
            json_str = response
    
    # 清理可能的前后空白和特殊字符
    json_str = json_str.strip()
    
    # 尝试多种解析方法
    parsers = [
        # 方法1: 标准 json.loads
        lambda s: json.loads(s),
        # 方法2: json5.loads (更宽松)
        lambda s: json5.loads(s),
        # 方法3: repair_json + json.loads
        lambda s: json.loads(repair_json(s)),
        # 方法4: repair_json + json5.loads
        lambda s: json5.loads(repair_json(s)),
        # 方法5: 清理 surrogate 后再解析
        lambda s: json.loads(fix_surrogates_robust(s)),
        # 方法6: 清理 surrogate + repair_json
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
    
    # 所有方法都失败，抛出详细错误
    raise ValueError(f"JSON解析失败，尝试了{len(parsers)}种方法。最后错误: {last_error}\n原始响应前500字符: {response[:500]}")


def validate_json_structure(json_result, required_keys=None):
    """
    验证 JSON 结构的完整性
    """
    if not isinstance(json_result, dict):
        return False, "结果不是字典类型"
    
    if required_keys:
        missing_keys = [key for key in required_keys if key not in json_result]
        if missing_keys:
            return False, f"缺少必需字段: {missing_keys}"
    
    return True, "验证通过"


    """
    鲁棒的分数解析，增加容错
    """
    # 增加 '熔断' 权重 (虽然 score 已经是 ±100 了，权重依然可加成)
    # 将 '核心' 权重提升至 5，以压倒多个'重要'(2)或'亮点'(1)
    weight_score_map = {'熔断': 100, '核心': 5, '重要': 2, '亮点': 1}
    epsilon = 1e-6
    
    # 验证基本结构
    if not isinstance(json_result, dict):
        raise ValueError(f"json_result 不是字典: {type(json_result)}")
    
    if 'rubric_compares' not in json_result:
        raise ValueError(f"缺少 rubric_compares 字段。可用字段: {list(json_result.keys())}")
    
    rubric_compares = json_result['rubric_compares']
    
    if not isinstance(rubric_compares, list):
        raise ValueError(f"rubric_compares 不是列表: {type(rubric_compares)}")
    
    if len(rubric_compares) == 0:
        raise ValueError("rubric_compares 为空列表")
    
    def validate_score(score, weight):
        """验证分数和权重的有效性"""
        # 增加 '熔断' 类型
        valid_weights = ('熔断', '核心', '重要', '亮点')
        # 允许 ±100 作为一票否决分
        valid_scores = (1, 2, 0, -2, -1, 100, -100)
        return weight in valid_weights and score in valid_scores
    
    weighted_sum, all_weight = [], []
    parse_errors = []
    
    for idx, rubric_compare in enumerate(rubric_compares):
        try:
            # 获取权重，默认为"重要"
            weight = rubric_compare.get('type', '重要')
            
            # 获取分数，尝试多种方式
            if 'score' not in rubric_compare:
                parse_errors.append(f"第{idx}项缺少score字段")
                continue
            
            try:
                score = int(rubric_compare['score'])
            except (ValueError, TypeError) as e:
                parse_errors.append(f"第{idx}项score无法转换为整数: {rubric_compare['score']}")
                continue
            
            # 根据 chosen 字段调整分数方向
            chosen = rubric_compare.get('chosen', 'S')
            if (chosen == 'A' and score < 0) or (chosen == 'B' and score > 0):
                score = -score
            
            # 跳过0分
            if score == 0:
                continue
            
            # 验证分数和权重
            if not validate_score(score, weight):
                parse_errors.append(f"第{idx}项score({score})或weight({weight})无效")
                continue
            
            # 计算加权分数
            weight_value = weight_score_map[weight]
            weighted_sum.append(weight_value * score)
            all_weight.append(weight_value)
            
        except Exception as e:
            parse_errors.append(f"第{idx}项解析异常: {e}")
            continue
    
    # 如果所有项都解析失败
    if len(weighted_sum) == 0:
        error_msg = f"所有rubric_compare项都解析失败。错误: {'; '.join(parse_errors)}"
        raise ValueError(error_msg)
    
    # 计算加权平均分
    weighted_avg_score = sum(weighted_sum) / (sum(all_weight) + epsilon)
    
    # 应用最小分数阈值
    if abs(weighted_avg_score) < min_score:
        weighted_avg_score = 0
    
    return weighted_avg_score


def safe_load_jsonl(filepath):
    """
    鲁棒的 JSONL 文件加载
    跳过损坏的行，记录错误
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
                    # 尝试修复
                    try:
                        data = json.loads(repair_json(line))
                        results.append(data)
                    except Exception as e2:
                        errors.append(f"行{line_num}: {e}")
                        continue
    except Exception as e:
        logger.error("文件读取失败: %s, 错误: %s", filepath, e)
        return []
    
    if errors:
        logger.warning("%s 有 %d 行解析失败", filepath, len(errors))
        if len(errors) <= 5:
            for err in errors:
                logger.warning("   %s", err)
    
    return results


def safe_save_jsonl(data_list, filepath, mode='w'):
    """
    鲁棒的 JSONL 文件保存
    确保每条数据都能成功写入
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
                    logger.warning("数据写入失败: %s", e)
                    continue
    except Exception as e:
        logger.error("文件打开失败: %s, 错误: %s", filepath, e)
        return False
    
    if fail_count > 0:
        logger.warning("%s: 成功%d条, 失败%d条", filepath, success_count, fail_count)
    
    return True
