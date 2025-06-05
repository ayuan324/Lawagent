import re
import json
from utils.helpers import OpenRouterClient

class InputAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
        
    def process_input(self, raw_input):
        cleaned_input = self._clean_text(raw_input)
        structured_input = self._extract_key_information(cleaned_input)
        return structured_input
    
    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】\-\.]', '', text)
        return text
    
    def _extract_key_information(self, text):
        prompt = f"""请从以下案件描述中提取关键信息，并按照以下JSON格式输出：

案件描述：{text}

请输出JSON格式：
{{
    "主体信息": {{
        "姓名": "当事人姓名",
        "年龄": "年龄信息",
        "前科情况": "是否有前科或累犯情况",
        "其他身份特征": "其他相关身份信息"
    }},
    "行为描述": {{
        "主要行为": "核心犯罪行为描述",
        "行为时间": "行为发生的时间",
        "行为地点": "行为发生的地点",
        "行为方式": "具体的行为方式和手段"
    }},
    "结果情况": {{
        "直接后果": "行为造成的直接后果",
        "损失程度": "造成的损失或伤害程度",
        "社会影响": "对社会造成的影响"
    }},
    "其他情节": {{
        "从轻情节": "可能的从轻处罚情节",
        "从重情节": "可能的从重处罚情节",
        "特殊情况": "其他需要考虑的特殊情况"
    }}
}}"""
        
        try:
            # 使用配置的模型参数
            model_config = getattr(self.client, 'model_config', {})
            model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
            temperature = model_config.get('temperature', 0.1)
            max_tokens = model_config.get('max_tokens', 4000)
            
            response = self.client.chat_completion(prompt, model, temperature, max_tokens)
            
            if response.startswith("API请求错误") or response.startswith("API响应格式错误") or response.startswith("编码错误") or response.startswith("未知错误"):
                return {
                    "原始输入": text,
                    "处理状态": f"处理失败: {response}",
                    "主体信息": {"姓名": "未知", "年龄": "未知", "前科情况": "未知", "其他身份特征": ""},
                    "行为描述": {"主要行为": text[:100], "行为时间": "未知", "行为地点": "未知", "行为方式": ""},
                    "结果情况": {"直接后果": "待分析", "损失程度": "待分析", "社会影响": "待分析"},
                    "其他情节": {"从轻情节": "", "从重情节": "", "特殊情况": ""}
                }
            
            structured_data = json.loads(response)
            return structured_data
            
        except json.JSONDecodeError:
            return {
                "原始输入": text,
                "处理状态": "JSON解析失败，使用原始输入",
                "主体信息": {"姓名": "未知", "年龄": "未知", "前科情况": "未知", "其他身份特征": ""},
                "行为描述": {"主要行为": text[:100], "行为时间": "未知", "行为地点": "未知", "行为方式": ""},
                "结果情况": {"直接后果": "待分析", "损失程度": "待分析", "社会影响": "待分析"},
                "其他情节": {"从轻情节": "", "从重情节": "", "特殊情况": ""}
            }
        except Exception as e:
            return {
                "原始输入": text,
                "处理状态": f"处理异常: {str(e)}",
                "主体信息": {"姓名": "未知", "年龄": "未知", "前科情况": "未知", "其他身份特征": ""},
                "行为描述": {"主要行为": text[:100], "行为时间": "未知", "行为地点": "未知", "行为方式": ""},
                "结果情况": {"直接后果": "待分析", "损失程度": "待分析", "社会影响": "待分析"},
                "其他情节": {"从轻情节": "", "从重情节": "", "特殊情况": ""}
            } 