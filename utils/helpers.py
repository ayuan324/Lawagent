import requests
import json
import os
from typing import Dict, List, Any
import urllib3
from urllib.parse import quote

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Law RAG System",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8",
            "Accept-Charset": "utf-8"
        })
        # 设置session的编码
        self.session.encoding = 'utf-8'
    
    def test_connection(self) -> str:
        """测试API连接 - 使用简化的方法"""
        return self.chat_completion("Hello, please respond in Chinese: 你好", "meta-llama/llama-3.1-8b-instruct:free")
    
    def chat_completion(self, prompt: str, model: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.1, max_tokens: int = 4000) -> str:
        url = f"{self.base_url}/chat/completions"
        
        # 确保prompt是UTF-8字符串
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8', errors='replace')
        
        # 确保所有字符串都是正确编码的
        prompt = str(prompt)
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            # 使用ensure_ascii=False和指定编码来处理JSON
            json_data = json.dumps(payload, ensure_ascii=False)
            
            # 显式编码为UTF-8字节
            json_bytes = json_data.encode('utf-8')
            
            # 使用data参数发送字节数据
            response = self.session.post(
                url, 
                data=json_bytes,
                timeout=60
            )
            
            response.raise_for_status()
            
            # 确保响应使用UTF-8解码
            response.encoding = 'utf-8' #  requests会自动根据header猜测，但显式设置更保险
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                return str(content)
            else:
                return "API响应格式错误: 未找到choices字段"
                
        except requests.exceptions.HTTPError as e:
            error_message = f"API请求错误: HTTP {e.response.status_code}, 响应: {e.response.text}"
            if e.response.status_code == 401:
                error_message = "API请求错误: API Key无效或已过期"
            elif e.response.status_code == 402:
                error_message = "API请求错误: 账户余额不足"
            elif e.response.status_code == 429:
                error_message = "API请求错误: 请求过于频繁，请稍后重试"
            return error_message
        except requests.exceptions.Timeout:
            return "API请求错误: 请求超时"
        except requests.exceptions.ConnectionError:
            return "API请求错误: 网络连接失败"
        except json.JSONDecodeError as e:
            return f"API响应格式错误: JSON解析失败 - {str(e)}"
        except UnicodeEncodeError as e:
            return f"编码错误: {str(e)} - 请检查输入文本的字符编码"
        except UnicodeDecodeError as e:
            return f"解码错误: {str(e)} - 服务器响应编码问题"
        except Exception as e:
            return f"未知错误: {str(e)}"

def ensure_directory_exists(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_json_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def save_json_file(data: Dict[str, Any], file_path: str):
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def format_case_for_display(case_data: Dict[str, Any]) -> str:
    if not case_data:
        return "无案件数据"
    
    formatted_parts = []
    
    for key, value in case_data.items():
        if isinstance(value, dict):
            formatted_parts.append(f"**{key}:**")
            for sub_key, sub_value in value.items():
                formatted_parts.append(f"  - {sub_key}: {sub_value}")
        else:
            formatted_parts.append(f"**{key}:** {value}")
    
    return "\n".join(formatted_parts)

def extract_keywords_from_text(text: str) -> List[str]:
    import re
    
    # 确保文本是UTF-8编码
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', str(text))
    words = text.split()
    
    keywords = []
    for word in words:
        if len(word) >= 2 and word not in ['的', '了', '是', '在', '有', '和', '与', '或', '但', '而', '就', '都', '为', '被', '把', '到', '从', '给', '对', '向', '由', '用', '以', '及', '等', '如', '像', '比', '按', '根据', '通过', '经过', '关于', '针对', '面对']:
            keywords.append(word)
    
    return list(set(keywords))

def calculate_text_similarity(text1: str, text2: str) -> float:
    keywords1 = set(extract_keywords_from_text(text1))
    keywords2 = set(extract_keywords_from_text(text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0

def validate_case_data(case_data: Dict[str, Any]) -> bool:
    required_fields = ['主体信息', '行为描述', '结果情况']
    
    if not isinstance(case_data, dict):
        return False
    
    for field in required_fields:
        if field not in case_data:
            return False
        
        if not case_data[field] or (isinstance(case_data[field], dict) and not any(case_data[field].values())):
            return False
    
    return True

def clean_analysis_text(text: str) -> str:
    # 确保文本是UTF-8编码
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
        
    lines = str(text).split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('##') and not line.startswith('**'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def get_law_article_number(text: str) -> List[str]:
    import re
    
    # 确保文本是UTF-8编码
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    pattern = r'第[一二三四五六七八九十百千万\d]+条'
    matches = re.findall(pattern, str(text))
    
    return list(set(matches)) 