from utils.helpers import OpenRouterClient

class SubjectAnalysisAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
    
    def analyze(self, case_data, knowledge_base):
        relevant_laws = self._filter_subject_related_laws(knowledge_base)
        relevant_cases = self._filter_subject_related_cases(knowledge_base)
        
        analysis_prompt = self._build_analysis_prompt(case_data, relevant_laws, relevant_cases)
        
        # 使用配置的模型参数
        model_config = getattr(self.client, 'model_config', {})
        model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = model_config.get('temperature', 0.1)
        max_tokens = model_config.get('max_tokens', 4000)
        
        analysis_result = self.client.chat_completion(analysis_prompt, model, temperature, max_tokens)
        
        return analysis_result
    
    def _filter_subject_related_laws(self, knowledge_base):
        subject_keywords = ['主体', '年龄', '累犯', '初犯', '未成年', '精神', '责任能力']
        
        relevant_laws = []
        for item in knowledge_base:
            if item['type'] == 'law':
                content_text = str(item['content'])
                if any(keyword in content_text for keyword in subject_keywords):
                    relevant_laws.append(item)
        
        return relevant_laws[:5]
    
    def _filter_subject_related_cases(self, knowledge_base):
        relevant_cases = []
        for item in knowledge_base:
            if item['type'] == 'case':
                relevant_cases.append(item)
        
        return relevant_cases[:3]
    
    def _build_analysis_prompt(self, case_data, laws, cases):
        prompt = f"""
        作为专业的刑法主体分析专家，请根据以下信息进行主体分析：
        
        案件信息：
        {case_data}
        
        相关法律条文：
        {self._format_laws(laws)}
        
        相关案例：
        {self._format_cases(cases)}
        
        请从以下维度进行主体分析：
        
        1. 主体身份认定
        - 当事人基本信息分析
        - 刑事责任能力评估
        - 特殊身份考量（如国家工作人员、未成年人等）
        
        2. 年龄因素分析
        - 行为时年龄确定
        - 年龄对量刑的影响
        - 是否适用特殊程序
        
        3. 前科及累犯分析
        - 前科情况梳理
        - 是否构成累犯
        - 对量刑的影响程度
        
        4. 其他主体因素
        - 精神状态评估
        - 社会危险性评估
        - 其他影响因素
        
        请提供详细的分析过程和结论，并引用相关法条。
        """
        
        return prompt
    
    def _format_laws(self, laws):
        formatted = []
        for law in laws:
            content = law['content']
            formatted.append(f"【{content['条文编号']}】{content['条文内容']}\n说明：{content['解释说明']}")
        return "\n\n".join(formatted)
    
    def _format_cases(self, cases):
        formatted = []
        for case in cases:
            content = case['content']
            formatted.append(f"案例{content['案件编号']}：{content['案件概述']}\n判决：{content['判决结果']}")
        return "\n\n".join(formatted) 