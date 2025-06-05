from utils.helpers import OpenRouterClient

class ScenarioAnalysisAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
    
    def analyze(self, case_data, knowledge_base):
        relevant_laws = self._filter_scenario_related_laws(knowledge_base)
        relevant_cases = self._filter_scenario_related_cases(knowledge_base)
        
        analysis_prompt = self._build_analysis_prompt(case_data, relevant_laws, relevant_cases)
        
        # 使用配置的模型参数
        model_config = getattr(self.client, 'model_config', {})
        model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = model_config.get('temperature', 0.1)
        max_tokens = model_config.get('max_tokens', 4000)
        
        analysis_result = self.client.chat_completion(analysis_prompt, model, temperature, max_tokens)
        
        return analysis_result
    
    def _filter_scenario_related_laws(self, knowledge_base):
        scenario_keywords = ['情节', '从轻', '从重', '减轻', '加重', '特别严重', '严重', '轻微']
        
        relevant_laws = []
        for item in knowledge_base:
            if item['type'] == 'law':
                content_text = str(item['content'])
                if any(keyword in content_text for keyword in scenario_keywords):
                    relevant_laws.append(item)
        
        return relevant_laws[:5]
    
    def _filter_scenario_related_cases(self, knowledge_base):
        relevant_cases = []
        for item in knowledge_base:
            if item['type'] == 'case':
                relevant_cases.append(item)
        
        return relevant_cases[:3]
    
    def _build_analysis_prompt(self, case_data, laws, cases):
        prompt = f"""
        作为专业的刑法情节分析专家，请根据以下信息进行情节分析：
        
        案件信息：
        {case_data}
        
        相关法律条文：
        {self._format_laws(laws)}
        
        相关案例：
        {self._format_cases(cases)}
        
        请从以下维度进行情节分析：
        
        1. 从重处罚情节识别
        - 犯罪手段特别残忍
        - 犯罪后果特别严重
        - 社会影响特别恶劣
        - 其他从重处罚情节
        
        2. 从轻处罚情节识别
        - 犯罪情节轻微
        - 主动投案自首
        - 积极退赃退赔
        - 真诚悔罪表现
        - 其他从轻处罚情节
        
        3. 减轻处罚情节识别
        - 法定减轻情节
        - 酌定减轻情节
        - 特殊减轻事由
        
        4. 量刑情节综合评估
        - 各类情节的权重分析
        - 情节之间的相互关系
        - 对最终量刑的影响程度
        
        5. 特殊情节考量
        - 是否存在法定从轻减轻情节
        - 是否存在免除处罚情节
        - 缓刑适用条件分析
        
        请详细分析各项情节，并评估其对量刑的具体影响，引用相关法条。
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