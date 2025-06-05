from utils.helpers import OpenRouterClient

class BehaviorAnalysisAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
    
    def analyze(self, case_data, knowledge_base):
        relevant_laws = self._filter_behavior_related_laws(knowledge_base)
        relevant_cases = self._filter_behavior_related_cases(knowledge_base)
        
        analysis_prompt = self._build_analysis_prompt(case_data, relevant_laws, relevant_cases)
        
        # 使用配置的模型参数
        model_config = getattr(self.client, 'model_config', {})
        model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = model_config.get('temperature', 0.1)
        max_tokens = model_config.get('max_tokens', 4000)
        
        analysis_result = self.client.chat_completion(analysis_prompt, model, temperature, max_tokens)
        
        return analysis_result
    
    def _filter_behavior_related_laws(self, knowledge_base):
        behavior_keywords = ['行为', '故意', '过失', '手段', '方法', '犯罪构成', '客观要件']
        
        relevant_laws = []
        for item in knowledge_base:
            if item['type'] == 'law':
                content_text = str(item['content'])
                if any(keyword in content_text for keyword in behavior_keywords):
                    relevant_laws.append(item)
        
        return relevant_laws[:5]
    
    def _filter_behavior_related_cases(self, knowledge_base):
        relevant_cases = []
        for item in knowledge_base:
            if item['type'] == 'case':
                relevant_cases.append(item)
        
        return relevant_cases[:3]
    
    def _build_analysis_prompt(self, case_data, laws, cases):
        prompt = f"""
        作为专业的刑法行为分析专家，请根据以下信息进行行为分析：
        
        案件信息：
        {case_data}
        
        相关法律条文：
        {self._format_laws(laws)}
        
        相关案例：
        {self._format_cases(cases)}
        
        请从以下维度进行行为分析：
        
        1. 犯罪构成要件分析
        - 客观行为要件分析
        - 主观要件（故意/过失）分析
        - 行为与结果的因果关系
        
        2. 行为性质认定
        - 行为的法律性质
        - 是否符合特定罪名构成要件
        - 行为的社会危害性程度
        
        3. 行为方式和手段分析
        - 具体行为方式描述
        - 作案手段的恶劣程度
        - 是否使用特殊工具或方法
        
        4. 行为时空要件
        - 行为发生的时间特征
        - 行为发生的地点特征
        - 时空因素对定罪量刑的影响
        
        5. 行为的连续性和复合性
        - 是否存在连续犯、继续犯
        - 是否涉及数罪并罚
        - 犯罪形态分析（预备、未遂、既遂）
        
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