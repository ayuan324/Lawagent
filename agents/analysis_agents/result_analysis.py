from utils.helpers import OpenRouterClient

class ResultAnalysisAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
    
    def analyze(self, case_data, knowledge_base):
        relevant_laws = self._filter_result_related_laws(knowledge_base)
        relevant_cases = self._filter_result_related_cases(knowledge_base)
        
        analysis_prompt = self._build_analysis_prompt(case_data, relevant_laws, relevant_cases)
        
        # 使用配置的模型参数
        model_config = getattr(self.client, 'model_config', {})
        model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = model_config.get('temperature', 0.1)
        max_tokens = model_config.get('max_tokens', 4000)
        
        analysis_result = self.client.chat_completion(analysis_prompt, model, temperature, max_tokens)
        
        return analysis_result
    
    def _filter_result_related_laws(self, knowledge_base):
        result_keywords = ['结果', '后果', '损失', '伤害', '死亡', '财产', '精神', '社会危害']
        
        relevant_laws = []
        for item in knowledge_base:
            if item['type'] == 'law':
                content_text = str(item['content'])
                if any(keyword in content_text for keyword in result_keywords):
                    relevant_laws.append(item)
        
        return relevant_laws[:5]
    
    def _filter_result_related_cases(self, knowledge_base):
        relevant_cases = []
        for item in knowledge_base:
            if item['type'] == 'case':
                relevant_cases.append(item)
        
        return relevant_cases[:3]
    
    def _build_analysis_prompt(self, case_data, laws, cases):
        prompt = f"""
        作为专业的刑法结果分析专家，请根据以下信息进行结果分析：
        
        案件信息：
        {case_data}
        
        相关法律条文：
        {self._format_laws(laws)}
        
        相关案例：
        {self._format_cases(cases)}
        
        请从以下维度进行结果分析：
        
        1. 直接后果分析
        - 人身伤害后果
        - 财产损失后果
        - 其他直接损害后果
        - 后果的严重程度评估
        
        2. 间接后果分析
        - 对被害人的长期影响
        - 对家庭和社会的影响
        - 心理创伤和精神损害
        - 经济损失的连带影响
        
        3. 社会危害性评估
        - 对社会秩序的冲击
        - 对公共安全的威胁
        - 对社会风气的影响
        - 示范效应和负面导向
        
        4. 因果关系认定
        - 行为与结果的直接因果关系
        - 介入因素的影响
        - 因果关系链条的完整性
        - 责任范围的界定
        
        5. 量刑参考依据
        - 结果严重程度对量刑的影响
        - 与类似案件的对比分析
        - 法定刑档的适用建议
        - 具体量刑区间的建议
        
        6. 民事责任考量
        - 经济损失的赔偿责任
        - 精神损害赔偿
        - 其他民事责任
        
        请详细分析各项结果，并评估其对定罪量刑的具体影响，引用相关法条和类似案例。
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