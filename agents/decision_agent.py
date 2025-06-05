from utils.helpers import OpenRouterClient

class DecisionAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
    
    def make_decision(self, case_data, knowledge_base, subject_analysis, 
                     behavior_analysis, scenario_analysis, result_analysis):
        
        similar_cases = self._find_similar_cases(knowledge_base)
        
        decision_prompt = self._build_decision_prompt(
            case_data, similar_cases, subject_analysis, 
            behavior_analysis, scenario_analysis, result_analysis
        )
        
        # 使用配置的模型参数
        model_config = getattr(self.client, 'model_config', {})
        model = model_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = model_config.get('temperature', 0.1)
        max_tokens = model_config.get('max_tokens', 4000)
        
        final_decision = self.client.chat_completion(decision_prompt, model, temperature, max_tokens)
        
        return final_decision
    
    def _find_similar_cases(self, knowledge_base):
        similar_cases = []
        for item in knowledge_base:
            if item['type'] == 'case' and item['relevance_score'] > 0.7:
                similar_cases.append(item)
        
        return similar_cases[:5]
    
    def _build_decision_prompt(self, case_data, similar_cases, subject_analysis, 
                              behavior_analysis, scenario_analysis, result_analysis):
        
        prompt = f"""
        作为资深的法官和刑法专家，请基于以下所有分析内容，做出最终的定罪量刑决策：
        
        【案件基本信息】
        {case_data}
        
        【主体分析结果】
        {subject_analysis}
        
        【行为分析结果】
        {behavior_analysis}
        
        【情节分析结果】
        {scenario_analysis}
        
        【结果分析结果】
        {result_analysis}
        
        【类似案例参考】
        {self._format_similar_cases(similar_cases)}
        
        请按照以下结构提供最终决策：
        
        # 案件定性分析
        
        ## 1. 罪名认定
        - 主要罪名：
        - 罪名依据：
        - 是否存在竞合：
        
        ## 2. 犯罪构成分析
        - 犯罪主体：
        - 主观方面：
        - 客观方面：
        - 犯罪客体：
        
        ## 3. 量刑情节综合评估
        
        ### 从重情节：
        - 具体情节及影响度评估
        
        ### 从轻情节：
        - 具体情节及影响度评估
        
        ### 减轻情节：
        - 具体情节及影响度评估
        
        ## 4. 量刑建议
        
        ### 基准刑确定：
        - 法定刑幅度：
        - 基准刑选择理由：
        
        ### 量刑调节：
        - 从重处罚调节：
        - 从轻处罚调节：
        - 减轻处罚调节：
        
        ### 最终刑罚建议：
        - 主刑：
        - 附加刑（如适用）：
        - 缓刑适用性分析：
        
        ## 5. 争议点分析
        - 可能存在的争议点：
        - 风险提示：
        - 补强证据建议：
        
        ## 6. 类似案例对比
        - 量刑区间参考：
        - 本案特殊性：
        - 量刑合理性论证：
        
        ## 7. 综合结论
        
        **定罪建议**：[具体罪名]
        
        **量刑建议**：[具体刑期及执行方式]
        
        **主要理由**：[简要说明定罪量刑的核心依据]
        
        **注意事项**：[审理和执行中需要特别关注的问题]
        
        请确保分析逻辑清晰、法条引用准确、量刑建议合理适当。
        """
        
        return prompt
    
    def _format_similar_cases(self, cases):
        if not cases:
            return "暂无高度相似案例"
        
        formatted = []
        for case in cases:
            content = case['content']
            formatted.append(
                f"【案例{content['案件编号']}】\n"
                f"案情：{content['案件概述']}\n"
                f"判决：{content['判决结果']}\n"
                f"适用条文：{content['适用条文']}\n"
                f"相似度：{case['relevance_score']:.2f}"
            )
        
        return "\n\n".join(formatted) 