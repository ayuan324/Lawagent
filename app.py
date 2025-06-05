# -*- coding: utf-8 -*-
import os
import sys

# 确保UTF-8编码环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONUTF8'] = '1'

import streamlit as st
import json
from agents.input_agent import InputAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.analysis_agents.subject_analysis import SubjectAnalysisAgent
from agents.analysis_agents.behavior_analysis import BehaviorAnalysisAgent
from agents.analysis_agents.scenario_analysis import ScenarioAnalysisAgent
from agents.analysis_agents.result_analysis import ResultAnalysisAgent
from agents.decision_agent import DecisionAgent
from utils.helpers import OpenRouterClient

def main():
    st.set_page_config(page_title="法律领域RAG系统", layout="wide")
    
    st.title("🏛️ 法律领域RAG系统")
    st.subheader("刑法案件智能分析与定罪建议")
    
    with st.sidebar:
        st.header("系统配置")
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
        
        # 自定义模型配置
        st.markdown("---")
        st.subheader("🔧 模型配置")
        
        # 常用模型选项
        model_options = {
            "Claude 3.5 Sonnet (推荐)": "anthropic/claude-3.5-sonnet",
            "Claude 3 Haiku (快速)": "anthropic/claude-3-haiku",
            "GPT-4 Turbo": "openai/gpt-4-turbo",
            "GPT-4 Mini": "openai/gpt-4o-mini",
            "Llama 3.1 8B (免费)": "meta-llama/llama-3.1-8b-instruct:free",
            "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
            "自定义模型": "custom"
        }
        
        selected_model_option = st.selectbox(
            "选择模型",
            list(model_options.keys()),
            index=0
        )
        
        if selected_model_option == "自定义模型":
            custom_model = st.text_input(
                "自定义模型名称",
                placeholder="例如: anthropic/claude-3-sonnet"
            )
            model_name = custom_model if custom_model else "anthropic/claude-3.5-sonnet"
        else:
            model_name = model_options[selected_model_option]
        
        # 模型参数配置
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "温度 (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.1,
                help="控制输出的随机性。0=确定性，2=高随机性"
            )
        
        with col2:
            max_tokens = st.slider(
                "最大Token数",
                min_value=100,
                max_value=8000,
                value=4000,
                step=100,
                help="生成文本的最大长度"
            )
        
        # 显示当前配置
        st.info(f"📋 当前配置:\n模型: {model_name}\n温度: {temperature}\nToken: {max_tokens}")
        
        # 保存配置到session state
        st.session_state.model_config = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("测试连接"):
                if openrouter_api_key:
                    with st.spinner("正在测试API连接..."):
                        client = OpenRouterClient(openrouter_api_key)
                        result = client.test_connection()
                        
                        # 显示详细的调试信息
                        st.subheader("🔍 调试信息")
                        st.text(f"API响应: {result}")
                        st.text(f"API Key长度: {len(openrouter_api_key)}")
                        st.text(f"API Key前缀: {openrouter_api_key[:10]}...")
                        st.text(f"系统编码: {sys.getdefaultencoding()}")
                        
                        if "你好" in result or "hello" in result.lower() or "中文" in result:
                            st.success("✅ API连接正常")
                        elif result.startswith("API请求错误"):
                            st.error(f"❌ API请求失败: {result}")
                            st.info("💡 请检查：\n1. API Key是否正确\n2. 网络连接是否正常\n3. OpenRouter账户是否有余额")
                        elif result.startswith("API响应格式错误"):
                            st.error(f"❌ API响应格式错误: {result}")
                            st.info("💡 可能是OpenRouter API格式变化，请联系技术支持")
                        elif result.startswith("编码错误") or result.startswith("解码错误"):
                            st.error(f"❌ 字符编码错误: {result}")
                            st.info("💡 这是UTF-8编码问题，系统正在尝试修复...")
                        elif "所有API调用方法都失败" in result:
                            st.error(f"❌ 所有调用方法都失败")
                            st.info("💡 请检查网络连接和API Key")
                        else:
                            st.warning(f"⚠️ API连接异常，但有响应: {result}")
                            st.info("💡 API可能工作正常，只是响应内容不符合预期")
                else:
                    st.error("请先输入API Key")
        
        with col2:
            if st.button("重建索引"):
                if openrouter_api_key:
                    try:
                        with st.spinner("正在重建知识库索引..."):
                            knowledge_agent = KnowledgeAgent(openrouter_api_key)
                            knowledge_agent.rebuild_index()
                        st.success("✅ 知识库索引重建完成")
                    except Exception as e:
                        st.error(f"❌ 重建索引失败: {str(e)}")
                else:
                    st.error("请先输入API Key")
        
        # 添加手动API测试
        if st.button("🔧 手动API测试"):
            if openrouter_api_key:
                test_manual_api(openrouter_api_key, model_name, temperature, max_tokens)
            else:
                st.error("请先输入API Key")
        
        st.markdown("---")
        st.markdown("### 📝 使用说明")
        st.markdown("""
        1. 输入OpenRouter API Key
        2. 配置模型参数（可选）
        3. 点击"测试连接"验证API
        4. 点击"重建索引"（首次必须）
        5. 输入案件描述进行分析
        """)
        
        st.markdown("### 🔧 故障排除")
        if st.button("查看系统状态"):
            st.info("正在检查系统状态...")
            try:
                import faiss
                import sentence_transformers
                st.success("✅ 核心依赖正常")
                
                import os
                if os.path.exists('data/laws.json'):
                    st.success("✅ 法律数据库存在")
                else:
                    st.error("❌ 法律数据库缺失")
                
                if os.path.exists('data/cases.json'):
                    st.success("✅ 案例数据库存在")
                else:
                    st.error("❌ 案例数据库缺失")
                    
                if os.path.exists('data/knowledge_index.faiss'):
                    st.success("✅ 知识库索引存在")
                else:
                    st.warning("⚠️ 知识库索引需要重建")
                    
            except Exception as e:
                st.error(f"❌ 系统检查失败: {str(e)}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("案件描述输入")
        
        # 添加案例选择
        st.subheader("快速测试案例")
        case_examples = {
            "故意伤害案例": "被告人张某，男，25岁，无前科。因感情纠纷与女友发生争执，情急之下用刀刺伤女友腹部，造成重伤二级。案发后张某主动报警并送被害人就医，积极赔偿医疗费用15万元，取得被害人谅解。",
            "盗窃案例": "被告人李某，男，30岁，有盗窃前科。深夜潜入民宅盗窃现金2万元、金银首饰价值8万元。被发现后逃跑，三日后被抓获。拒不退赃，无悔罪表现。",
            "未成年人抢劫案例": "被告人王某，男，16岁，初中生。伙同他人持刀抢劫同学手机和现金共计2000元。案发后主动投案自首，如实供述犯罪事实，积极退赃，其监护人代为取得被害人谅解。"
        }
        
        selected_example = st.selectbox("选择测试案例（可选）", ["自定义输入"] + list(case_examples.keys()))
        
        if selected_example != "自定义输入":
            case_description = st.text_area(
                "案件描述：",
                value=case_examples[selected_example],
                height=200
            )
        else:
            case_description = st.text_area(
                "请输入案件描述：",
                height=200,
                placeholder="请详细描述案件的具体情况，包括当事人信息、行为过程、结果等..."
            )
        
        if st.button("🚀 开始分析", type="primary"):
            if case_description and openrouter_api_key:
                analyze_case(case_description, openrouter_api_key, col2)
            elif not case_description:
                st.error("❌ 请输入案件描述")
            else:
                st.error("❌ 请输入API Key")
    
    with col2:
        st.header("分析结果")
        if "analysis_result" not in st.session_state:
            st.info("💡 请在左侧输入案件描述并点击分析")

def test_manual_api(api_key, model_name, temperature, max_tokens):
    """手动API测试功能"""
    st.subheader("🔧 手动API调试")
    
    import requests
    import json
    
    # 测试最基本的API调用
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello, please say 你好 in response"}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        st.info(f"正在测试模型: {model_name}")
        st.info(f"参数: temperature={temperature}, max_tokens={max_tokens}")
        
        # 使用UTF-8编码发送数据
        json_data = json.dumps(payload, ensure_ascii=False)
        json_bytes = json_data.encode('utf-8')
        
        response = requests.post(url, headers=headers, data=json_bytes, timeout=30)
        
        st.text(f"响应状态码: {response.status_code}")
        st.text(f"响应头部: {dict(response.headers)}")
        
        if response.status_code == 200:
            response.encoding = 'utf-8'
            data = response.json()
            st.json(data)
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                st.success(f"✅ API调用成功，响应内容: {content}")
            else:
                st.warning("⚠️ API响应成功但格式异常")
        else:
            st.error(f"❌ API调用失败，状态码: {response.status_code}")
            st.text(f"错误内容: {response.text}")
            
            if response.status_code == 401:
                st.error("🔑 API Key无效或已过期")
            elif response.status_code == 402:
                st.error("💰 账户余额不足")
            elif response.status_code == 429:
                st.error("⏰ 请求过于频繁，请稍后重试")
            elif response.status_code >= 500:
                st.error("🔧 OpenRouter服务器错误")
                
    except requests.exceptions.Timeout:
        st.error("⏰ 请求超时，请检查网络连接")
    except requests.exceptions.ConnectionError:
        st.error("🌐 网络连接错误，请检查网络设置")
    except Exception as e:
        st.error(f"❌ 未知错误: {str(e)}")

def analyze_case(case_description, api_key, result_col):
    with result_col:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 获取模型配置
        model_config = st.session_state.get('model_config', {
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.1,
            "max_tokens": 4000
        })
        
        st.info(f"🤖 使用模型: {model_config['model']}")
        
        try:
            status_text.text("🔄 正在处理输入...")
            input_agent = InputAgent(api_key)
            # 传递模型配置
            input_agent.client.model_config = model_config
            processed_input = input_agent.process_input(case_description)
            progress_bar.progress(0.1)
            
            status_text.text("🔍 正在检索知识库...")
            knowledge_agent = KnowledgeAgent(api_key)
            relevant_knowledge = knowledge_agent.retrieve_knowledge(processed_input)
            progress_bar.progress(0.3)
            
            status_text.text("👤 正在进行主体分析...")
            subject_agent = SubjectAnalysisAgent(api_key)
            subject_agent.client.model_config = model_config
            subject_analysis = subject_agent.analyze(processed_input, relevant_knowledge)
            progress_bar.progress(0.4)
            
            status_text.text("⚖️ 正在进行行为分析...")
            behavior_agent = BehaviorAnalysisAgent(api_key)
            behavior_agent.client.model_config = model_config
            behavior_analysis = behavior_agent.analyze(processed_input, relevant_knowledge)
            progress_bar.progress(0.6)
            
            status_text.text("🔍 正在进行情节分析...")
            scenario_agent = ScenarioAnalysisAgent(api_key)
            scenario_agent.client.model_config = model_config
            scenario_analysis = scenario_agent.analyze(processed_input, relevant_knowledge)
            progress_bar.progress(0.7)
            
            status_text.text("📊 正在进行结果分析...")
            result_agent = ResultAnalysisAgent(api_key)
            result_agent.client.model_config = model_config
            result_analysis = result_agent.analyze(processed_input, relevant_knowledge)
            progress_bar.progress(0.8)
            
            status_text.text("🎯 正在生成最终决策...")
            decision_agent = DecisionAgent(api_key)
            decision_agent.client.model_config = model_config
            final_decision = decision_agent.make_decision(
                processed_input,
                relevant_knowledge,
                subject_analysis,
                behavior_analysis,
                scenario_analysis,
                result_analysis
            )
            progress_bar.progress(1.0)
            
            status_text.text("✅ 分析完成！")
            display_results(final_decision, subject_analysis, behavior_analysis, 
                          scenario_analysis, result_analysis)
            
        except Exception as e:
            st.error(f"❌ 分析过程中出现错误: {str(e)}")
            st.info("💡 建议检查API Key是否正确，或稍后重试")

def display_results(final_decision, subject_analysis, behavior_analysis, 
                   scenario_analysis, result_analysis):
    
    st.success("✅ 案件分析完成")
    
    with st.expander("📋 主体分析", expanded=True):
        st.write(subject_analysis)
    
    with st.expander("⚖️ 行为分析"):
        st.write(behavior_analysis)
    
    with st.expander("🔍 情节分析"):
        st.write(scenario_analysis)
    
    with st.expander("📊 结果分析"):
        st.write(result_analysis)
    
    st.subheader("🎯 最终决策建议")
    st.markdown(final_decision)
    
    # 添加下载功能
    if st.button("📥 下载分析报告"):
        report = f"""
# 法律案件分析报告

## 主体分析
{subject_analysis}

## 行为分析  
{behavior_analysis}

## 情节分析
{scenario_analysis}

## 结果分析
{result_analysis}

## 最终决策建议
{final_decision}

---
报告生成时间：{st.session_state.get('analysis_time', '未知')}
        """
        st.download_button(
            label="📄 下载完整报告",
            data=report,
            file_name="法律案件分析报告.md",
            mime="text/markdown"
        )
    
    st.session_state.analysis_result = {
        "final_decision": final_decision,
        "subject_analysis": subject_analysis,
        "behavior_analysis": behavior_analysis,
        "scenario_analysis": scenario_analysis,
        "result_analysis": result_analysis
    }

if __name__ == "__main__":
    main() 