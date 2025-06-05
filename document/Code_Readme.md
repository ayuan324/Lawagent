## 项目代码风格规范文档

### 一、整体风格

* 采用PEP 8标准，保证代码的整洁统一。
* 变量名、函数名使用蛇形命名法（snake_case），类名使用大驼峰命名法（PascalCase）。

### 二、目录结构

```
project_root/
├── app.py                     # Streamlit入口文件
├── agents/
│   ├── input_agent.py         # 输入处理Agent
│   ├── knowledge_agent.py     # 知识库管理Agent
│   ├── analysis_agents/
│   │   ├── subject_analysis.py
│   │   ├── behavior_analysis.py
│   │   ├── scenario_analysis.py
│   │   └── result_analysis.py
│   └── decision_agent.py      # 决策Agent
├── data/
│   ├── laws.json              # 法律条文及说明
│   └── cases.json             # 案例库
├── utils/
│   └── helpers.py             # 工具函数
├── requirements.txt           # 依赖管理
```

### 三、代码书写要求

* 代码清晰、结构化，但是不要撰写注释。
* 统一使用`requirements.txt`管理Python依赖。

