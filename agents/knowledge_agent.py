import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils.helpers import OpenRouterClient

# 定义本地模型路径 (如果使用本地模型，请取消注释并设置正确路径)
# LOCAL_MODEL_PATH = "E:/外快/1 LAWAGENT/local_models/all-MiniLM-L6-v2"
LOCAL_MODEL_PATH = None # 默认从HuggingFace Hub下载

class KnowledgeAgent:
    def __init__(self, api_key):
        self.client = OpenRouterClient(api_key)
        
        model_name_or_path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH) else 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"加载SentenceTransformer模型: {model_name_or_path}")
        try:
            self.encoder = SentenceTransformer(model_name_or_path, device='cpu')
        except Exception as e:
            print(f"加载SentenceTransformer模型失败: {e}")
            print("如果网络问题持续，请尝试手动下载模型并配置LOCAL_MODEL_PATH")
            raise
            
        self.index_path = 'data/knowledge_index.faiss'
        self.metadata_path = 'data/knowledge_metadata.json'
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        self._load_knowledge_base()
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                # 验证维度是否匹配
                if self.index.d != self.dimension:
                    print(f"警告: 索引维度 ({self.index.d})与模型嵌入维度 ({self.dimension}) 不匹配. 将重建索引.")
                    self.rebuild_index()
                else:
                    print(f"索引已加载: {self.index.ntotal} 个文档")
            except Exception as e:
                print(f"加载索引失败: {e}，将重建索引")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.rebuild_index()
        else:
            print("索引文件不存在或元数据文件不存在，将创建新索引")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.rebuild_index()
    
    def _load_knowledge_base(self):
        try:
            with open('data/laws.json', 'r', encoding='utf-8') as f:
                self.laws_data = json.load(f)
        except FileNotFoundError:
            print("警告: data/laws.json 文件不存在，将使用空法律数据")
            self.laws_data = []
        except json.JSONDecodeError:
            print("警告: data/laws.json 文件格式错误，将使用空法律数据")
            self.laws_data = []
        
        try:
            with open('data/cases.json', 'r', encoding='utf-8') as f:
                self.cases_data = json.load(f)
        except FileNotFoundError:
            print("警告: data/cases.json 文件不存在，将使用空案例数据")
            self.cases_data = []
        except json.JSONDecodeError:
            print("警告: data/cases.json 文件格式错误，将使用空案例数据")
            self.cases_data = []
    
    def rebuild_index(self):
        if not hasattr(self, 'laws_data') or not hasattr(self, 'cases_data'):
            print("错误: 知识库数据未加载，无法重建索引")
            return
        
        all_texts = []
        metadata = []
        
        for law in self.laws_data:
            if not isinstance(law, dict):
                print(f"警告: 跳过格式不正确的法律条目: {law}")
                continue
            text = f"{law.get('条文编号', '')} {law.get('条文内容', '')} {law.get('解释说明', '')}"
            all_texts.append(text)
            metadata.append({
                'type': 'law',
                'id': law.get('条文编号', ''),
                'content': law
            })
        
        for case in self.cases_data:
            if not isinstance(case, dict):
                print(f"警告: 跳过格式不正确的案例条目: {case}")
                continue
            text = f"{case.get('案件概述', '')} {case.get('判决结果', '')} {case.get('适用条文', '')}"
            all_texts.append(text)
            metadata.append({
                'type': 'case',
                'id': case.get('案件编号', ''),
                'content': case
            })
        
        if not all_texts:
            print("警告: 没有可索引的文本数据，索引将为空")
            self.metadata = []
            self.index = faiss.IndexFlatIP(self.dimension)
            os.makedirs('data', exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            print("空的索引已创建.")
            return
        
        print(f"开始编码 {len(all_texts)} 个文档...")
        try:
            embeddings = self.encoder.encode(all_texts, show_progress_bar=True)
        except Exception as e:
            print(f"文本编码失败: {e}")
            print("请检查PyTorch和sentence-transformers的安装以及文本数据.")
            return
            
        embeddings = np.array(embeddings).astype('float32')
        
        if embeddings.shape[0] > 0 and embeddings.shape[1] != self.dimension:
            print(f"错误: 嵌入维度 ({embeddings.shape[1]}) 与模型维度 ({self.dimension}) 不匹配. 索引重建失败.")
            return
            
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        os.makedirs('data', exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.metadata = metadata
        print(f"成功重建索引: {self.index.ntotal} 个文档")
    
    def retrieve_knowledge(self, query_data, top_k=10):
        if isinstance(query_data, dict):
            query_text = self._dict_to_text(query_data)
        else:
            query_text = str(query_data)
        
        if not hasattr(self, 'index') or not hasattr(self, 'metadata') or self.index.ntotal == 0:
            print("警告: 知识库索引为空或未初始化")
            return []
        
        query_embedding = self.encoder.encode([query_text])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        k_search = min(top_k, self.index.ntotal)
        if k_search == 0:
            return []
            
        scores, indices = self.index.search(query_embedding, k_search)
        
        retrieved_knowledge = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['relevance_score'] = float(score)
                retrieved_knowledge.append(item)
        
        return retrieved_knowledge
    
    def add_new_case(self, case_data):
        self.cases_data.append(case_data)
        
        os.makedirs('data', exist_ok=True)
        with open('data/cases.json', 'w', encoding='utf-8') as f:
            json.dump(self.cases_data, f, ensure_ascii=False, indent=2)
        
        self.rebuild_index()
    
    def update_law_content(self, law_id, new_content):
        for law in self.laws_data:
            if law.get('条文编号') == law_id:
                law.update(new_content)
                break
        
        os.makedirs('data', exist_ok=True)
        with open('data/laws.json', 'w', encoding='utf-8') as f:
            json.dump(self.laws_data, f, ensure_ascii=False, indent=2)
        
        self.rebuild_index()
    
    def _dict_to_text(self, data_dict):
        text_parts = []
        
        def extract_text(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_text(value, f"{prefix}{key}: ")
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, prefix)
            else:
                text_parts.append(f"{prefix}{str(obj)}")
        
        extract_text(data_dict)
        return " ".join(text_parts) 