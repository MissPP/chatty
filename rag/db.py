import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentSearch:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """
        初始化 DocumentSearch 对象，加载 Sentence-BERT 模型和 FAISS 索引。
        :param model_name: 预训练 Sentence-BERT 模型的名称。
        """
        # 加载预训练的 Sentence-BERT 模型
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_vectors = None
    
    def add_documents(self, documents):
        """
        将文档添加到搜索索引中。
        :param documents: 一个字符串列表，表示要添加的文档。
        """
        self.documents = documents
        # 将文档转换为向量（嵌入）
        self.document_vectors = self.model.encode(documents)
        # 创建 FAISS 索引
        d = self.document_vectors.shape[1]  # 向量的维度
        self.index = faiss.IndexFlatL2(d)  # 基于 L2 距离的索引
        self.index.add(np.array(self.document_vectors).astype('float32'))  # 将文档向量添加到索引中
    
    def search(self, query, k=2):
        """
        对查询进行相似度搜索。
        :param query: 一个字符串，表示查询内容。
        :param k: 要检索的最相似文档数量（默认为 2）。
        :return: 一个元组列表，每个元组包含（文档，距离），表示最相似的文档及其距离。
        """
        if self.index is None:
            raise ValueError("文档索引尚未初始化。请先添加文档。")
        
        # 将查询转换为向量
        query_vector = self.model.encode([query])
        # 搜索最相似的文档
        D, I = self.index.search(np.array(query_vector).astype('float32'), k)  # D: 距离，I: 索引
        
        results = []
        for i in range(k):
            results.append((self.documents[I[0][i]], D[0][i]))  # 返回文档及其距离
        
        return results

# 示例用法：
def test():
    doc_search = DocumentSearch()
    
    documents = [
        "Python 是一种强大的编程语言。",
        "FAISS 是一个高效的相似度搜索库。",
        "机器学习是人工智能的一个子集。",
        "自然语言处理是人工智能的一个领域。",
        "深度学习是机器学习算法的一类。"
    ]
    doc_search.add_documents(documents)
    
    query = "告诉我关于机器学习的内容。"
    results = doc_search.search(query, k=2)
    
    print("查询:", query)
    print("最相似的文档:")
    for doc, dist in results:
        print(f"文档: {doc}, 距离: {dist}")
