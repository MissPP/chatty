from transformers import AutoTokenizer
import os
import re

class DataPreprocessor:
    def __init__(self, model_name="gpt2"):
        # 加载预训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # 去掉多余的空格
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # 去掉非 ASCII 字符
        return text.strip()

    def tokenize_text(self, text):
        # 使用分词器将文本转换为模型的输入格式
        cleaned_text = self.clean_text(text)
        return self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    def preprocess_documents(self, document_folder):
        # 遍历文档文件夹，处理每个文档
        processed_data = []
        for filename in os.listdir(document_folder):
            file_path = os.path.join(document_folder, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    processed_data.append(self.tokenize_text(text))
        return processed_data
