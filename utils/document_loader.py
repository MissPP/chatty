import os
from PyPDF2 import PdfReader

def load_documents(path: str):
    documents = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        
        if file_name.endswith(".pdf"):
            documents.append(load_pdf(file_path))
        elif file_name.endswith(".txt"):
            documents.append(load_txt(file_path))

    return documents

def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_txt(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
