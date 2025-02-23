from utils.document_loader import load_documents

def test_document_loader():
    documents = load_documents("data/documents/")
    assert len(documents) > 0, "Should load at least one document"
