from models.local_llm import LocalLLM

def test_local_llm():
    llm = LocalLLM()
    response = llm.generate("What is AI?")
    assert len(response) > 0, "Response should not be empty"
