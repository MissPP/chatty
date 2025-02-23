from flask import Flask, request, jsonify
from models.local_llm import LanguageModel 
import torch

app = Flask(__name__)

# model_path = "/ds/ds7b"  # 预训练模型路径
model_path = "path1"
sft_model_path = "path2"  # 微调模型路径

lm = LanguageModel(model_path=model_path, sft_model_path=sft_model_path)

# 默认的文本生成配置
config = {
    "max_length": 50,
    "num_return_sequences": 1
}

@app.route('/generate', methods=['POST'])
def generate():
    """
    接收用户请求的文本并生成响应
    """
    data = request.get_json()  # 获取请求体中的 JSON 数据
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    response = lm.generate_response(input_text, **config)
    
    return jsonify({"response": response})


@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    """
    接收一批文本并返回批量生成的结果
    """
    data = request.get_json()
    texts = data.get('texts', [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid input. Provide a list of texts."}), 400

    responses = [lm.generate_response(text, **config) for text in texts]
    
    return jsonify({"responses": responses})


@app.route('/status', methods=['GET'])
def status():
    """
    返回模型的当前状态
    """
    device = lm.device
    cuda_available = torch.cuda.is_available()
    
    return jsonify({
        "status": "Model loaded",
        "device": device,
        "cuda_available": cuda_available
    })


@app.route('/config', methods=['GET'])
def get_config():
    """
    获取当前的生成参数配置
    """
    return jsonify(config)


@app.route('/set_config', methods=['POST'])
def set_config():
    """
    允许用户动态调整生成参数，如 max_length
    """
    data = request.get_json()
    max_length = data.get('max_length')
    num_return_sequences = data.get('num_return_sequences')

    if max_length:
        config["max_length"] = max_length
    if num_return_sequences:
        config["num_return_sequences"] = num_return_sequences

    return jsonify({"message": "Configuration updated", "config": config})


@app.route('/health', methods=['GET'])
def health():
    """
    健康检查 API，返回模型是否正常运行
    """
    try:
        test_text = "This is a test."
        _ = lm.generate_response(test_text, max_length=10)
        return jsonify({"status": "healthy"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


