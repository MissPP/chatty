import os
import torch
from datetime import datetime
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LanguageModel:
    def __init__(self, model_path, sft_model_path, device=None):
        """
        初始化语言模型类，加载预训练的模型和微调权重。
        :param model_path: 预训练模型路径
        :param sft_model_path: 微调模型权重路径
        :param device: 设备配置，默认为None
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 加载分词器和预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"  # Automatically place model on available devices
            
            # load_in_8bit=True,  # Enable 4-bit quantization
            # bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
            # bnb_4bit_quant_type="nf4",  # Use 'nf4' quantization (you can also try other types)
            # bnb_4bit_use_double_quant=True, 
            # load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            # llm_int8_enable_fp32_cpu_offload=True
        )
        # 加载微调模型
        self.model = PeftModel.from_pretrained(self.model, sft_model_path, device_map="auto")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        # self.model.to("cpu")
        # self.model.to("cuda")
        self.model.eval()

    def generate_response(self, text, max_length=50, num_return_sequences=1):
        """
        生成模型的响应
        :param text: 输入文本
        :param max_length: 生成响应的最大长度
        :param num_return_sequences: 返回的响应数量
        :return: 生成的响应文本
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                attention_mask=inputs['attention_mask']
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def get_current_time(self):
        return datetime.now()

    def print_response_time(self, text):
        """
        打印生成响应的时间以及响应长度
        :param text: 输入文本
        """
        current_time = self.get_current_time()
        print(current_time)

        response = self.generate_response(text)
        print(response)

        current_time = self.get_current_time()
        print(len(response))
        print(current_time)


def test():
    model_path = "path1"  # 预训练模型路径
    sft_model_path = "path2"  # 微调模型路径

    lm = LanguageModel(model_path=model_path, sft_model_path=sft_model_path)

    input_text = "who are you"
    lm.print_response_time(input_text)
