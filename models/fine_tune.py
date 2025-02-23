import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator

class FineTuning:
    def __init__(self, model_name, dataset_path, output_dir="../data/fine-tuned-model"):
        """
        初始化 FineTuning 类，设置模型路径、数据集路径和输出目录。
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator()
        # self.model = accelerator.prepare(model)  # 自动处理设备选择
        # self.model.gradient_checkpointing_enable()  # 启用梯度检查点
        # self.model = dispatch_model(model)  # 让 `accelerate` 自动管理 CPU/GPU 迁移
        # torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        # 清空GPU内存
        # torch.cuda.empty_cache()
        # gc.collect()

        # 设置显存配置
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        # print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])

        # 加载模型和分词器
        self._load_model_and_tokenizer()

        # 加载数据集
        self._load_dataset()

        # 准备模型进行微调
        self._prepare_model_for_finetuning()

        # 设置训练参数
        self._setup_training_args()

    def _load_model_and_tokenizer(self):
        """
        加载模型和分词器
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # 自动分配设备（CPU/GPU）
        )
        self.model.to(self.device)

    def _load_dataset(self):
        """
        加载数据集并进行预处理（tokenization）
        """
        self.dataset = load_dataset("json", data_files=self.dataset_path)

        # 数据集预处理：将输入和标签都进行tokenization
        def preprocess_function(examples):
            inputs = self.tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(examples["output"], padding="max_length", truncation=True, max_length=128)
            inputs["labels"] = labels["input_ids"]
            return inputs

        self.tokenized_datasets = self.dataset.map(preprocess_function, batched=True, remove_columns=["input_ids", "labels"])

    def _prepare_model_for_finetuning(self):
        """
        配置LoRA（低秩适配器）以减少显存占用
        """
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

    def _setup_training_args(self):
        """
        设置训练参数
        """
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            learning_rate=1e-6,  # 设置较低的学习率
            num_train_epochs=3,
            logging_dir="./logs",  # 日志目录
            logging_steps=10,  # 每10步记录一次日志
            save_steps=500,  # 每500步保存一次模型
            evaluation_strategy="no",  # 不进行评估
            save_total_limit=2,  # 保留最新的2个检查点
            # fp16=True,  # 混合精度训练（适用于GPU）
            # remove_unused_columns=False,  # 保留未使用的列
            # no_cuda=False,  # 允许使用GPU
            max_grad_norm=1.0,  # 梯度裁剪
            gradient_accumulation_steps=1  # 梯度累积步数
        )

    def _prepare_trainer(self):
        """
        准备训练器
        """
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],# !!!
            tokenizer=self.tokenizer
        )

    def train(self):
        """
        开始训练过程
        """
        self._prepare_trainer()
        self.trainer.train()

    def save_model(self):
        """
        保存微调后的模型和分词器
        """
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

def test(model_name, dataset_path, output_dir="./fine-tuned-model"):
    """
    测试函数，调用 FineTuning 类进行训练和模型保存。
    """
    fine_tuning = FineTuning(model_name=model_name, dataset_path=dataset_path, output_dir=output_dir)
    fine_tuning.train()
    fine_tuning.save_model()
