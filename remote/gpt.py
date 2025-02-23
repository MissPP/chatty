import openai
from typing import Optional, List, Dict, Union
import os
import time
import json
import logging
from dataclasses import dataclass
import random
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChatConfig:
    max_tokens: int = 100
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay_base: float = 1.0
    timeout: float = 30.0

class ChatGPT:
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "gpt-3.5-turbo",
                 config: ChatConfig = None):
        """
        初始化 ChatGPT 客户端
        
        Args:
            api_key (str): OpenAI API 密钥
            model (str): 默认模型
            config (ChatConfig): 配置对象
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key 未提供！请检查配置或环境变量 OPENAI_API_KEY")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.config = config or ChatConfig()
        self.message_history: List[Dict[str, str]] = []
        self.rate_limit_count = 0
        logger.info(f"ChatGPT 客户端初始化完成，使用模型: {self.model}")

    def _handle_rate_limit(self, attempt: int) -> None:
        """处理速率限制"""
        delay = self.config.retry_delay_base * (2 ** attempt) + (random.random() * 0.1)
        logger.warning(f"触发速率限制，第 {attempt + 1} 次重试，等待 {delay:.2f} 秒")
        time.sleep(delay)

    def generate_text(self, 
                     prompt: str, 
                     role: str = "user",
                     max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     return_format: str = "text") -> Optional[Union[str, Dict]]:
        """
        生成文本，支持多种格式返回
        
        Args:
            prompt (str): 输入提示
            role (str): 消息角色 (system/user/assistant)
            max_tokens (int): 最大 tokens，可覆盖默认值
            temperature (float): 温度参数，可覆盖默认值
            return_format (str): 返回格式 (text/json)
        
        Returns:
            Union[str, Dict]: 根据 return_format 返回文本或 JSON
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # 添加到消息历史
        self.message_history.append({"role": role, "content": prompt})
        
        try:
            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.message_history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=self.config.timeout,
                        n=1,
                        stop=None
                    )
                    
                    result = response.choices[0].message.content.strip()
                    self.message_history.append({"role": "assistant", "content": result})
                    
                    if return_format == "json":
                        return {
                            "content": result,
                            "model": self.model,
                            "timestamp": time.time(),
                            "usage": dict(response.usage)
                        }
                    return result
                
                except openai.RateLimitError:
                    self.rate_limit_count += 1
                    if attempt < self.config.max_retries - 1:
                        self._handle_rate_limit(attempt)
                        continue
                    raise
                
                except openai.APIError as e:
                    logger.error(f"API 错误: {str(e)}")
                    return None
            
            logger.error("达到最大重试次数")
            return None
            
        except Exception as e:
            logger.error(f"意外错误: {str(e)}")
            return None

    def set_model(self, model: str) -> None:
        """更改模型"""
        logger.info(f"切换模型: {self.model} -> {model}")
        self.model = model

    def clear_history(self) -> None:
        """清除消息历史"""
        self.message_history.clear()
        logger.info("消息历史已清除")

    def save_session(self, filepath: str) -> bool:
        """保存会话到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": self.model,
                    "history": self.message_history,
                    "rate_limit_count": self.rate_limit_count
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"会话保存至: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存会话失败: {str(e)}")
            return False

def test():
    api_key = "key"
    
    config = ChatConfig(
        max_tokens=150,
        temperature=0.8,
        max_retries=5,
        retry_delay_base=2.0
    )
    
    gpt_client = ChatGPT(api_key=api_key, config=config)
    
    # 测试复杂交互
    prompt = "写一首关于春天的四行诗，要求押韵"
    result = gpt_client.generate_text(
        prompt=prompt,
        role="user",
        return_format="json"
    )
    
    if result:
        print("生成结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("生成失败")
    
    # 保存会话
    gpt_client.save_session("chat_session.json")
