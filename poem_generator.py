"""
诗词对联生成器 - 中华古诗词对联生成器
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Any, Optional
import logging
from config import GenerationConfig

logger = logging.getLogger(__name__)


class PoemGenerator:
    """诗词对联生成器类"""
    
    def __init__(self, 
                 model_path: str,
                 base_model_name: str = "Qwen/Qwen3-0.6B",
                 device: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            model_path: 微调模型路径
            base_model_name: 基础模型名称
            device: 设备类型
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
        
    def _load_model(self):
        """加载模型和分词器"""
        logger.info("🤖 正在加载模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        try:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            logger.info("✅ LoRA适配器加载成功")
        except Exception as e:
            logger.warning(f"⚠️ LoRA适配器加载失败，使用基础模型: {e}")
        
        self.model.eval()
        logger.info(f"✅ 模型加载完成，设备: {self.device}")
    
    def generate_couplet(self, 
                        first_line: str,
                        config: Optional[GenerationConfig] = None) -> str:
        """
        生成对联下联
        
        Args:
            first_line: 上联
            config: 生成配置
            
        Returns:
            下联
        """
        if config is None:
            config = GenerationConfig()
            
        # 构建prompt
        messages = [
            {"role": "system", "content": "你是一位精通对对联的古代诗人。"},
            {"role": "user", "content": f"上联：{first_line}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 分词
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            )
        
        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 提取下联
        if "下联：" in response:
            return response.split("下联：")[1].strip()
        return response.strip()
    
    def batch_generate(self, 
                      first_lines: List[str],
                      config: Optional[GenerationConfig] = None) -> List[str]:
        """
        批量生成对联
        
        Args:
            first_lines: 上联列表
            config: 生成配置
            
        Returns:
            下联列表
        """
        results = []
        for line in first_lines:
            result = self.generate_couplet(line, config)
            results.append(result)
        return results
    
    def interactive_mode(self):
        """交互式生成模式"""
        print("🎭 中华古诗词对联生成器 - 交互模式")
        print("输入上联，AI将为您生成下联")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n请输入上联: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    print("⚠️ 请输入有效的上联")
                    continue
                
                print("🤔 AI正在思考...")
                result = self.generate_couplet(user_input)
                
                print(f"✨ 上联: {user_input}")
                print(f"🎯 下联: {result}")
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\n👋 程序已退出")
                break
            except Exception as e:
                print(f"❌ 生成失败: {e}")
    
    def evaluate_samples(self, test_samples: List[str]) -> Dict[str, Any]:
        """
        评估样本生成质量
        
        Args:
            test_samples: 测试样本列表
            
        Returns:
            评估结果
        """
        results = []
        total_time = 0
        
        for sample in test_samples:
            import time
            start_time = time.time()
            
            generated = self.generate_couplet(sample)
            
            gen_time = time.time() - start_time
            total_time += gen_time
            
            results.append({
                "input": sample,
                "output": generated,
                "time": gen_time
            })
        
        return {
            "results": results,
            "total_time": total_time,
            "avg_time": total_time / len(test_samples),
            "samples_count": len(test_samples)
        }


def load_generator(model_path: str, base_model_name: str = "Qwen/Qwen3-0.6B") -> PoemGenerator:
    """
    快速加载生成器
    
    Args:
        model_path: 模型路径
        base_model_name: 基础模型名称
        
    Returns:
        PoemGenerator实例
    """
    return PoemGenerator(model_path, base_model_name)