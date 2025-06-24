"""
数据集处理模块 - 中华古诗词对联生成器
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PoemDataset(Dataset):
    """诗词对联数据集类"""
    
    def __init__(self, corpus: List[str], tokenizer, max_len: int = 64):
        """
        初始化数据集
        
        Args:
            corpus: 诗词语料列表
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.couplets = []
        
        self._process_corpus(corpus)
        
    def _process_corpus(self, corpus: List[str]) -> None:
        """处理语料，提取对联"""
        logger.info("🔄 正在处理对联数据...")
        processed_count = 0
        
        for poem in tqdm(corpus, desc="解析对联", unit="句", colour="blue"):
            parts = poem.split('.')
            for part in parts:
                halves = part.split(',')
                if len(halves) == 2 and halves[0].strip() and halves[1].strip():
                    self.couplets.append({
                        "source": halves[0].strip(),
                        "target": halves[1].strip()
                    })
                    processed_count += 1
        
        logger.info(f"✅ 处理完成，得到 {len(self.couplets):,} 个有效对联")
        
        if not self.couplets:
            raise ValueError("没有找到有效的对联数据")
    
    def __len__(self) -> int:
        return len(self.couplets)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个训练样本"""
        item = self.couplets[idx]
        source_text = item["source"]
        target_text = item["target"]
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": "你是一位精通对对联的古代诗人。"},
            {"role": "user", "content": f"上联：{source_text}"},
            {"role": "assistant", "content": f"下联：{target_text}"}
        ]
        
        # 生成完整prompt
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 生成用于计算prompt长度的版本
        messages_for_prompt_len = messages[:-1]
        prompt_only = self.tokenizer.apply_chat_template(
            messages_for_prompt_len,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 分词
        full_tokenized = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None
        )
        
        # 计算prompt长度并设置labels
        prompt_len = len(self.tokenizer.encode(prompt_only))
        labels = list(full_tokenized['input_ids'])
        labels[:prompt_len] = [-100] * prompt_len
        
        full_tokenized["labels"] = labels
        return full_tokenized
    
    def get_sample_couplets(self, n: int = 5) -> List[Dict[str, str]]:
        """获取样本对联用于展示"""
        return self.couplets[:min(n, len(self.couplets))]


def load_corpus(file_path: str) -> List[str]:
    """
    加载诗词语料
    
    Args:
        file_path: 语料文件路径
        
    Returns:
        语料列表
    """
    logger.info(f"📖 从文件加载数据: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info("🔄 正在处理数据...")
        corpus = []
        for line in tqdm(lines, desc="处理诗句", unit="行", colour="green"):
            cleaned_line = line.strip()
            if cleaned_line:
                corpus.append(cleaned_line)
        
        logger.info(f"✅ 成功加载 {len(corpus):,} 行诗词数据")
        
        if not corpus:
            raise ValueError("语料库文件内容为空")
            
        return corpus
        
    except FileNotFoundError:
        logger.error(f"❌ 文件未找到: '{file_path}'")
        raise
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {str(e)}")
        raise


def create_data_splits(corpus: List[str], 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1) -> tuple:
    """
    划分训练、验证、测试数据集
    
    Args:
        corpus: 原始语料
        train_ratio: 训练集比例
        val_ratio: 验证集比例  
        test_ratio: 测试集比例
        
    Returns:
        (train_corpus, val_corpus, test_corpus)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    total_size = len(corpus)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size:train_size + val_size]
    test_corpus = corpus[train_size + val_size:]
    
    logger.info(f"数据集划分完成:")
    logger.info(f"  训练集: {len(train_corpus):,} ({len(train_corpus)/total_size:.1%})")
    logger.info(f"  验证集: {len(val_corpus):,} ({len(val_corpus)/total_size:.1%})")
    logger.info(f"  测试集: {len(test_corpus):,} ({len(test_corpus)/total_size:.1%})")
    
    return train_corpus, val_corpus, test_corpus