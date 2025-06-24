# 🎭 中华古诗词对联生成器

基于Qwen大语言模型微调的中华古诗词对联生成系统，使用LoRA技术实现高效微调。

## ✨ 特性

- 🤖 基于Qwen-0.6B模型微调
- 🔧 使用LoRA技术，参数高效微调
- 💾 支持4-bit量化，降低显存需求
- 🎨 智能对联生成，保持平仄韵律
- 📚 大规模中华古诗词语料训练
- 🚀 快速推理，实时生成

## 🛠️ 环境要求

### 硬件要求
- GPU: 建议4GB+显存（支持CUDA）
- CPU: 多核处理器
- 内存: 8GB+

### 软件依赖
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU版本)

## 📦 安装

1. 克隆项目
```bash
git clone https://github.com/your-username/chinese-poem-generator.git
cd chinese-poem-generator
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
将您的诗词语料文件放置为 `chinese_poems.txt`

## 🚀 快速开始

### 训练模型
```bash
python train.py
```

### 生成对联
```python
from poem_generator import PoemGenerator

generator = PoemGenerator("./qwen-poem-final")
result = generator.generate_couplet("春风又绿江南岸")
print(f"下联: {result}")
```

### 交互式生成
```bash
python interactive.py
```

## 📊 模型架构

- **基础模型**: Qwen/Qwen3-0.6B
- **微调方法**: LoRA (Low-Rank Adaptation)
- **量化**: 4-bit NF4量化
- **目标模块**: q_proj, k_proj, v_proj, o_proj
- **LoRA秩**: 8
- **LoRA Alpha**: 16

## 📈 训练配置

```python
training_config = {
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "epochs": 1,
    "learning_rate": 2e-4,
    "scheduler": "cosine",
    "warmup_ratio": 0.1
}
```

## 🎨 使用示例

```python
# 示例对联生成
test_cases = [
    "徐行踏断流水声",
    "春风又绿江南岸", 
    "无边落木萧萧下",
    "大漠孤烟直",
    "十年生死两茫茫"
]

for line in test_cases:
    result = generator.generate_couplet(line)
    print(f"上联: {line}")
    print(f"下联: {result}")
    print("-" * 30)
```

## 📁 项目结构

```
chinese-poem-generator/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── train.py              # 主训练脚本
├── poem_generator.py     # 生成器类
├── interactive.py        # 交互式界面
├── dataset.py           # 数据集处理
├── config.py            # 配置文件
├── utils/
│   ├── __init__.py
│   ├── data_utils.py    # 数据处理工具
│   └── model_utils.py   # 模型工具
├── examples/
│   ├── basic_usage.py
│   └── batch_generate.py
├── docs/
│   ├── training_guide.md
│   └── api_reference.md
└── tests/
    ├── test_dataset.py
    ├── test_generator.py
    └── test_utils.py
```

## 📖 数据格式

诗词数据文件格式：
```
上联1,下联1.上联2,下联2
春风又绿江南岸,明月何时照我还
无边落木萧萧下,不尽长江滚滚来
```

## 🔧 配置说明

在 `config.py` 中可以调整：
- 模型参数
- 训练超参数  
- 数据处理配置
- 生成参数

## 📚 API文档

### PoemGenerator 类

```python
class PoemGenerator:
    def __init__(self, model_path: str)
    def generate_couplet(self, first_line: str, **kwargs) -> str
    def batch_generate(self, lines: List[str]) -> List[str]
```

详细API文档请参考 [docs/api_reference.md](docs/api_reference.md)

## 🧪 测试

运行测试套件：
```bash
python -m pytest tests/
```

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🤝 贡献

欢迎贡献代码！请查看贡献指南：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📞 支持

如有问题或建议，请：
- 📧 发邮件至: your-email@example.com
- 🐛 提交Issue: [GitHub Issues](https://github.com/your-username/chinese-poem-generator/issues)
- 💬 参与讨论: [GitHub Discussions](https://github.com/your-username/chinese-poem-generator/discussions)

## 🙏 致谢

- [Qwen模型](https://github.com/QwenLM/Qwen) - 提供强大的基础模型
- [Hugging Face](https://huggingface.co/) - 模型和工具支持
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调框架

## 📈 路线图

- [ ] 支持更多诗词格式（七言、五言等）
- [ ] 添加韵律检测和评分
- [ ] Web界面开发
- [ ] API服务部署
- [ ] 移动端应用
- [ ] 多语言支持

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！
