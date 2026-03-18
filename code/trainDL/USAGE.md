# quant_code 快速上手

## 项目结构

```
quant_code/
├── model/
│   └── model.py        # Transformer 主体 + TradeModel
└── trainer/
    ├── __init__.py
    └── trainer.py      # Trainer + TrainingArguments
```

---

## 一、模型：TradeModel

### 两种输入模式

| 模式 | 参数 | 输入 shape | Token 语义 |
|---|---|---|---|
| 股票模式 | `factor_size=F` | `[B, N_stocks, F]` | 每个 token = 一只股票的全部因子 |
| 因子模式 | `portfolio_size=N` | `[B, N_factors, N]` | 每个 token = 一个因子在全部股票上的值 |

两个参数**必须且只能指定一个**。

### 输出规范化（output_mode）

| 值 | 效果 | 适用场景 |
|---|---|---|
| `"raw"` | 原始实数，无约束 | 灵活，由 loss 自行处理 |
| `"long_only"` | softmax，权重 ≥ 0 且和为 1 | 纯多头策略 |
| `"long_short"` | 绝对值归一化，保留正负号 | 多空对冲策略 |

### 四种 Loss

| loss_type | 说明 | 适用场景 |
|---|---|---|
| `"direct"` | Huber loss，直接回归 label 权重 | 高买低卖，精确权重 |
| `"ranking"` | ListNet loss，全局排名对齐 | 做多做空，只关心相对顺序 |
| `"topk"` | 仅对极端 top-K / bottom-K 计算 ListNet | 只关心最强 / 最弱信号 |
| `"combined"` | `α·direct + β·ranking + γ·topk` | 综合多种目标 |

### 完整初始化示例

```python
from model.model import TradeModel

model = TradeModel(
    d_model=128,          # Transformer 隐层维度
    n_heads=4,            # 注意力头数（必须整除 d_model）
    n_layers=4,           # Transformer 层数
    factor_size=10,       # 每只股票有 10 个因子 → 股票模式
    # portfolio_size=20,  # 改用因子模式时替换上行
    max_seq_len=512,      # 最大序列长度（≥ N_stocks 或 N_factors）
    intermediate_size=None,  # FFN 中间层，None = 自动推断 (8/3 * d_model)
    dropout=0.1,
    output_mode="long_only",         # 纯多头
    loss_type="combined",            # 联合 loss
    k=0.1,                           # topk：取前/后 10%
    k_mode="ratio",                  # "ratio" | "count"
    tau=1.0,                         # ListNet 温度系数
    loss_weights=(1.0, 1.0, 1.0),   # (α, β, γ) → (direct, ranking, topk)
)
```

### Forward

```python
# 训练：传入 labels，返回 (loss, logits)
loss, logits = model(x, labels=labels)

# 推理：不传 labels，返回 logits
logits = model(x)
# logits shape: [B, N_stocks]
```

---

## 二、数据集

Dataset 的 `__getitem__` 返回一个 **dict**，键名与 `model.forward` 参数名对应：

```python
from torch.utils.data import Dataset
import torch

class TradeDataset(Dataset):
    """
    最简示例：随机生成因子数据和 label 权重。
    实际使用时替换为真实数据加载逻辑。
    """
    def __init__(self, n_samples=1000, n_stocks=50, factor_size=10):
        # 因子矩阵：[n_samples, n_stocks, factor_size]
        self.x = torch.randn(n_samples, n_stocks, factor_size)
        # label 权重：[n_samples, n_stocks]，softmax 归一化
        self.labels = torch.softmax(torch.randn(n_samples, n_stocks), dim=-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> dict:
        return {
            "x": self.x[idx],           # [n_stocks, factor_size]
            "labels": self.labels[idx], # [n_stocks]
            # 可选：padding mask
            # "key_padding_mask": self.mask[idx],  # [n_stocks] bool
        }
```

**数据键名规则：**

| 键名 | 对应 model.forward 参数 | 说明 |
|---|---|---|
| `x` | `x` | 输入因子矩阵（必须） |
| `labels` | `labels` | 真实权重（训练时必须，推理时省略） |
| `key_padding_mask` | `key_padding_mask` | padding 位置标记，`True`=忽略（可选） |

---

## 三、训练

### TrainingArguments

```python
from trainer import TrainingArguments

args = TrainingArguments(
    output_dir="output",           # checkpoint 保存目录
    num_epochs=30,
    per_device_batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,             # 梯度裁剪，None = 不裁剪
    warmup_ratio=0.05,             # 前 5% step 做线性 warmup
    lr_scheduler_type="cosine",    # "cosine" | "linear" | "constant"
    eval_strategy="epoch",         # "epoch" | "steps" | "no"
    eval_steps=100,                # eval_strategy="steps" 时生效
    save_strategy="epoch",         # "epoch" | "steps" | "no"
    save_steps=100,
    logging_steps=20,              # 每 20 step 打印一次 loss
    device=None,                   # None = 自动选择（cuda > mps > cpu）
    dataloader_num_workers=0,
    seed=42,
)
```

### 启动训练

```python
from trainer import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TradeDataset(split="train"),
    eval_dataset=TradeDataset(split="val"),
)

state = trainer.train()
# state.global_step  → 总训练步数
# state.history      → 每次 logging 的 metrics 列表
```

### 自定义评估指标（可选）

```python
import torch

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    logits: [N, n_stocks]  预测权重
    labels: [N, n_stocks]  真实权重
    返回 dict[str, float]，key 会自动加 "eval_" 前缀
    """
    # 示例：用 Spearman 秩相关衡量排名质量
    from scipy.stats import spearmanr
    import numpy as np
    corrs = []
    for p, l in zip(logits.numpy(), labels.numpy()):
        corr, _ = spearmanr(p, l)
        corrs.append(corr)
    return {"spearman": float(np.mean(corrs))}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
```

### 自定义 data_collator（可选）

```python
import torch

def my_collator(samples: list[dict]) -> dict:
    """
    当样本长度不等（需要 padding）时，使用自定义 collator。
    """
    max_len = max(s["x"].shape[0] for s in samples)
    x_list, label_list, mask_list = [], [], []
    for s in samples:
        T = s["x"].shape[0]
        pad = max_len - T
        x_list.append(torch.cat([s["x"], torch.zeros(pad, s["x"].shape[1])]))
        label_list.append(torch.cat([s["labels"], torch.zeros(pad)]))
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[T:] = True
        mask_list.append(mask)
    return {
        "x": torch.stack(x_list),
        "labels": torch.stack(label_list),
        "key_padding_mask": torch.stack(mask_list),
    }

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds,
                  data_collator=my_collator)
```

### 外部注入 optimizer / scheduler（可选）

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                       steps_per_epoch=100, epochs=30)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds,
                  optimizers=(optimizer, scheduler))
```

---

## 四、推理与 Checkpoint

```python
# 推理（不传 labels）
logits = trainer.predict(test_dataset)
# logits shape: [N_total, n_stocks]

# 手动保存
path = trainer.save_checkpoint(tag="best")

# 加载权重
trainer.load_checkpoint("output/epoch-10")
# 或直接加载到模型
import torch
model.load_state_dict(torch.load("output/epoch-10/model.pt"))
```

---

## 五、最简完整示例

```python
import torch
from torch.utils.data import Dataset
from model.model import TradeModel
from trainer import Trainer, TrainingArguments


class TradeDataset(Dataset):
    def __init__(self, n=500):
        self.x      = torch.randn(n, 30, 10)   # 30只股票，10个因子
        self.labels = torch.softmax(torch.randn(n, 30), dim=-1)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return {"x": self.x[i], "labels": self.labels[i]}


model = TradeModel(
    d_model=64, n_heads=4, n_layers=3,
    factor_size=10,
    output_mode="long_only",
    loss_type="combined",
)

args = TrainingArguments(
    output_dir="output",
    num_epochs=20,
    per_device_batch_size=32,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TradeDataset(400),
    eval_dataset=TradeDataset(100),
)

state = trainer.train()
print(f"训练完成，共 {state.global_step} 步")

# 推理
logits = trainer.predict(TradeDataset(10))
print(f"推理输出 shape: {logits.shape}")  # [10, 30]
```
