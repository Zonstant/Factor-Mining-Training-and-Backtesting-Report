from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ──────────────────────────────────────────────────────────────────────────────
# Training Arguments
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingArguments:
    """
    训练超参与行为控制。

    Args:
        output_dir             : 模型 / checkpoint 保存根目录
        num_epochs             : 训练总轮数
        per_device_batch_size  : 每个 batch 的样本数
        learning_rate          : AdamW 初始学习率
        weight_decay           : L2 正则系数（bias 和 norm 参数不受影响）
        max_grad_norm          : 梯度裁剪上界，None = 不裁剪
        warmup_ratio           : 线性 warmup 占总 step 的比例（0 = 不 warmup）
        lr_scheduler_type      : "cosine" | "linear" | "constant"
        eval_strategy          : "epoch" | "steps" | "no"
        eval_steps             : eval_strategy="steps" 时的评估间隔
        save_strategy          : "epoch" | "steps" | "no"
        save_steps             : save_strategy="steps" 时的保存间隔
        logging_steps          : 打印训练 loss 的 step 间隔
        device                 : "cpu" | "cuda" | "cuda:N" | "mps"，
                                 None 则自动选择（优先 cuda → mps → cpu）
        dataloader_num_workers : DataLoader worker 进程数
        seed                   : 随机种子
    """
    output_dir: str = "output"
    num_epochs: int = 10
    per_device_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"   # "cosine" | "linear" | "constant"
    eval_strategy: str = "epoch"        # "epoch" | "steps" | "no"
    eval_steps: int = 100
    save_strategy: str = "epoch"        # "epoch" | "steps" | "no"
    save_steps: int = 100
    logging_steps: int = 20
    device: Optional[str] = None
    dataloader_num_workers: int = 0
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Trainer State
# ──────────────────────────────────────────────────────────────────────────────

class TrainerState:
    """训练过程中的只读状态与 metric 历史记录。"""

    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.total_steps: int = 0
        self.history: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        self.history.append({"step": self.global_step, "epoch": self.epoch, **metrics})


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    轻量 Trainer，核心流程参考 transformers.Trainer，去除分布式 / FP16 等复杂功能。

    数据约定
    --------
    Dataset.__getitem__ 返回一个 dict，键名与 model.forward 的参数名对应。
    Trainer 会将 batch dict 以 **kwargs 方式传入 model.forward。

    示例：
        batch = {"x": tensor, "labels": tensor, "key_padding_mask": tensor}
        →  model(x=tensor, labels=tensor, key_padding_mask=tensor)

    model.forward 约定：
    - 有 labels 时返回 (loss: scalar Tensor, logits: Tensor)
    - 无 labels 时返回 logits: Tensor（用于 predict）

    Args:
        model           : nn.Module
        args            : TrainingArguments
        train_dataset   : 训练集 Dataset
        eval_dataset    : 评估集 Dataset，可为 None
        data_collator   : 将样本列表合并为 batch dict 的函数；
                          None 则使用 PyTorch 默认 default_collate
        compute_metrics : (logits: Tensor, labels: Tensor) -> Dict[str, float]
                          自定义评估指标回调，None 则仅报告 eval_loss
        optimizers      : (optimizer, scheduler) 外部注入，
                          设为 (None, None) 则 Trainer 自动构建
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: tuple = (None, None),
    ):
        self.model           = model
        self.args            = args
        self.train_dataset   = train_dataset
        self.eval_dataset    = eval_dataset
        self.data_collator   = data_collator or _default_collate
        self.compute_metrics = compute_metrics

        self.device = _resolve_device(args.device)
        self.model.to(self.device)

        self._ext_optimizer, self._ext_scheduler = optimizers
        self.state = TrainerState()

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def train(self) -> TrainerState:
        """
        启动完整训练流程，返回最终 TrainerState。
        TrainerState.history 记录了每个 logging step 及 eval 的全部 metrics。
        """
        args = self.args
        torch.manual_seed(args.seed)
        os.makedirs(args.output_dir, exist_ok=True)

        train_loader = self._build_dataloader(self.train_dataset, shuffle=True)
        total_steps  = len(train_loader) * args.num_epochs
        self.state.total_steps = total_steps

        optimizer = self._ext_optimizer or self._build_optimizer()
        scheduler = self._ext_scheduler or self._build_scheduler(optimizer, total_steps)

        logger.info(
            f"开始训练 | epochs={args.num_epochs}  "
            f"steps/epoch={len(train_loader)}  "
            f"total_steps={total_steps}  device={self.device}"
        )

        self.model.train()

        for epoch in range(1, args.num_epochs + 1):
            self.state.epoch = epoch
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader, 1):
                self.state.global_step += 1
                batch = _to_device(batch, self.device)

                optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()

                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()

                loss_val    = loss.item()
                epoch_loss += loss_val
                lr_now      = scheduler.get_last_lr()[0]

                if self.state.global_step % args.logging_steps == 0:
                    avg = epoch_loss / step
                    logger.info(
                        f"epoch {epoch}/{args.num_epochs}  "
                        f"step {self.state.global_step}/{total_steps}  "
                        f"loss={loss_val:.4f}  avg={avg:.4f}  lr={lr_now:.2e}"
                    )
                    self.state.log({"train_loss": loss_val, "lr": lr_now})

                # step 级评估
                if (
                    args.eval_strategy == "steps"
                    and self.state.global_step % args.eval_steps == 0
                    and self.eval_dataset is not None
                ):
                    self._run_eval()
                    self.model.train()

                # step 级保存
                if (
                    args.save_strategy == "steps"
                    and self.state.global_step % args.save_steps == 0
                ):
                    self.save_checkpoint(tag=f"step-{self.state.global_step}")

            # ── epoch 结束 ──────────────────────────────────────────────────
            avg_epoch = epoch_loss / len(train_loader)
            elapsed   = time.time() - t0
            logger.info(
                f"── epoch {epoch} 完成  avg_loss={avg_epoch:.4f}  "
                f"耗时={elapsed:.1f}s"
            )

            if args.eval_strategy == "epoch" and self.eval_dataset is not None:
                self._run_eval()
                self.model.train()

            if args.save_strategy == "epoch":
                self.save_checkpoint(tag=f"epoch-{epoch}")

        logger.info("训练完成。")
        return self.state

    # ── 评估 ─────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        对指定数据集评估并返回 metrics dict。
        可在训练循环外直接调用。
        """
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("没有可用的 eval_dataset，请传入 eval_dataset 参数。")
        return self._run_eval(dataset=dataset, prefix=prefix)

    # ── 推理 ─────────────────────────────────────────────────────────────────

    def predict(self, test_dataset: Dataset) -> torch.Tensor:
        """
        对 test_dataset 纯推理，返回拼接后的 logits [N_total, ...]。
        test_dataset 的 batch dict 中不应含 "labels" 键。
        """
        loader     = self._build_dataloader(test_dataset, shuffle=False)
        all_logits: List[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch  = _to_device(batch, self.device)
                output = self.model(**batch)
                logits = output[1] if isinstance(output, (tuple, list)) else output
                all_logits.append(logits.cpu())

        return torch.cat(all_logits, dim=0)

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = "checkpoint") -> str:
        """
        保存模型权重到 {output_dir}/{tag}/model.pt。
        返回保存路径。
        """
        save_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "model.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"checkpoint 已保存 → {path}")
        return path

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        从 {checkpoint_dir}/model.pt 恢复模型权重。
        """
        path  = os.path.join(checkpoint_dir, "model.pt")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        logger.info(f"checkpoint 已加载 ← {path}")

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """forward 并提取标量 loss。"""
        output = self.model(**batch)
        loss   = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError(
                "model.forward 应返回标量 loss 或 (loss, logits)，"
                f"实际得到 {type(output)}"
            )
        return loss

    def _run_eval(
        self,
        dataset: Optional[Dataset] = None,
        prefix: str = "eval",
    ) -> Dict[str, float]:
        dataset = dataset or self.eval_dataset
        loader  = self._build_dataloader(dataset, shuffle=False)

        total_loss  = 0.0
        all_logits:  List[torch.Tensor] = []
        all_labels:  List[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch  = _to_device(batch, self.device)
                output = self.model(**batch)

                if isinstance(output, (tuple, list)):
                    loss, logits = output[0], output[1]
                    total_loss  += loss.item()
                    all_logits.append(logits.cpu())
                else:
                    total_loss += output.item()

                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())

        metrics: Dict[str, float] = {
            f"{prefix}_loss": total_loss / max(len(loader), 1)
        }

        if self.compute_metrics is not None and all_logits and all_labels:
            extra = self.compute_metrics(
                torch.cat(all_logits, dim=0),
                torch.cat(all_labels, dim=0),
            )
            metrics.update({f"{prefix}_{k}": v for k, v in extra.items()})

        logger.info("  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        self.state.log(metrics)
        return metrics

    def _build_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def _build_optimizer(self) -> AdamW:
        """AdamW，对 bias / LayerNorm weight 不施加 weight_decay。"""
        no_wd_keywords = {"bias", "norm"}
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_wd_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return AdamW(
            [
                {"params": decay_params,    "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
        )

    def _build_scheduler(self, optimizer: AdamW, total_steps: int):
        args         = self.args
        warmup_steps = max(0, int(total_steps * args.warmup_ratio))

        def _main_scheduler(optim, n_steps: int):
            if args.lr_scheduler_type == "cosine":
                return CosineAnnealingLR(
                    optim,
                    T_max=max(n_steps, 1),
                    eta_min=args.learning_rate * 0.01,
                )
            elif args.lr_scheduler_type == "linear":
                return LinearLR(
                    optim, start_factor=1.0, end_factor=0.0,
                    total_iters=max(n_steps, 1),
                )
            else:  # "constant"
                return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda _: 1.0)

        if warmup_steps == 0:
            return _main_scheduler(optimizer, total_steps)

        warmup = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0,
            total_iters=warmup_steps,
        )
        main = _main_scheduler(optimizer, total_steps - warmup_steps)
        return SequentialLR(optimizer, schedulers=[warmup, main],
                            milestones=[warmup_steps])


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _default_collate(samples: List[dict]) -> dict:
    """将 list[dict] 合并为 dict[str, Tensor]（使用 PyTorch 内置 collate）。"""
    from torch.utils.data import default_collate as _dc
    return _dc(samples)
