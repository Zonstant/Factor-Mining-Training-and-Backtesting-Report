import torch
from torch import nn
from torch.nn import functional as F
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = nn.Linear(vocab_size, d_model, bias=False)  # 简单线性映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)  # [B, T, d_model]

class LearnablePositionEmbedding(nn.Module):
    """
    可学习的绝对位置编码。
    输入: x, shape [B, T, d_model]
    输出: x + pos_emb, shape [B, T, d_model]
    """
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)   # [1, T]
        x = x + self.pos_emb(positions)                             # [B, T, d_model]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力（双向，无因果掩码）。
    支持 padding mask：mask 形状为 [B, T]，True 表示该位置为 padding，需要被忽略。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                        # [B, T, d_model]
        key_padding_mask: torch.Tensor = None,  # [B, T] bool, True=padding
    ) -> torch.Tensor:
        B, T, d_model = x.shape
        H, d_h = self.n_heads, self.d_head

        # 计算 Q, K, V 并拆分多头
        qkv = self.qkv_proj(x)                  # [B, T, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)          # 各 [B, T, d_model]
        # reshape -> [B, H, T, d_h]
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # 应用 padding mask
        if key_padding_mask is not None:
            # [B, 1, 1, T] -> 广播到 [B, H, T, T]
            mask = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, v)             # [B, H, T, d_h]
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        return self.out_proj(out)               # [B, T, d_model]

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络:
        FFN(x) = (SiLU(W1·x) ⊙ W2·x) · W3
    intermediate_size 通常取 d_model * 8/3（再对齐到 64 的倍数）。
    """
    def __init__(self, d_model: int, intermediate_size: int = None, dropout: float = 0.0):
        super().__init__()
        if intermediate_size is None:
            # 按 LLaMA 惯例: 8/3 * d_model，对齐到 64
            raw = int(d_model * 8 / 3)
            intermediate_size = ((raw + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)  # gate (SiLU)
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False)  # value
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))    # SiLU 激活
        val  = self.w2(x)            # 线性门控
        x = gate * val               # 逐元素相乘
        x = self.dropout(x)
        return self.w3(x)

class TransformerBlock(nn.Module):
    """单个 Transformer 层（Pre-LayerNorm）。"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        intermediate_size: int = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn  = SwiGLUFFN(d_model, intermediate_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 残差 + Pre-LN 自注意力
        x = x + self.attn(self.ln1(x), key_padding_mask)
        # 残差 + Pre-LN FFN
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    """
    完整 Transformer 编码器。

    Args:
        vocab_size     : 词表大小（TokenEmbedding 实现后传入）
        d_model        : 隐层维度
        n_heads        : 注意力头数
        n_layers       : Transformer 层数
        max_seq_len    : 最大序列长度（用于 position embedding）
        intermediate_size : FFN 中间层维度，None 则自动推断
        dropout        : dropout 比例
        norm_eps       : LayerNorm epsilon
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        portfolio_size: int = None,
        factor_size: int = None,
        max_seq_len: int = 512,
        intermediate_size: int = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert (portfolio_size is None and factor_size is not None) \
            or (portfolio_size is not None and factor_size is None), "只能指定 portfolio_size 或 factor_size 之一"
        if portfolio_size is not None:
            vocab_size = portfolio_size
        else:
            vocab_size = factor_size
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb   = LearnablePositionEmbedding(max_seq_len, d_model, dropout)
        self.layers    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, intermediate_size, dropout, norm_eps)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model, eps=norm_eps)          # 最终归一化

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, input_dim]
        key_padding_mask: torch.Tensor = None, # [B, T] bool，True=padding
    ) -> torch.Tensor:
        """返回编码后的隐层表示 [B, T, d_model]，不做任何输出投影。"""
        x = self.token_emb(x)                  # [B, T, d_model]
        x = self.pos_emb(x)                    # [B, T, d_model]
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.ln_out(x)                  # [B, T, d_model]

class TradeModel(nn.Module):
    """
    交易专用 Transformer 模型，兼容 HuggingFace Trainer。

    输入模式（二选一）：
      - factor_size  模式：x shape = [B, N_stocks, factor_size]，
                           每个 token 代表一只股票的全部因子。
      - portfolio_size 模式：x shape = [B, N_factors, portfolio_size]，
                           每个 token 代表一个因子在全部股票上的值。

    输出：[B, N_stocks] 的权重向量（经 output_mode 规范化后）。

    Args:
        d_model          : 隐层维度
        n_heads          : 注意力头数
        n_layers         : Transformer 层数
        portfolio_size   : 投资组合股票数（与 factor_size 互斥）
        factor_size      : 因子数量（与 portfolio_size 互斥）
        max_seq_len      : 最大序列长度
        intermediate_size: FFN 中间层维度，None 自动推断
        dropout          : dropout 比例
        norm_eps         : LayerNorm epsilon
        output_mode      : 权重规范化方式
                           "raw"         - 原始 logits，无约束
                           "long_only"   - softmax，权重 ≥ 0 且和为 1（纯多头）
                           "long_short"  - 归一化为权重绝对值之和为 1（多空对冲）
        loss_type        : 损失函数类型
                           "direct"   - Huber loss，直接回归权重（高买低卖）
                           "ranking"  - ListNet loss，全局排名（做多做空）
                           "topk"     - 仅对极端排名（top-K / bottom-K）计算 ListNet
                           "combined" - direct + ranking + topk 加权求和
        k                : topk/combined 中极端比例 (ratio 模式) 或数量 (count 模式)
        k_mode           : "ratio" | "count"
        tau              : ListNet 温度系数（越大越平滑）
        loss_weights     : combined 模式下各项权重 (alpha, beta, gamma)
                           对应 (direct, ranking, topk)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        portfolio_size: int = None,
        factor_size: int = None,
        max_seq_len: int = 512,
        intermediate_size: int = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        output_mode: str = "raw",
        loss_type: str = "direct",
        k: float = 0.1,
        k_mode: str = "ratio",
        tau: float = 1.0,
        loss_weights: tuple = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        assert (portfolio_size is None) != (factor_size is None), \
            "portfolio_size 和 factor_size 必须且只能指定一个"
        assert output_mode in ("raw", "long_only", "long_short"), \
            f"output_mode 必须为 'raw' / 'long_only' / 'long_short'，得到 '{output_mode}'"
        assert loss_type in ("direct", "ranking", "topk", "combined"), \
            f"loss_type 必须为 'direct' / 'ranking' / 'topk' / 'combined'，得到 '{loss_type}'"
        assert k_mode in ("ratio", "count"), \
            f"k_mode 必须为 'ratio' / 'count'，得到 '{k_mode}'"

        self._factor_mode = factor_size is not None
        self.n_stocks = factor_size if self._factor_mode else portfolio_size
        self.output_mode = output_mode
        self.loss_type   = loss_type
        self.k           = k
        self.k_mode      = k_mode
        self.tau         = tau
        self.loss_weights = loss_weights

        self.transformer = Transformer(
            d_model, n_heads, n_layers, portfolio_size, factor_size,
            max_seq_len, intermediate_size, dropout, norm_eps
        )

        # Weight head: 将 [B, T, d_model] → [B, N_stocks]
        if self._factor_mode:
            # T = N_stocks，每个 token 出一个标量权重
            self.weight_head = nn.Linear(d_model, 1, bias=False)
        else:
            # T = N_factors，先 mean-pool 后投影到 N_stocks
            self.weight_head = nn.Linear(d_model, self.n_stocks, bias=False)

    # ── 输出规范化 ────────────────────────────────────────────────────────────
    def _apply_output_mode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [B, N_stocks] → 规范化后的权重 [B, N_stocks]"""
        if self.output_mode == "long_only":
            return F.softmax(logits, dim=-1)
        elif self.output_mode == "long_short":
            # 权重绝对值之和归一化为 1，保留正负号（多空对冲）
            return logits / (logits.abs().sum(-1, keepdim=True) + 1e-8)
        else:  # "raw"
            return logits

    # ── Loss 1: 直接回归权重（Huber / Smooth-L1） ─────────────────────────────
    def _direct_loss(
        self,
        pred: torch.Tensor,   # [B, N_stocks]
        label: torch.Tensor,  # [B, N_stocks]
    ) -> torch.Tensor:
        """高买低卖：预测权重与 label 权重的 Huber loss。"""
        return F.huber_loss(pred, label, reduction="mean")

    # ── ListNet 核心（可微分排序 loss） ───────────────────────────────────────
    def _listnet_loss(
        self,
        pred: torch.Tensor,   # [B, N] 或 [B, 2K]（任意最后一维）
        label: torch.Tensor,  # 同 pred
    ) -> torch.Tensor:
        """
        ListNet loss:
            L = -sum_i  P_label(i) * log P_pred(i)
        其中 P(i) = softmax(x / tau)_i。
        """
        p_label   = F.softmax(label / self.tau, dim=-1)
        log_p_pred = F.log_softmax(pred  / self.tau, dim=-1)
        return -(p_label * log_p_pred).sum(dim=-1).mean()

    # ── Loss 2: 全局排名 ──────────────────────────────────────────────────────
    def _ranking_loss(
        self,
        pred: torch.Tensor,   # [B, N_stocks]
        label: torch.Tensor,  # [B, N_stocks]
    ) -> torch.Tensor:
        """做多做空：只关心相对排名，将 label 的排名分布作为软目标。"""
        return self._listnet_loss(pred, label)

    # ── Loss 3: 极端排名 ──────────────────────────────────────────────────────
    def _topk_loss(
        self,
        pred: torch.Tensor,   # [B, N_stocks]
        label: torch.Tensor,  # [B, N_stocks]
    ) -> torch.Tensor:
        """
        TopK loss：仅对 label 中排名最高和最低的 K 只股票计算 ListNet。
        中间排名的股票被忽略，降低对平庸信号的过拟合。
        """
        N = label.size(-1)
        if self.k_mode == "ratio":
            k = max(1, int(N * self.k))
        else:
            k = max(1, int(self.k))
        k = min(k, N // 2)  # 确保 top+bottom 不超过全集

        top_idx = label.topk(k, dim=-1).indices               # [B, k]
        bot_idx = label.topk(k, dim=-1, largest=False).indices # [B, k]
        idx = torch.cat([top_idx, bot_idx], dim=-1)            # [B, 2k]

        pred_sub  = pred.gather(-1, idx)    # [B, 2k]
        label_sub = label.gather(-1, idx)   # [B, 2k]
        return self._listnet_loss(pred_sub, label_sub)

    # ── Loss 4: 联合 loss ─────────────────────────────────────────────────────
    def _combined_loss(
        self,
        pred: torch.Tensor,   # [B, N_stocks]
        label: torch.Tensor,  # [B, N_stocks]
    ) -> torch.Tensor:
        """
        Combined loss:
            L = alpha * L_direct + beta * L_ranking + gamma * L_topk
        """
        alpha, beta, gamma = self.loss_weights
        loss = pred.new_zeros(1).squeeze()
        if alpha != 0.0:
            loss = loss + alpha * self._direct_loss(pred, label)
        if beta != 0.0:
            loss = loss + beta  * self._ranking_loss(pred, label)
        if gamma != 0.0:
            loss = loss + gamma * self._topk_loss(pred, label)
        return loss

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                        # [B, T, input_dim]
        labels: torch.Tensor = None,            # [B, N_stocks]，有时返回 (loss, logits)
        key_padding_mask: torch.Tensor = None,  # [B, T] bool，True=padding
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x                : 输入特征
                               factor_size  模式 → [B, N_stocks, factor_size]
                               portfolio_size 模式 → [B, N_factors, portfolio_size]
            labels           : 真实权重 [B, N_stocks]。
                               提供时返回 (loss, logits)，
                               不提供时仅返回 logits。
            key_padding_mask : [B, T] bool，padding 位置为 True。

        Returns:
            有 labels → (loss: scalar, logits: [B, N_stocks])
            无 labels → logits: [B, N_stocks]
        """
        hidden = self.transformer(x, key_padding_mask)  # [B, T, d_model]

        # Weight head
        if self._factor_mode:
            # 每个 token（= 每只股票）出一个权重
            logits = self.weight_head(hidden).squeeze(-1)  # [B, N_stocks]
        else:
            # 先对因子维度 mean-pool，再投影到 N_stocks
            pooled = hidden.mean(dim=1)                    # [B, d_model]
            logits = self.weight_head(pooled)              # [B, N_stocks]

        # 输出规范化
        logits = self._apply_output_mode(logits)           # [B, N_stocks]

        if labels is None:
            return logits

        # 计算 loss
        if self.loss_type == "direct":
            loss = self._direct_loss(logits, labels)
        elif self.loss_type == "ranking":
            loss = self._ranking_loss(logits, labels)
        elif self.loss_type == "topk":
            loss = self._topk_loss(logits, labels)
        else:  # "combined"
            loss = self._combined_loss(logits, labels)

        return loss, logits