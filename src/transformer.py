"""
Transformer From Scratch — "Attention Is All You Need" 직접 구현
================================================================
Vaswani et al. (2017) 의 Transformer 구조를 PyTorch 로 처음부터 만들고,
시퀀스 역순 변환 작업으로 학습/시각화까지 검증.

핵심 부품 (논문 Figure 1 / 2):
  1. PositionalEncoding             — sinusoidal 위치 인코딩
  2. scaled_dot_product_attention   — Q · K · V 기반 가중합
  3. MultiHeadAttention             — h 개 헤드 병렬 attention
  4. PositionwiseFeedForward        — 두 개 FC + ReLU
  5. EncoderLayer / DecoderLayer    — 잔차 연결 + LayerNorm
  6. Transformer                    — 전체 합본 (Encoder + Decoder)

토이 작업: 8자리 숫자 시퀀스 → 역순 출력
  입력:  [BOS, 3, 7, 1, 4, 9, 2, 8, 5, EOS]
  출력:  [BOS, 5, 8, 2, 9, 4, 1, 7, 3, EOS]

실행 모드:
  --mode train      : 학습 (CPU 약 1분, GPU 수십 초)
  --mode visualize  : 시각화 9장 생성 (캐시된 모델이 없으면 자동 학습 후 생성)
  --mode all        : 학습 + 모든 시각화

원본 논문:
  Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_CACHE = RESULTS_DIR / "transformer_state.pt"
HISTORY_CACHE = RESULTS_DIR / "training_history.json"

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ─── 토큰 / 하이퍼파라미터 ───────────────────────────────────────────────
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
DIGIT_OFFSET = 3        # 디지트 d → 토큰 (d + 3)
VOCAB_SIZE = 13          # PAD, BOS, EOS, 0~9
SEQ_LEN = 8              # 디지트 개수
MAX_LEN = SEQ_LEN + 2    # BOS / EOS 포함

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1


# ════════════════════════════════════════════════════════════════════════
# 1. 위치 인코딩 — sinusoidal
# ════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """논문 식 (PE_(pos, 2i) = sin, PE_(pos, 2i+1) = cos) 그대로 구현."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ════════════════════════════════════════════════════════════════════════
# 2. Scaled Dot-Product Attention — 함수형
# ════════════════════════════════════════════════════════════════════════
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


# ════════════════════════════════════════════════════════════════════════
# 3. Multi-Head Attention
# ════════════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None  # 시각화용 캐시 (B, h, Lq, Lk)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        # (B, L, D) → (B, h, L, d_k)
        Q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # 헤드 차원 broadcast
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        self.last_attn = attn.detach()

        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        out = self.W_o(out)
        return self.dropout(out)


# ════════════════════════════════════════════════════════════════════════
# 4. Position-wise Feed-Forward
# ════════════════════════════════════════════════════════════════════════
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ════════════════════════════════════════════════════════════════════════
# 5. Encoder / Decoder 레이어
# ════════════════════════════════════════════════════════════════════════
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, src_mask))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x


# ════════════════════════════════════════════════════════════════════════
# 6. 전체 Transformer
# ════════════════════════════════════════════════════════════════════════
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        max_len: int = MAX_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encode(src, src_mask)
        dec = self.decode(tgt, enc, src_mask, tgt_mask)
        return self.fc_out(dec)


def make_pad_mask(seq: torch.Tensor) -> torch.Tensor:
    return (seq != PAD_IDX).unsqueeze(1)


def make_causal_mask(L: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, device=device, dtype=torch.bool)).unsqueeze(0)


# ════════════════════════════════════════════════════════════════════════
# 7. 토이 데이터 — 8자리 시퀀스 역순
# ════════════════════════════════════════════════════════════════════════
def generate_reverse_data(n_samples: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    digits = rng.integers(0, 10, size=(n_samples, SEQ_LEN)) + DIGIT_OFFSET

    src = np.concatenate([
        np.full((n_samples, 1), BOS_IDX),
        digits,
        np.full((n_samples, 1), EOS_IDX),
    ], axis=1)

    rev = digits[:, ::-1]
    tgt = np.concatenate([
        np.full((n_samples, 1), BOS_IDX),
        rev,
        np.full((n_samples, 1), EOS_IDX),
    ], axis=1)

    return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def token_to_label(t: int) -> str:
    if t == PAD_IDX: return "<PAD>"
    if t == BOS_IDX: return "<BOS>"
    if t == EOS_IDX: return "<EOS>"
    return str(t - DIGIT_OFFSET)


# ════════════════════════════════════════════════════════════════════════
# 8. 학습 / 캐시 로드
# ════════════════════════════════════════════════════════════════════════
def train_model(epochs: int = 25, batch_size: int = 64, n_samples: int = 4000,
                lr: float = 3e-4, seed: int = 42) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ train ] device = {device}")

    src_train, tgt_train = generate_reverse_data(n_samples, seed=seed)
    src_val, tgt_val = generate_reverse_data(500, seed=seed + 1)
    train_loader = DataLoader(
        TensorDataset(src_train, tgt_train),
        batch_size=batch_size, shuffle=True,
    )

    model = Transformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {"loss": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_pad_mask(src)
            tgt_pad = make_pad_mask(tgt_in)
            causal = make_causal_mask(tgt_in.size(1), device)
            tgt_mask = tgt_pad & causal

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            src, tgt = src_val.to(device), tgt_val.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_pad_mask(src)
            tgt_pad = make_pad_mask(tgt_in)
            causal = make_causal_mask(tgt_in.size(1), device)
            tgt_mask = tgt_pad & causal
            logits = model(src, tgt_in, src_mask, tgt_mask)
            preds = logits.argmax(dim=-1)
            valid = tgt_out != PAD_IDX
            acc = (preds[valid] == tgt_out[valid]).float().mean().item()

        history["loss"].append(ep_loss)
        history["val_acc"].append(acc)
        print(f"  epoch {ep+1:2d}/{epochs}  loss={ep_loss:.4f}  val_acc={acc*100:.1f}%")

    torch.save(model.state_dict(), MODEL_CACHE)
    HISTORY_CACHE.write_text(json.dumps(history), encoding="utf-8")
    print(f"[ saved ] {MODEL_CACHE.relative_to(ROOT_DIR)}")
    return model.cpu(), history


def get_or_train_model(force: bool = False) -> tuple[nn.Module, dict]:
    if MODEL_CACHE.exists() and HISTORY_CACHE.exists() and not force:
        print(f"[ load ] {MODEL_CACHE.relative_to(ROOT_DIR)}")
        model = Transformer()
        model.load_state_dict(torch.load(MODEL_CACHE, map_location="cpu"))
        history = json.loads(HISTORY_CACHE.read_text(encoding="utf-8"))
        return model, history
    return train_model()


def run_inference(model: nn.Module, src: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """그리디 디코딩 + attention 캐시 회수."""
    model.eval()
    device = next(model.parameters()).device
    src = src.to(device)
    src_mask = make_pad_mask(src)
    with torch.no_grad():
        enc_out = model.encode(src, src_mask)

        # 디코더는 BOS 부터 시작해 한 토큰씩 생성
        ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        for _ in range(MAX_LEN - 1):
            tgt_pad = make_pad_mask(ys)
            causal = make_causal_mask(ys.size(1), device)
            tgt_mask = tgt_pad & causal
            dec_out = model.decode(ys, enc_out, src_mask, tgt_mask)
            logits = model.fc_out(dec_out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == EOS_IDX:
                break

    # attention 캐시 정리
    cache: dict[str, list[torch.Tensor]] = {
        "encoder_self": [l.self_attn.last_attn.cpu() for l in model.encoder_layers],
        "decoder_self": [l.self_attn.last_attn.cpu() for l in model.decoder_layers],
        "decoder_cross": [l.cross_attn.last_attn.cpu() for l in model.decoder_layers],
    }
    return ys.cpu(), cache


# ════════════════════════════════════════════════════════════════════════
# 9. 시각화
# ════════════════════════════════════════════════════════════════════════
def visualize_architecture() -> Path:
    """논문 Figure 1 — Encoder/Decoder 구조도."""
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(8, 11.5, "Transformer 구조 — Encoder + Decoder",
            ha="center", fontsize=24, fontweight="bold")
    ax.text(8, 11.05,
            "RNN 없이 Self-Attention 만으로 시퀀스 변환 (Vaswani et al., NeurIPS 2017)",
            ha="center", fontsize=16, color="#37474F", fontweight="bold", style="italic")

    def block(x, y, w, h, text, fc, ec, fontsize=14):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                    edgecolor=ec, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#37474F", style="-|>", lw=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                     mutation_scale=18))

    # ── Encoder (left)
    enc_x = 1.0
    block(enc_x, 1.0, 4.5, 0.8, "Input Embedding + Positional Encoding",
          "#FFE0B2", "#E65100", 13)
    block(enc_x, 2.2, 4.5, 1.2, "Multi-Head Self-Attention",
          "#C5CAE9", "#283593", 16)
    block(enc_x, 3.6, 4.5, 0.6, "Add & LayerNorm", "#E8EAF6", "#283593", 14)
    block(enc_x, 4.4, 4.5, 1.2, "Feed-Forward (d_ff=128)",
          "#C8E6C9", "#1B5E20", 16)
    block(enc_x, 5.8, 4.5, 0.6, "Add & LayerNorm", "#E8F5E9", "#1B5E20", 14)
    ax.add_patch(plt.Rectangle((enc_x - 0.2, 2.0), 4.9, 4.6,
                                facecolor="none", edgecolor="#37474F",
                                linewidth=1.4, linestyle="--"))
    ax.text(enc_x + 4.95, 4.3, "× N\n(=2)",
            fontsize=16, color="#37474F", va="center", fontweight="bold")
    ax.text(enc_x + 2.25, 7.0, "ENCODER",
            ha="center", fontsize=20, fontweight="bold", color="#283593")

    # ── Decoder (right)
    dec_x = 9.5
    block(dec_x, 1.0, 4.5, 0.8, "Output Embedding + Positional Encoding",
          "#FFE0B2", "#E65100", 13)
    block(dec_x, 2.2, 4.5, 1.0, "Masked Multi-Head Self-Attention",
          "#FFCCBC", "#BF360C", 15)
    block(dec_x, 3.4, 4.5, 0.5, "Add & LayerNorm", "#FBE9E7", "#BF360C", 13)
    block(dec_x, 4.1, 4.5, 1.0, "Cross-Attention\n(Q=Decoder, K=V=Encoder)",
          "#F8BBD0", "#AD1457", 14)
    block(dec_x, 5.3, 4.5, 0.5, "Add & LayerNorm", "#FCE4EC", "#AD1457", 13)
    block(dec_x, 6.0, 4.5, 1.0, "Feed-Forward (d_ff=128)",
          "#C8E6C9", "#1B5E20", 16)
    block(dec_x, 7.2, 4.5, 0.5, "Add & LayerNorm", "#E8F5E9", "#1B5E20", 13)
    block(dec_x, 8.0, 4.5, 0.6, "Linear", "#E1BEE7", "#6A1B9A", 16)
    block(dec_x, 8.8, 4.5, 0.6, "Softmax → 다음 토큰 확률",
          "#D1C4E9", "#4527A0", 15)
    ax.add_patch(plt.Rectangle((dec_x - 0.2, 2.0), 4.9, 5.7,
                                facecolor="none", edgecolor="#37474F",
                                linewidth=1.4, linestyle="--"))
    ax.text(dec_x + 4.95, 4.85, "× N\n(=2)",
            fontsize=16, color="#37474F", va="center", fontweight="bold")
    ax.text(dec_x + 2.25, 9.9, "DECODER",
            ha="center", fontsize=20, fontweight="bold", color="#AD1457")

    ax.text(enc_x + 2.25, 0.4, "Source: [BOS, 3, 7, 1, 4, 9, 2, 8, 5, EOS]",
            ha="center", fontsize=15, fontweight="bold", color="#E65100")
    ax.text(dec_x + 2.25, 0.4, "Target (shifted): [BOS, 5, 8, 2, 9, 4, 1, 7, 3]",
            ha="center", fontsize=15, fontweight="bold", color="#E65100")

    arrow(enc_x + 2.25, 1.8, enc_x + 2.25, 2.2)
    arrow(enc_x + 2.25, 3.4, enc_x + 2.25, 3.6)
    arrow(enc_x + 2.25, 4.2, enc_x + 2.25, 4.4)
    arrow(enc_x + 2.25, 5.6, enc_x + 2.25, 5.8)

    arrow(dec_x + 2.25, 1.8, dec_x + 2.25, 2.2)
    arrow(dec_x + 2.25, 3.2, dec_x + 2.25, 3.4)
    arrow(dec_x + 2.25, 3.9, dec_x + 2.25, 4.1)
    arrow(dec_x + 2.25, 5.1, dec_x + 2.25, 5.3)
    arrow(dec_x + 2.25, 5.8, dec_x + 2.25, 6.0)
    arrow(dec_x + 2.25, 7.0, dec_x + 2.25, 7.2)
    arrow(dec_x + 2.25, 7.7, dec_x + 2.25, 8.0)
    arrow(dec_x + 2.25, 8.6, dec_x + 2.25, 8.8)

    ax.annotate("", xy=(dec_x, 4.6), xytext=(enc_x + 4.5, 6.4),
                arrowprops=dict(arrowstyle="-|>", color="#AD1457", lw=2.6,
                                 linestyle="--", mutation_scale=22))
    ax.text((enc_x + 4.5 + dec_x) / 2 + 0.3, 5.7,
            "Encoder 출력\n(K, V 로 사용)",
            ha="center", fontsize=15, color="#AD1457", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#AD1457",
                       boxstyle="round,pad=0.4", linewidth=1.5))

    legend = [
        mpatches.Patch(facecolor="#FFE0B2", edgecolor="#E65100", label="입력 임베딩 + PE"),
        mpatches.Patch(facecolor="#C5CAE9", edgecolor="#283593", label="Self-Attention"),
        mpatches.Patch(facecolor="#FFCCBC", edgecolor="#BF360C", label="Masked Self-Attention"),
        mpatches.Patch(facecolor="#F8BBD0", edgecolor="#AD1457", label="Cross-Attention"),
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="#1B5E20", label="Feed-Forward"),
        mpatches.Patch(facecolor="#D1C4E9", edgecolor="#4527A0", label="출력 Linear+Softmax"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=6, frameon=False,
              fontsize=14, bbox_to_anchor=(0.5, -0.02))

    out = RESULTS_DIR / "fig_01_transformer_architecture.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_qkv_search_engine() -> Path:
    """Q · K · V 의미 검색 엔진 — Query=찾는 것 / Key=특징 / Value=정보."""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(10, 11.4, "Q · K · V — 의미 검색 엔진",
            ha="center", fontsize=24, fontweight="bold")
    ax.text(10, 10.8,
            "같은 입력 벡터를 W_q · W_k · W_v 세 가중치로 곱해 세 가지 역할로 분화",
            ha="center", fontsize=16, color="#37474F", fontweight="bold", style="italic")

    # 입력 박스
    ax.add_patch(plt.Rectangle((0.6, 4.8), 2.6, 2.0,
                                facecolor="#FFE0B2", edgecolor="#E65100", linewidth=2.2))
    ax.text(1.9, 6.1, "입력 x",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(1.9, 5.4, "(B, L, 64)",
            ha="center", va="center", fontsize=14, color="#5D4037", fontweight="bold")

    # W_q, W_k, W_v
    weight_specs = [
        ("W_q", 8.4, "#3949AB"),
        ("W_k", 5.8, "#00897B"),
        ("W_v", 3.2, "#D32F2F"),
    ]
    for name, y, color in weight_specs:
        ax.add_patch(plt.Rectangle((4.6, y), 1.8, 1.4,
                                    facecolor=color, edgecolor="black", linewidth=1.8))
        ax.text(5.5, y + 0.7, name,
                ha="center", va="center", fontsize=20, fontweight="bold", color="white")
        # Arrow from input to W
        ax.annotate("", xy=(4.6, y + 0.7), xytext=(3.2, 5.8),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=18))

    # Q, K, V 박스 + 의미
    role_specs = [
        # name, y_top, color, role_label, metaphor, role
        ("Q  (Query)", 9.1, "#3949AB",
         "내가 무엇을 찾고 있는가?",
         "각 query 위치가 attend 할 대상을 탐색하는 역할",
         "→ 다른 토큰들에 던지는 \"질문\""),
        ("K  (Key)", 6.5, "#00897B",
         "나는 어떤 특징을 가졌는가?",
         "각 key 위치의 정체성·매칭 단서",
         "→ Q 와 매칭되는 \"색인\""),
        ("V  (Value)", 3.9, "#D32F2F",
         "전달할 핵심 정보",
         "매칭되었을 때 실제로 가져올 콘텐츠",
         "→ 가중 평균으로 합산되는 \"정보\""),
    ]
    for name, y, color, label, desc, tail in role_specs:
        # 큰 박스
        ax.add_patch(plt.Rectangle((8.0, y - 1.5), 11.4, 2.0,
                                    facecolor="white", edgecolor=color, linewidth=2.4))
        # 헤더 띠
        ax.add_patch(plt.Rectangle((8.0, y + 0.3), 11.4, 0.55,
                                    facecolor=color, edgecolor=color))
        ax.text(8.25, y + 0.575, name,
                ha="left", va="center", fontsize=18, fontweight="bold", color="white")
        ax.text(8.4, y + 0.0, f"“{label}”",
                ha="left", va="center", fontsize=16, fontweight="bold", color=color)
        ax.text(8.4, y - 0.6, desc,
                ha="left", va="center", fontsize=14, color="#37474F", fontweight="bold")
        ax.text(8.4, y - 1.15, tail,
                ha="left", va="center", fontsize=13, color="#424242", fontweight="bold", style="italic")
        # W → 박스 화살표
        ax.annotate("", xy=(8.0, y - 0.2), xytext=(6.4, y - 0.6 + 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=18))

    # 하단 식 박스
    ax.add_patch(plt.Rectangle((1.0, 0.5), 18.0, 1.6,
                                facecolor="#263238", edgecolor="#37474F", linewidth=1.2))
    ax.text(10, 1.55,
            r"Attention(Q, K, V) = softmax( Q · K$^T$ / √d_k ) · V",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#A5D6A7",
            family="monospace")
    ax.text(10, 0.85,
            "Q 가 모든 K 와 매칭(유사도) → 매칭 강한 V 가 가중합 → 출력",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#FFD54F")

    out = RESULTS_DIR / "fig_02_qkv_search_engine.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_scaled_dot_product_attention() -> Path:
    """수식과 작은 수치 예시로 Scaled Dot-Product Attention 의 5단계."""
    np.random.seed(7)
    L, d_k = 4, 4
    Q = np.random.randn(L, d_k)
    K = np.random.randn(L, d_k)
    V = np.random.randn(L, d_k)

    scores = Q @ K.T
    scaled = scores / math.sqrt(d_k)
    e = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    attn = e / e.sum(axis=-1, keepdims=True)
    out = attn @ V

    fig = plt.figure(figsize=(24, 11))
    fig.suptitle(
        "Scaled Dot-Product Attention — Q · K^T → ÷√d_k → Softmax → · V",
        fontsize=23, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.935,
             "각 단계의 행렬 형태와 실제 수치를 함께 표시 (L=4, d_k=4 예시)",
             ha="center", fontsize=16, color="#37474F", fontweight="bold", style="italic")

    def panel(idx, mat, title, cmap="RdBu_r", center=True):
        ax = fig.add_subplot(1, 5, idx)
        if center:
            vmax = max(abs(mat.min()), abs(mat.max()))
            im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=17, color="black", fontweight="bold")
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    panel(1, Q, "① Q (질의)\n(L, d_k) = (4, 4)")
    panel(2, scores, "② scores = Q · K^T\n(L, L)")
    panel(3, scaled, f"③ scores ÷ √d_k\n(÷ {math.sqrt(d_k):.2f})")
    panel(4, attn, "④ attn = softmax(③)\n(행 합 = 1)", cmap="YlOrRd", center=False)
    panel(5, out, "⑤ output = attn · V\n(L, d_k)")

    fig.text(0.5, 0.04,
             "softmax 행 합이 1 이 되도록 정규화 — 각 query 위치는 자기를 포함한 모든 key 에 가중치를 분배",
             ha="center", fontsize=15, color="#37474F", fontweight="bold")

    out_path = RESULTS_DIR / "fig_03_scaled_dot_product_attention.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_multihead_attention() -> Path:
    """h 개 헤드가 서로 다른 부분 공간을 학습하는 구조 도식 + 차원 표기."""
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.55, "Multi-Head Attention — 4 헤드 병렬 (d_model=64 → d_k=16 × 4)",
            ha="center", fontsize=23, fontweight="bold")
    ax.text(8, 8.05,
            "한 헤드당 더 작은 부분 공간(16 차원)에서 독립적으로 attention 을 계산하고, "
            "마지막에 concat 후 W^O 로 다시 64 차원으로 합침",
            ha="center", fontsize=15, color="#37474F", fontweight="bold", style="italic")

    ax.add_patch(plt.Rectangle((0.5, 3.5), 1.6, 2.0,
                                facecolor="#FFE0B2", edgecolor="#E65100", linewidth=2.0))
    ax.text(1.3, 4.5, "입력\n(B, L, 64)",
            ha="center", va="center", fontsize=16, fontweight="bold")

    head_colors = ["#C5CAE9", "#FFCCBC", "#C8E6C9", "#F8BBD0"]
    head_edges = ["#283593", "#BF360C", "#1B5E20", "#AD1457"]
    head_y = [6.5, 5.0, 3.5, 2.0]
    for i, (y, fc, ec) in enumerate(zip(head_y, head_colors, head_edges)):
        for j, name in enumerate(["W_q", "W_k", "W_v"]):
            ax.add_patch(plt.Rectangle((3.0 + j * 0.75, y), 0.65, 0.65,
                                        facecolor=fc, edgecolor=ec, linewidth=1.4))
            ax.text(3.0 + j * 0.75 + 0.325, y + 0.325, name,
                    ha="center", va="center", fontsize=13, fontweight="bold")
        ax.add_patch(plt.Rectangle((5.7, y - 0.1), 2.8, 0.85,
                                    facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(7.1, y + 0.32, f"Head {i+1}: Attention\n(B, L, 16)",
                ha="center", va="center", fontsize=14, fontweight="bold")

        ax.annotate("", xy=(3.0, y + 0.3), xytext=(2.1, 4.5),
                    arrowprops=dict(arrowstyle="-|>", color=ec,
                                     lw=1.6, mutation_scale=14))
        ax.annotate("", xy=(5.7, y + 0.3), xytext=(5.3, y + 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=ec,
                                     lw=1.6, mutation_scale=14))
        ax.annotate("", xy=(9.5, 4.5), xytext=(8.5, y + 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=ec,
                                     lw=1.4, mutation_scale=14, alpha=0.75))

    ax.add_patch(plt.Rectangle((9.5, 3.5), 2.0, 2.0,
                                facecolor="#FFF59D", edgecolor="#F9A825", linewidth=2.0))
    ax.text(10.5, 4.5, "Concat\n(B, L, 64)",
            ha="center", va="center", fontsize=16, fontweight="bold")

    ax.add_patch(plt.Rectangle((12.5, 3.5), 1.6, 2.0,
                                facecolor="#E1BEE7", edgecolor="#6A1B9A", linewidth=2.0))
    ax.text(13.3, 4.5, "W_O\nLinear",
            ha="center", va="center", fontsize=16, fontweight="bold")

    ax.add_patch(plt.Rectangle((14.5, 3.5), 1.3, 2.0,
                                facecolor="#FFCDD2", edgecolor="#C62828", linewidth=2.0))
    ax.text(15.15, 4.5, "출력\n(B, L, 64)",
            ha="center", va="center", fontsize=16, fontweight="bold")

    ax.annotate("", xy=(12.5, 4.5), xytext=(11.5, 4.5),
                arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=2.0, mutation_scale=18))
    ax.annotate("", xy=(14.5, 4.5), xytext=(14.1, 4.5),
                arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=2.0, mutation_scale=18))

    ax.text(8, 0.6,
            "Attention(Q, K, V) = Concat(head_1, …, head_4) · W^O,   "
            "head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)",
            ha="center", fontsize=15, family="monospace", fontweight="bold",
            bbox=dict(facecolor="#263238", edgecolor="#37474F",
                       boxstyle="round,pad=0.6", linewidth=1.2), color="#A5D6A7")

    out = RESULTS_DIR / "fig_04_multihead_attention.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_positional_encoding() -> Path:
    """sinusoidal 위치 인코딩 — 히트맵 + 4개 차원 라인 플롯."""
    pe_module = PositionalEncoding(D_MODEL, max_len=50)
    pe = pe_module.pe.squeeze(0).numpy()  # (50, 64)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(22, 10),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    fig.suptitle(
        "Positional Encoding — sinusoidal 패턴 (max_len=50, d_model=64)",
        fontsize=23, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.965,
        "PE(pos, 2i) = sin(pos / 10000^(2i/d)),   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
        ha="center", fontsize=16, color="#37474F", fontweight="bold",
        style="italic", family="monospace",
    )

    im = ax1.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xlabel("위치 (pos)", fontweight="bold", fontsize=16)
    ax1.set_ylabel("임베딩 차원 (i)", fontweight="bold", fontsize=16)
    ax1.set_title("위치 × 차원 히트맵 — 차원이 커질수록 파장이 길어짐",
                   fontsize=17, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=14)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=13)

    chosen_dims = [0, 4, 16, 32]
    colors = ["#1565C0", "#388E3C", "#E65100", "#6A1B9A"]
    for d, c in zip(chosen_dims, colors):
        ax2.plot(pe[:, d], label=f"dim {d}", color=c, linewidth=2.8)
    ax2.set_xlabel("위치 (pos)", fontweight="bold", fontsize=16)
    ax2.set_ylabel("PE 값", fontweight="bold", fontsize=16)
    ax2.set_title("선택 차원의 sinusoidal 곡선 — 차원별 다른 주기",
                   fontsize=17, fontweight="bold")
    ax2.axhline(0, color="#9E9E9E", linewidth=0.6)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=15)
    ax2.tick_params(axis="both", labelsize=14)

    fig.tight_layout()
    out = RESULTS_DIR / "fig_05_positional_encoding.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_masked_attention() -> Path:
    """디코더의 causal mask — 미래 토큰을 -∞로 가려 autoregressive 보장."""
    SEQ = 8
    tokens = ["<BOS>", "I", "love", "eating", "apples", ".", "<EOS>", "<PAD>"]

    # mask[i, j] = True (보임) if i >= j else False (가림)
    mask = np.tril(np.ones((SEQ, SEQ), dtype=bool))

    # 가상 점수 (학습 전이라 가정)
    rng = np.random.default_rng(13)
    scores = rng.normal(0, 1, (SEQ, SEQ))
    masked = np.where(mask, scores, -np.inf)
    # softmax (행 단위)
    e = np.exp(masked - np.nanmax(np.where(np.isfinite(masked), masked, 0),
                                     axis=1, keepdims=True))
    e = np.where(mask, e, 0.0)
    softmax_out = e / e.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle("Masked Attention — 디코더의 인과 관계 마스크 (M_ij = −∞ if i < j)",
                 fontsize=22, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945,
             "디코더는 학습 시 정답 시퀀스 전체를 받지만, 미래 토큰은 미리 보면 안 됨 — "
             "softmax 전에 −∞ 로 가려 0 으로 만들어 autoregressive 보장",
             ha="center", fontsize=15, color="#37474F", fontweight="bold", style="italic")

    # ── 왼쪽: 마스크 행렬 (1 vs -∞)
    ax = axes[0]
    display = mask.astype(float)
    ax.imshow(display, cmap="Blues", vmin=-0.3, vmax=1.2)
    for i in range(SEQ):
        for j in range(SEQ):
            if mask[i, j]:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=22, fontweight="bold", color="white")
            else:
                ax.text(j, i, "−∞", ha="center", va="center",
                        fontsize=18, fontweight="bold", color="#B71C1C")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor="#FFCDD2", edgecolor="#B71C1C",
                                            linewidth=0.8, alpha=0.5))
                # X 표시
                ax.plot([j - 0.4, j + 0.4], [i - 0.4, i + 0.4],
                        color="#B71C1C", linewidth=1.4, alpha=0.4)
                ax.plot([j - 0.4, j + 0.4], [i + 0.4, i - 0.4],
                        color="#B71C1C", linewidth=1.4, alpha=0.4)
    ax.set_xticks(range(SEQ))
    ax.set_yticks(range(SEQ))
    ax.set_xticklabels(tokens, fontsize=14, fontweight="bold", rotation=15)
    ax.set_yticklabels(tokens, fontsize=14, fontweight="bold")
    ax.set_xlabel("Key 위치 (j) — 참조 대상", fontsize=16, fontweight="bold")
    ax.set_ylabel("Query 위치 (i) — 현재 토큰", fontsize=16, fontweight="bold")
    ax.set_title("① 마스크 행렬\n파란 셀 = 보임 (i ≥ j)   빨간 셀 = 가림 (i < j)",
                  fontsize=17, fontweight="bold")

    # ── 오른쪽: softmax 후 weights
    ax = axes[1]
    im = ax.imshow(softmax_out, cmap="YlOrRd", vmin=0, vmax=softmax_out.max())
    for i in range(SEQ):
        for j in range(SEQ):
            v = softmax_out[i, j]
            if v < 1e-6:
                ax.text(j, i, "0.00", ha="center", va="center",
                        fontsize=13, color="#9E9E9E", fontweight="bold")
            else:
                color = "white" if v > 0.5 else "#37474F"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=13, color=color, fontweight="bold")
    ax.set_xticks(range(SEQ))
    ax.set_yticks(range(SEQ))
    ax.set_xticklabels(tokens, fontsize=14, fontweight="bold", rotation=15)
    ax.set_yticklabels(tokens, fontsize=14, fontweight="bold")
    ax.set_xlabel("Key 위치 (j)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Query 위치 (i)", fontsize=16, fontweight="bold")
    ax.set_title("② softmax 후 attention 가중치\n미래 위치 (i < j) 는 모두 0",
                  fontsize=17, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = RESULTS_DIR / "fig_06_masked_attention.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_training_curve(history: dict) -> Path:
    losses = history["loss"]
    accs = history["val_acc"]
    epochs = list(range(1, len(losses) + 1))

    fig, ax1 = plt.subplots(figsize=(16, 8.5))
    ax1.plot(epochs, losses, color="#1565C0", linewidth=3.0, marker="o",
             markersize=8, label="train loss")
    ax1.set_xlabel("Epoch", fontweight="bold", fontsize=16)
    ax1.set_ylabel("Cross-Entropy Loss", color="#1565C0", fontweight="bold", fontsize=16)
    ax1.tick_params(axis="y", labelcolor="#1565C0", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(epochs, [a * 100 for a in accs], color="#2E7D32", linewidth=3.0,
             marker="s", markersize=8, label="val accuracy (%)")
    ax2.set_ylabel("Validation Accuracy (%)", color="#2E7D32",
                    fontweight="bold", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="#2E7D32", labelsize=14)
    ax2.set_ylim(0, 105)

    final_acc = accs[-1] * 100
    ax2.axhline(final_acc, color="#2E7D32", linestyle="--", linewidth=1.4, alpha=0.6)
    ax2.text(epochs[-1], final_acc + 2.5, f"최종 {final_acc:.1f}%",
             ha="right", color="#2E7D32", fontsize=16, fontweight="bold")

    fig.suptitle(
        "학습 곡선 — 8자리 시퀀스 역순 변환 작업 (CrossEntropy + AdamW)",
        fontsize=20, fontweight="bold",
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=15)

    out = RESULTS_DIR / "fig_07_training_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_multihead_pattern_comparison(model: nn.Module) -> Path:
    """Layer × Head 별 cross-attention 패턴 비교 (역순 작업이라 anti-diagonal 이 정답)."""
    src_demo = torch.tensor([[BOS_IDX, 3+3, 7+3, 1+3, 4+3, 9+3, 2+3, 8+3, 5+3, EOS_IDX]],
                              dtype=torch.long)
    pred, cache = run_inference(model, src_demo)

    src_labels = [token_to_label(t.item()) for t in src_demo[0]]
    pred_labels = [token_to_label(t.item()) for t in pred[0]]

    cross = cache["decoder_cross"]
    n_layers = len(cross)
    n_heads = cross[0].shape[1]

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(5.8 * n_heads, 5.4 * n_layers),
    )
    fig.suptitle(
        "Layer × Head 별 Cross-Attention 패턴 — 헤드마다 다른 부분 공간을 학습",
        fontsize=22, fontweight="bold", y=0.995,
    )

    if n_layers == 1:
        axes = np.array([axes])

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li, hi]
            attn = cross[li][0, hi].numpy()
            ax.imshow(attn, cmap="YlOrRd", vmin=0, vmax=attn.max())
            ax.set_title(f"Layer {li+1} · Head {hi+1}",
                          fontsize=17, fontweight="bold")
            ax.set_xticks(range(len(src_labels)))
            ax.set_yticks(range(attn.shape[0]))
            ax.set_xticklabels(src_labels, fontsize=13, fontweight="bold")
            ax.set_yticklabels(pred_labels[: attn.shape[0]], fontsize=13, fontweight="bold")
            if li == n_layers - 1:
                ax.set_xlabel("Source", fontsize=15, fontweight="bold")
            if hi == 0:
                ax.set_ylabel("Target", fontsize=15, fontweight="bold")

    fig.tight_layout()
    out = RESULTS_DIR / "fig_08_multihead_pattern_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_attention_heatmap(model: nn.Module) -> Path:
    """마지막 레이어 head-averaged Encoder Self / Decoder Cross attention 히트맵."""
    src_demo = torch.tensor(
        [[BOS_IDX, 3+3, 7+3, 1+3, 4+3, 9+3, 2+3, 8+3, 5+3, EOS_IDX]],
        dtype=torch.long,
    )
    pred, cache = run_inference(model, src_demo)

    src_labels = [token_to_label(t.item()) for t in src_demo[0]]
    pred_labels = [token_to_label(t.item()) for t in pred[0]]

    # 마지막 레이어 head-average → (Ls, Ls), (Lt, Ls)
    enc_self = cache["encoder_self"][-1][0].mean(dim=0).numpy()
    dec_cross = cache["decoder_cross"][-1][0].mean(dim=0).numpy()
    Lt = dec_cross.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(26, 12))
    fig.suptitle(
        "Attention 가중치 히트맵 — 마지막 레이어 (4 Head 평균)",
        fontsize=24, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.965,
        "입력: [BOS, 3, 7, 1, 4, 9, 2, 8, 5, EOS]  →  역순 예측: " +
        " ".join(pred_labels),
        ha="center", fontsize=15, color="#37474F", fontweight="bold", style="italic",
    )

    def draw_heatmap(ax, mat, cmap, title, xlabels, ylabels, xlabel, ylabel):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(xlabels)))
        ax.set_yticks(range(len(ylabels)))
        ax.set_xticklabels(xlabels, fontsize=15, fontweight="bold", rotation=15)
        ax.set_yticklabels(ylabels, fontsize=15, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=17, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
        ax.set_title(title, fontsize=18, fontweight="bold")
        threshold = mat.max() * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "white" if v > threshold else "#37474F"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=13, color=color, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

    draw_heatmap(
        axes[0], enc_self, "Blues",
        "Encoder Self-Attention\n(Layer 2, Head 평균)",
        src_labels, src_labels,
        "Key (source)", "Query (source)",
    )
    draw_heatmap(
        axes[1], dec_cross, "YlOrRd",
        "Decoder Cross-Attention\n(Layer 2, Head 평균) — anti-diagonal 패턴 ⭐",
        src_labels, pred_labels[:Lt],
        "Key (source)", "Query (target)",
    )

    fig.tight_layout()
    out = RESULTS_DIR / "fig_09_attention_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════════
# 10. CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer From Scratch — 학습/시각화")
    p.add_argument("--mode", choices=["train", "visualize", "all"], default="all")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--retrain", action="store_true",
                   help="캐시 무시하고 다시 학습")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"train", "all"}:
        if args.retrain or not MODEL_CACHE.exists():
            model, history = train_model(epochs=args.epochs)
        else:
            model, history = get_or_train_model()
    else:
        model, history = get_or_train_model()

    if args.mode in {"visualize", "all"}:
        print("\n[ visualize ] 9 figures …")
        paths = [
            visualize_architecture(),                       # fig_01
            visualize_qkv_search_engine(),                  # fig_02
            visualize_scaled_dot_product_attention(),       # fig_03
            visualize_multihead_attention(),                # fig_04
            visualize_positional_encoding(),                # fig_05
            visualize_masked_attention(),                   # fig_06
            visualize_training_curve(history),              # fig_07
            visualize_multihead_pattern_comparison(model),  # fig_08
            visualize_attention_heatmap(model),             # fig_09
        ]
        for p in paths:
            print(f"  - saved: {p.relative_to(ROOT_DIR)}")

    # 추론 샘플 출력
    if args.mode in {"all", "visualize"}:
        print("\n[ sample inference ]")
        examples = generate_reverse_data(3, seed=999)[0]
        for i in range(3):
            src_i = examples[i:i+1]
            pred, _ = run_inference(model, src_i)
            src_str = " ".join(token_to_label(t.item()) for t in src_i[0])
            pred_str = " ".join(token_to_label(t.item()) for t in pred[0])
            print(f"  SRC : {src_str}")
            print(f"  PRED: {pred_str}\n")


if __name__ == "__main__":
    main()
