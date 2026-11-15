#!/usr/bin/env python
"""
Standalone evaluation for the DeMamba (X-CLIP + Mamba) AI-generated-video detector
trained on GenVideo-100K.  Weights: https://huggingface.co/Ahsanz/DeMamba-GenVideo100K

One file, no repo needed. The checkpoint already contains the fine-tuned X-CLIP
encoder, so nothing else has to be downloaded.

    pip install torch "transformers>=4.30" opencv-python numpy pandas scikit-learn huggingface_hub tqdm

Input (pick one):
  --data_root DIR   Walk DIR for videos. Label comes from the path:
                    a folder named "real" -> 0 (real), a folder named "fake" -> 1 (fake),
                    and the folder right after "fake" is the generator name.
                    The GenVideo-Val layout works as-is:  Real/*.mp4,  Fake/<Generator>/*.mp4
                    Videos under neither folder are still predicted, just not scored.
  --csv FILE        Columns: path, label (0 real / 1 fake), optional: generator

Examples:
  python eval_demamba_standalone.py --data_root /data/GenVideo-Val --out_dir results
  python eval_demamba_standalone.py --csv my_videos.csv --ckpt /models/best_acc.pth
  python eval_demamba_standalone.py --data_root /data/vids --num_clips 4   # average 4 windows/video

Outputs (in --out_dir):
  predictions.csv   one row per video: path, label, generator, prob_fake, pred
  metrics.txt       accuracy / AUC / AP, real recall, per-generator recall + AP
"""
import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import XCLIPVisionConfig, XCLIPVisionModel

HF_REPO = "Ahsanz/DeMamba-GenVideo100K"
HF_FILE = "best_acc.pth"

# Preprocessing, identical to training/eval
NUM_FRAMES = 8            # frames per clip fed to the model
MAX_FRAMES = 32           # max sampled frames kept per video
IMG_SIZE = 224
CROP_PCT = 0.10           # center crop: trim 10% off each side of the longer dimension
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".m4v", ".mpg", ".mpeg", ".flv", ".wmv"}

# X-CLIP-base-patch16 vision tower (weights come from the checkpoint)
XCLIP_CFG = dict(
    hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
    mit_hidden_size=512, mit_intermediate_size=2048, mit_num_hidden_layers=1,
    mit_num_attention_heads=8, image_size=224, patch_size=16, num_frames=8,
    hidden_act="quick_gelu", layer_norm_eps=1e-5, attention_dropout=0.0, dropout=0.0,
    drop_path_rate=0.0,
)


# =========================================================================== #
# Mamba (pure PyTorch, from alxndrTL/mamba.py as used by DeMamba), inference only
# =========================================================================== #
def _npo2(n):
    return 2 ** math.ceil(math.log2(n))


def _pscan_inplace(A, X):
    # A, X : (B, D, L, N), L a power of two. Blelloch scan; modifies X in place:
    # H[t] = A[t] * H[t-1] + X[t]
    B, D, L, _ = A.size()
    num_steps = int(math.log2(L))

    Aa, Xa = A, X
    for _ in range(num_steps - 2):
        T = Xa.size(2)
        Aa = Aa.view(B, D, T // 2, 2, -1)
        Xa = Xa.view(B, D, T // 2, 2, -1)
        Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
        Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
        Aa = Aa[:, :, :, 1]
        Xa = Xa[:, :, :, 1]

    if Xa.size(2) == 4:
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
    elif Xa.size(2) == 2:
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        return
    else:
        return

    Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
    Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
    Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
    Aa[:, :, 2].mul_(Aa[:, :, 1])

    for k in range(num_steps - 3, -1, -1):
        Aa = A[:, :, 2 ** k - 1:L:2 ** k]
        Xa = X[:, :, 2 ** k - 1:L:2 ** k]
        T = Xa.size(2)
        Aa = Aa.view(B, D, T // 2, 2, -1)
        Xa = Xa.view(B, D, T // 2, 2, -1)
        Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
        Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])


def pscan(A_in, X_in):
    # A_in, X_in : (B, L, D, N) -> H : (B, L, D, N)
    L = X_in.size(1)
    if L == _npo2(L):
        A, X = A_in.clone(), X_in.clone()
    else:
        pad = (0, 0, 0, 0, 0, _npo2(L) - L)
        A, X = F.pad(A_in, pad), F.pad(X_in, pad)
    A, X = A.transpose(2, 1), X.transpose(2, 1)
    _pscan_inplace(A, X)
    return X.transpose(2, 1)[:, :L]


@dataclass
class MambaConfig:
    d_model: int = 768
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    bias: bool = False
    conv_bias: bool = True
    bimamba: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, c: MambaConfig):
        super().__init__()
        self.config = c
        self.in_proj = nn.Linear(c.d_model, 2 * c.d_inner, bias=c.bias)
        self.conv1d = nn.Conv1d(c.d_inner, c.d_inner, kernel_size=c.d_conv, bias=c.conv_bias,
                                groups=c.d_inner, padding=c.d_conv - 1)
        self.x_proj = nn.Linear(c.d_inner, c.dt_rank + 2 * c.d_state, bias=False)
        self.dt_proj = nn.Linear(c.dt_rank, c.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(c.d_inner, c.d_state))
        self.D = nn.Parameter(torch.ones(c.d_inner))
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=c.bias)
        self.bimamba = c.bimamba
        if self.bimamba:
            self.A_b_log = nn.Parameter(torch.zeros(c.d_inner, c.d_state))
            self.conv1d_b = nn.Conv1d(c.d_inner, c.d_inner, kernel_size=c.d_conv, bias=c.conv_bias,
                                      groups=c.d_inner, padding=c.d_conv - 1)
            self.x_proj_b = nn.Linear(c.d_inner, c.dt_rank + 2 * c.d_state, bias=False)
            self.dt_proj_b = nn.Linear(c.dt_rank, c.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(c.d_inner))

    def forward(self, x):
        _, L, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        y = self.ssm(F.silu(x))
        return self.out_proj(y * F.silu(z))

    def _scan(self, x, x_proj, dt_proj, A_log, D):
        c = self.config
        A = -torch.exp(A_log.float())
        delta, B, C = torch.split(x_proj(x), [c.dt_rank, c.d_state, c.d_state], dim=-1)
        delta = F.softplus(dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        BX = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        hs = pscan(deltaA, BX)
        return (hs @ C.unsqueeze(-1)).squeeze(3) + D.float() * x

    def ssm(self, x):
        y = self._scan(x, self.x_proj, self.dt_proj, self.A_log, self.D)
        if self.bimamba:
            # NB: the original implementation flips the channel dim (-1), kept as-is
            x_b = x.flip([-1])
            y_b = self._scan(x_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)
            y = y + y_b.flip([-1])
        return y


class ResidualBlock(nn.Module):
    def __init__(self, c: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(c)
        self.norm = RMSNorm(c.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x  # DropPath is identity at eval


# =========================================================================== #
# XCLIP_DeMamba
# =========================================================================== #
def reorder_data(data, N):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N * N, N))
        else:
            new_order.extend(range(col + N * (N - 1), col - 1, -N))
    new_order = torch.tensor(new_order, device=data.device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    return torch.gather(data, 2, index.expand_as(data))


class XCLIP_DeMamba(nn.Module):
    def __init__(self, class_num=1):
        super().__init__()
        self.encoder = XCLIPVisionModel(XCLIPVisionConfig(**XCLIP_CFG))
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14 // self.fusing_ratios) ** 2
        self.mamba = ResidualBlock(MambaConfig(d_model=channel))
        self.fc1 = nn.Linear((self.patch_nums + 1) * channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums * channel)
        self.fc_norm2 = nn.LayerNorm(768)

    def forward(self, x):
        # x : (B, T, 3, H, W) -> logits (B, 1)
        b, t, _, h, w = x.shape
        outputs = self.encoder(x.view(b * t, 3, h, w), output_hidden_states=False)
        seq = outputs["last_hidden_state"][:, 1:, :]
        c = seq.shape[-1]

        global_feat = self.fc_norm2(outputs["pooler_output"].reshape(b, t, -1).mean(1))

        seq = seq.view(b, t, -1, c)
        s = int(math.sqrt(seq.shape[2])) // self.fusing_ratios
        seq = seq.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = seq.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b * s * s, t, -1, c)
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b * s * s, -1, c)
        res = self.mamba(x)

        feats = self.fc_norm(res.mean(1).view(b, -1))
        return self.fc1(torch.cat((global_feat, feats), dim=1))


def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict", state)
    # saved from nn.DataParallel -> strip "module."
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    model_keys = set(model.state_dict().keys())
    if not set(sd) <= model_keys:
        # transformers 4.x nests the vision tower as encoder.vision_model.*, 5.x does not
        remapped = {}
        for k, v in sd.items():
            if k.startswith("encoder.vision_model."):
                k = "encoder." + k[len("encoder.vision_model."):]
            elif k.startswith("encoder."):
                k = "encoder.vision_model." + k[len("encoder."):]
            remapped[k] = v
        if set(remapped) <= model_keys:
            sd = remapped

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]  # buffer in some versions
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match model.\n  missing: {missing[:10]}\n"
                           f"  unexpected: {unexpected[:10]}")
    return state.get("epoch") if isinstance(state, dict) else None


# =========================================================================== #
# Video -> frames (same sampling as training: 2 fps if >= 4 s, else ceil(8/len) fps)
# =========================================================================== #
def _target_fps(duration):
    return 2 if duration >= 4 else math.ceil(8 / max(duration, 1.0))


def _preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if w > h:
        p = int(w * CROP_PCT)
        img = img[:, p:w - p]
    else:
        p = int(h * CROP_PCT)
        img = img[p:h - p, :]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - MEAN) / STD
    return img.transpose(2, 0, 1)  # (3, H, W)


def sample_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0:
        fps = 8.0
    duration = total / fps if total > 0 else 0.0
    stride = max(1, int(round(fps / (_target_fps(duration) if duration > 0 else 2))))

    frames, idx = [], 0
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(_preprocess(frame))
        idx += 1
    cap.release()

    if not frames:  # bad metadata: just take the first frames
        cap = cv2.VideoCapture(path)
        while len(frames) < NUM_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(_preprocess(frame))
        cap.release()
    return frames


def make_clips(frames, num_clips):
    if len(frames) < NUM_FRAMES:  # zero-pad short videos, as in training
        pad = [np.zeros((3, IMG_SIZE, IMG_SIZE), np.float32)] * (NUM_FRAMES - len(frames))
        return np.stack([np.stack(frames + pad)])
    last = len(frames) - NUM_FRAMES
    if num_clips <= 1:
        starts = [last // 2]
    else:
        starts = sorted(set(np.linspace(0, last, num_clips).round().astype(int).tolist()))
    return np.stack([np.stack(frames[s:s + NUM_FRAMES]) for s in starts])


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, num_clips):
        self.df = df.reset_index(drop=True)
        self.num_clips = num_clips

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = self.df.at[i, "path"]
        try:
            frames = sample_frames(path)
            clips = make_clips(frames, self.num_clips) if frames else None
        except Exception as e:  # corrupt file etc.
            print(f"[warn] failed to read {path}: {e}")
            clips = None
        return i, clips


def collate_list(batch):
    return batch


# =========================================================================== #
# Inputs
# =========================================================================== #
def label_from_path(rel_path):
    parts = Path(rel_path).parts[:-1]
    lower = [p.lower() for p in parts]
    if "fake" in lower:
        i = lower.index("fake")
        return 1, (parts[i + 1] if i + 1 < len(parts) else "fake")
    if "real" in lower:
        return 0, "real"
    return -1, ""


def build_index(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        if "path" not in df.columns:
            raise SystemExit("--csv needs a 'path' column (and 'label' to compute metrics)")
        if "label" not in df.columns:
            df["label"] = -1
        if "generator" not in df.columns:
            df["generator"] = np.where(df["label"] == 1, "fake", np.where(df["label"] == 0, "real", ""))
        base = Path(args.csv).resolve().parent
        df["path"] = [p if os.path.isabs(p) else str(base / p) for p in df["path"]]
    else:
        root = Path(args.data_root)
        rows = []
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__MACOSX"]  # macOS zip junk
            for f in files:
                if Path(f).suffix.lower() in VIDEO_EXTS and not f.startswith("._"):
                    full = Path(dirpath) / f
                    label, gen = label_from_path(full.relative_to(root))
                    rows.append({"path": str(full), "label": label, "generator": gen})
        df = pd.DataFrame(rows, columns=["path", "label", "generator"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("path").reset_index(drop=True)
    if args.max_videos and len(df) > args.max_videos:
        df = df.sample(n=args.max_videos, random_state=args.seed).sort_values("path").reset_index(drop=True)
    return df


# =========================================================================== #
# Metrics
# =========================================================================== #
def write_metrics(df, threshold, out_path):
    lines = []
    scored = df[(df["label"] >= 0) & df["prob_fake"].notna()]
    if scored.empty:
        lines.append("No labelled videos: predictions only (see predictions.csv).")
    else:
        y, p = scored["label"].values, scored["prob_fake"].values
        acc = (scored["pred"] == scored["label"]).mean()
        lines.append(f"Videos scored: {len(scored)}  (real={int((y == 0).sum())}, fake={int((y == 1).sum())})")
        lines.append(f"Threshold: {threshold}")
        lines.append(f"Overall Accuracy: {acc:.2%}")
        if len(set(y)) == 2:
            lines.append(f"Overall AUC: {roc_auc_score(y, p):.2%}")
            lines.append(f"Overall AP: {average_precision_score(y, p):.2%}")
        lines.append("")
        real = scored[scored["label"] == 0]
        fake = scored[scored["label"] == 1]
        if len(real):
            lines.append(f"Real recall (1-FPR): {(real['pred'] == 0).mean():.4f}  (n={len(real)})")
        if len(fake):
            lines.append(f"Fake recall (TPR):   {(fake['pred'] == 1).mean():.4f}  (n={len(fake)})")
        lines.append("")
        for gen, grp in fake.groupby("generator"):
            rec = (grp["pred"] == 1).mean()
            if len(real):
                comb = pd.concat([grp, real])
                gap = average_precision_score(comb["label"], comb["prob_fake"])
                lines.append(f"{gen:16s}  Recall={rec:.4f}  AP={gap:.4f}  (n={len(grp)})")
            else:
                lines.append(f"{gen:16s}  Recall={rec:.4f}  (n={len(grp)})")
    failed = df["prob_fake"].isna().sum()
    if failed:
        lines.append(f"\nUnreadable videos skipped: {failed} (prob_fake empty in predictions.csv)")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_root", help="folder of videos (labels from real/ and fake/<gen>/ folders)")
    src.add_argument("--csv", help="CSV with columns path,label[,generator]")
    ap.add_argument("--ckpt", default=None, help=f"local checkpoint; default downloads {HF_REPO}/{HF_FILE}")
    ap.add_argument("--out_dir", default="demamba_eval")
    ap.add_argument("--batch_size", type=int, default=8, help="videos per batch")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_clips", type=int, default=1,
                    help="8-frame windows per video, probabilities averaged (1 = center window)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--max_videos", type=int, default=None, help="random subset, for a quick check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.precision == "auto":
        args.precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]

    ckpt = args.ckpt
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(HF_REPO, HF_FILE)
    model = XCLIP_DeMamba()
    epoch = load_checkpoint(model, ckpt)
    model.to(device).eval()
    print(f"Loaded {ckpt} (epoch {epoch}) | device={device} precision={args.precision}")

    df = build_index(args)
    if df.empty:
        raise SystemExit("No videos found.")
    print(f"Videos: {len(df)}  labelled real={int((df.label == 0).sum())} "
          f"fake={int((df.label == 1).sum())} unlabelled={int((df.label < 0).sum())}")

    loader = torch.utils.data.DataLoader(
        VideoDataset(df, args.num_clips), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_list, pin_memory=device.type == "cuda")

    probs = np.full(len(df), np.nan)
    t0 = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            owners, clips = [], []
            for i, c in batch:
                if c is None:
                    continue
                owners += [i] * len(c)
                clips.append(c)
            if not clips:
                continue
            x = torch.from_numpy(np.concatenate(clips)).to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device.type, dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)
            p = logits.float()[:, 0].sigmoid().cpu().numpy()
            owners = np.array(owners)
            for i in np.unique(owners):
                probs[i] = p[owners == i].mean()

    df["prob_fake"] = probs
    df["pred"] = np.where(np.isnan(probs), -1, (probs > args.threshold).astype(int))
    df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    text = write_metrics(df, args.threshold, os.path.join(args.out_dir, "metrics.txt"))
    print(f"\nDone in {time.time() - t0:.0f}s. Wrote {args.out_dir}/predictions.csv and metrics.txt\n")
    print(text)


if __name__ == "__main__":
    main()
