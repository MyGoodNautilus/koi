"""train.py: trains Koi, a hybrid character level language model.

Every odd layer is a Gated DeltaNet implementation, every even layer is local
dense attention. Eats every .txt file found in ./dataset.

Usage:
    python -m koi.train [--epochs 10] [--batch-size 1] [--grad-accum 8] ...
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import KoiConfig, KoiForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "model.pth"

PAD_TOKEN = "<pad>"
EOT_TOKEN = "<end of text>"


# ---------------------------------------------------------------------------
# vocabulary + dataset
# ---------------------------------------------------------------------------

def build_vocab(txt_files):
    chars = set()
    for fp in txt_files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            chars.update(f.read())
    chars = sorted(chars)
    itos = list(chars) + [EOT_TOKEN, PAD_TOKEN]
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos, stoi[EOT_TOKEN], stoi[PAD_TOKEN]


class PackedCharDataset(Dataset):
    """Each example is one window of context_len + 1 ids: input = window[:-1],
    target = window[1:]. Files end with <end of text>, the last short window
    gets <pad> stuffed in so even tiny files feed the goblin."""

    def __init__(self, txt_files, stoi, eot_id, pad_id, context_len):
        self.context_len = context_len
        self.pad_id = pad_id
        window = context_len + 1
        sequences = []

        for fp in txt_files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            ids = [stoi[c] for c in text] + [eot_id]
            n = len(ids)
            if n == 0:
                continue
            for i in range(0, n, context_len):
                chunk = ids[i:i + window]
                if len(chunk) < window:
                    chunk = chunk + [pad_id] * (window - len(chunk))
                sequences.append(np.asarray(chunk, dtype=np.int32))
                if i + window >= n:
                    break

        if not sequences:
            raise RuntimeError("No training data produced. Is the dataset folder empty?")
        self.data = np.stack(sequences, axis=0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        return seq[:-1], seq[1:]


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, cfg, stoi, itos, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pth.tmp")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg.__dict__,
        "stoi": stoi,
        "itos": itos,
        "model_name": "Koi",
        "arch": "koi-hybrid-1",  # odd layers Gated DeltaNet, even layers local dense
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)  # atomic-ish replace, no half eaten checkpoints


def try_load_checkpoint(path, device):
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[resume] Checkpoint at {path} failed to load ({e}). Starting fresh.")
        return None
    if "local_window" not in ckpt.get("config", {}):
        # checkpoint from the old all-delta architecture. weights will not fit,
        # the layer plan changed. do not force feed the fish
        print("[resume] Checkpoint predates the hybrid Koi architecture "
              "(no local_window in config). Starting fresh instead.")
        return None
    print(f"[resume] Found checkpoint at {path} "
          f"(epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}). Resuming.")
    return ckpt


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def generate_sample(model, stoi, itos, eot_id, pad_id, device, prompt=None, max_new_tokens=400):
    model.eval()
    if not prompt:
        prompt = random.choice([" ", "\n", "The ", "Once "])
        prompt = "".join(c for c in prompt if c in stoi) or itos[0]
    ids = [stoi.get(c, eot_id) for c in prompt]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8,
                         top_k=40, eot_id=eot_id)
    text = "".join(itos[i] for i in out[0].tolist() if i != pad_id)
    model.train()
    return text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--local-window", type=int, default=1024,
                        help="sliding window width for the local dense layers")
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] No CUDA. Training will crawl at 16k context. The goblin proceeds anyway.")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    print(f"[amp] autocast dtype: {amp_dtype}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Expected a 'dataset' folder with .txt files at {DATASET_DIR}. "
            f"Create it and drop your training text in."
        )
    txt_files = sorted(DATASET_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {DATASET_DIR}.")
    print(f"[data] found {len(txt_files)} .txt file(s) in {DATASET_DIR}")

    # resume support: reuse vocab/config from an existing checkpoint if it fits
    existing_ckpt = try_load_checkpoint(CHECKPOINT_PATH, device)

    if existing_ckpt is not None:
        stoi = existing_ckpt["stoi"]
        itos = existing_ckpt["itos"]
        eot_id = stoi[EOT_TOKEN]
        pad_id = stoi[PAD_TOKEN]
        cfg = KoiConfig(**existing_ckpt["config"])
        print(f"[vocab] reusing vocabulary from checkpoint ({len(itos)} tokens).")
    else:
        stoi, itos, eot_id, pad_id = build_vocab(txt_files)
        print(f"[vocab] built from dataset: {len(itos)} tokens "
              f"({len(itos) - 2} characters + <end of text> + <pad>).")
        d_model = args.d_model
        head_dim = d_model // args.n_heads
        d_ff = int(round((8 / 3 * d_model) / 64)) * 64
        cfg = KoiConfig(
            vocab_size=len(itos),
            pad_id=pad_id,
            eot_id=eot_id,
            context_len=args.context_len,
            d_model=d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            head_dim=head_dim,
            d_ff=d_ff,
            chunk_size=args.chunk_size,
            local_window=args.local_window,
        )

    print(f"[data] packing into {cfg.context_len}-token windows...")
    dataset = PackedCharDataset(txt_files, stoi, eot_id, pad_id, cfg.context_len)
    print(f"[data] {len(dataset)} training windows of length {cfg.context_len}.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device == "cuda"),
        # no worker respawn churn on tiny datasets, epoch restarts are pure overhead
        persistent_workers=(args.num_workers > 0),
    )

    model = KoiForCausalLM(cfg).to(device)
    model.enable_gradient_checkpointing()  # 16k tokens want the memory back
    print(f"[model] Koi -- {model.num_parameters() / 1e6:.1f}M parameters, "
          f"{cfg.n_layers} layers, d_model={cfg.d_model}, heads={cfg.n_heads}, "
          f"context={cfg.context_len}, local_window={cfg.local_window}")
    print(f"[model] layer plan: {model.layer_plan()}  (G=Gated DeltaNet, L=local dense)")

    if args.compile:
        try:
            model = torch.compile(model)
            print("[model] torch.compile enabled.")
        except Exception as e:
            print(f"[model] torch.compile failed ({e}); going without.")

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device == "cuda"),
    )

    # ceiling divide: small datasets used to floor this to 0 and the scheduler
    # never stepped. the end of epoch flush below keeps the count honest
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(steps_per_epoch * args.epochs, 1)

    warmup_steps = args.warmup_steps
    if warmup_steps >= total_steps:
        # warmup alone would eat the whole run and LR never reaches its peak.
        # goblin clamps the ramp
        warmup_steps = max(1, total_steps // 10)
        print(f"[warmup] --warmup-steps ({args.warmup_steps}) >= total steps ({total_steps}). "
              f"Clamping warmup to {warmup_steps}.")

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0
    if existing_ckpt is not None:
        try:
            (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(
                existing_ckpt["model_state"]
            )
            optimizer.load_state_dict(existing_ckpt["optimizer_state"])
            if existing_ckpt.get("scheduler_state"):
                scheduler.load_state_dict(existing_ckpt["scheduler_state"])
            start_epoch = existing_ckpt.get("epoch", 0)
            global_step = existing_ckpt.get("global_step", 0)
            print(f"[resume] restored model/optimizer/scheduler. "
                  f"Continuing from epoch {start_epoch}, step {global_step}.")
        except Exception as e:
            print(f"[resume] could not restore optimizer/scheduler ({e}); "
                  f"continuing with fresh states where it failed.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[train] starting training loop.")
    epoch = start_epoch  # defined up front so the crash handler always has it
    try:
        for epoch in range(start_epoch, args.epochs):
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_count = 0

            for i, (input_ids, targets) in enumerate(pbar):
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                    _, loss = model(input_ids, labels=targets, ignore_index=pad_id)
                    loss_to_backprop = loss / args.grad_accum

                loss_to_backprop.backward()
                cur_loss = loss.item()
                running_loss += cur_loss
                running_count += 1

                # tick the bar every micro batch, else it freezes for grad_accum steps
                avg_loss = running_loss / max(1, running_count)
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "ppl": f"{math.exp(min(cur_loss, 20)):.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "accum": f"{running_count}/{args.grad_accum}",
                    "step": global_step,
                })

                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    running_loss, running_count = 0.0, 0

                    if global_step % args.save_every_steps == 0:
                        save_checkpoint(CHECKPOINT_PATH,
                                        model._orig_mod if hasattr(model, "_orig_mod") else model,
                                        optimizer, scheduler, epoch, global_step, cfg, stoi, itos)

            # flush leftover gradients from a partial accum group, else an epoch
            # that is not a multiple of grad_accum silently updates nothing and
            # the compute gets eaten for free. waste not, say the goblins
            if running_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = running_loss / running_count
                print(f"[train] flushed leftover gradients at end of epoch {epoch + 1} "
                      f"(step {global_step}, loss {avg_loss:.4f}, "
                      f"lr {scheduler.get_last_lr()[0]:.2e}).")
                running_loss, running_count = 0.0, 0

            # end of epoch: checkpoint + a sample, free snack included
            save_checkpoint(CHECKPOINT_PATH,
                            model._orig_mod if hasattr(model, "_orig_mod") else model,
                            optimizer, scheduler, epoch + 1, global_step, cfg, stoi, itos)
            print(f"\n[checkpoint] saved {CHECKPOINT_PATH} after epoch {epoch + 1}.")

            print(f"[sample] generating text sample after epoch {epoch + 1}...")
            sample = generate_sample(model, stoi, itos, eot_id, pad_id, device,
                                     max_new_tokens=args.sample_tokens)
            print("=" * 70)
            print(sample)
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n[train] interrupted. saving checkpoint before exit...")
        save_checkpoint(CHECKPOINT_PATH,
                        model._orig_mod if hasattr(model, "_orig_mod") else model,
                        optimizer, scheduler, epoch, global_step, cfg, stoi, itos)
        print("[train] checkpoint saved. re-run to resume.")
        raise
    except Exception:
        print("\n[train] crash detected. attempting emergency checkpoint save...")
        try:
            save_checkpoint(CHECKPOINT_PATH,
                            model._orig_mod if hasattr(model, "_orig_mod") else model,
                            optimizer, scheduler, epoch, global_step, cfg, stoi, itos)
            print("[train] checkpoint saved. re-run to resume training.")
        except Exception as save_err:
            print(f"[train] could not save an emergency checkpoint: {save_err}")
        raise

    print("[train] training complete.")


if __name__ == "__main__":
    main()
