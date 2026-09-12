"""inference.py: interactive text generation with a trained Koi model.

Usage:
    python -m koi.inference
    python -m koi.inference --max-new-tokens 500 --temperature 0.9 --top-k 50
    python -m koi.inference --prompt "Once upon a time"   # single shot mode
"""

import argparse
from pathlib import Path

import torch

from .model import KoiConfig, KoiForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "model.pth"


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train a fish first with: python -m koi.train"
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = KoiConfig(**ckpt["config"])
    model = KoiForCausalLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    name = ckpt.get("model_name", "Koi")
    print(f"[load] '{name}' -- {model.num_parameters() / 1e6:.1f}M params, "
          f"{len(itos)} vocab tokens, trained through epoch {ckpt.get('epoch', '?')} "
          f"(step {ckpt.get('global_step', '?')}).")
    print(f"[load] layer plan: {model.layer_plan()}  (G=Gated DeltaNet, L=local dense)")
    return model, cfg, stoi, itos, cfg.eot_id, cfg.pad_id


def encode(text, stoi, unk_char=" "):
    unk_id = stoi.get(unk_char, 0)
    return [stoi.get(c, unk_id) for c in text]


def decode(ids, itos, pad_id):
    return "".join(itos[i] for i in ids if i != pad_id)


def run_generation(model, stoi, itos, eot_id, pad_id, device, prompt,
                   max_new_tokens, temperature, top_k, top_p):
    if len(prompt) == 0:
        prompt = "\n"
    ids = encode(prompt, stoi)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p > 0 else None,
            eot_id=eot_id,
        )
    return decode(out[0].tolist(), itos, pad_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--prompt", type=str, default=None,
                        help="generate once from this prompt and exit instead of interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, stoi, itos, eot_id, pad_id = load_model(Path(args.checkpoint), device)

    if args.prompt is not None:
        text = run_generation(model, stoi, itos, eot_id, pad_id, device, args.prompt,
                              args.max_new_tokens, args.temperature, args.top_k, args.top_p)
        print(text)
        return

    print("\nKoi is ready. Type a prompt and press Enter "
          "(Ctrl+C or 'quit' to exit). The goblin waits.\n")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        text = run_generation(model, stoi, itos, eot_id, pad_id, device, prompt,
                              args.max_new_tokens, args.temperature, args.top_k, args.top_p)
        print(text)
        print()


if __name__ == "__main__":
    main()
