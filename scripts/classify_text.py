#!/usr/bin/env python3
"""Classify text using the trained transformer's ArcFace head.

NOTE: For the step_00162000 checkpoint (pre-MoCo), the ArcFace head
operated on raw 768-dim pooled features and collapsed (all logits = -30).
Use classify_projection.py instead for that checkpoint.

Post-MoCo checkpoints have ArcFace on 384-dim projected features and
should work correctly.

Usage:
    python scripts/classify_text.py "Che boludo qué decís"
    python scripts/classify_text.py --interactive
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT = ROOT / "outputs" / "eigen3" / "transformer" / "checkpoints" / "step_00162000.pt"

VARIETY_NAMES = {
    0: ("ES_PEN", "Peninsular (Spain)"),
    1: ("ES_AND", "Andalusian"),
    2: ("ES_CAN", "Canarian"),
    3: ("ES_RIO", "Rioplatense (Argentina)"),
    4: ("ES_MEX", "Mexican"),
    5: ("ES_CAR", "Caribbean"),
    6: ("ES_CHI", "Chilean"),
    7: ("ES_AND_BO", "Andean-Bolivian"),
}


def load_model():
    from eigen3.model import DialectTransformer

    model = DialectTransformer(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        proj_dim=384,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, step={ckpt['global_step']}")
    del ckpt

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    return model, device


def classify(model, device, text: str, top_k: int = 3):
    """Classify a text using the ArcFace classification head."""
    tokenizer = model.tokenizer

    encoding = tokenizer(
        text,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        _, cls_logits, proj_emb = model(input_ids, attention_mask, labels=None)

    # cls_logits are ArcFace cosine logits (without margin, since labels=None)
    probs = F.softmax(cls_logits[0], dim=0).cpu().numpy()

    # Sort by probability
    ranked = sorted(enumerate(probs), key=lambda x: -x[1])

    return ranked, proj_emb[0].cpu().numpy()


def main():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    model, device = load_model()

    if len(sys.argv) > 1 and sys.argv[1] != "--interactive":
        text = " ".join(sys.argv[1:])
        ranked, _ = classify(model, device, text)
        code, name = VARIETY_NAMES[ranked[0][0]]
        print(f"\n  \"{text}\"")
        print(f"  -> {code} ({name})")
        print()
        for idx, prob in ranked:
            code, name = VARIETY_NAMES[idx]
            bar = "#" * int(prob * 40)
            print(f"  {code:>10} ({name:>25})  {prob:5.1%}  {bar}")
        return

    # Interactive mode
    print("\nDialect Classifier (type 'quit' to exit)")
    print("-" * 50)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break

        ranked, _ = classify(model, device, text)
        code, name = VARIETY_NAMES[ranked[0][0]]
        print(f"  -> {code} ({name})")
        for idx, prob in ranked[:4]:
            code, name = VARIETY_NAMES[idx]
            bar = "#" * int(prob * 40)
            print(f"     {code:>10}  {prob:5.1%}  {bar}")


if __name__ == "__main__":
    main()
