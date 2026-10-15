"""CLIP ViT-B/32 (OpenAI weights, open_clip): эмбеддинги изображений спектрограмм + zero-shot сходство с промптами.

Выход: data/processed/{mtat,sdd}_clip.npz: id, emb (N,512, L2-норм.), zs (N,4) — косинусные сходства с промптами классов.
"""
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
LAB1 = ROOT.parents[1] / "1" / "lab1_vocal_llm"
CLASSES = ["male", "female", "group", "instrumental"]
PROMPTS = {
    "male": "a mel spectrogram of music with a male singer",
    "female": "a mel spectrogram of music with a female singer",
    "group": "a mel spectrogram of music with a choir or several singers",
    "instrumental": "a mel spectrogram of instrumental music without vocals",
}


def main(which):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(dev).eval()
    if which == "mtat":
        ids = pd.read_csv(LAB1 / "data" / "processed" / "dataset_final.csv").clip_id.values
        paths = [LAB1 / "data" / "processed" / "spec" / f"{i}.png" for i in ids]
    else:
        ids = pd.read_csv(P / "sdd_labeled.csv").track_id.values
        paths = [P / "sdd_spec" / f"{i}.png" for i in ids]
        keep = [p.exists() for p in paths]
        ids, paths = ids[keep], [p for p, k in zip(paths, keep) if k]
    with torch.no_grad():
        t = model.encode_text(tok([PROMPTS[c] for c in CLASSES]).to(dev))
        t = t / t.norm(dim=-1, keepdim=True)
        embs = []
        t0 = time.perf_counter()
        for b in range(0, len(paths), 64):
            x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[b:b + 64]]).to(dev)
            e = model.encode_image(x)
            embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        emb = torch.cat(embs)
        sec_per_img = (time.perf_counter() - t0) / len(paths)
        zs = (emb.to(dev) @ t.T).cpu().numpy()
    np.savez(P / f"{which}_clip.npz", id=ids, emb=emb.numpy(), zs=zs, sec_per_img=sec_per_img)
    print(which, emb.shape, f"{sec_per_img * 1000:.1f} ms/img")


if __name__ == "__main__":
    main(sys.argv[1])
