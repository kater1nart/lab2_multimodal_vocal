"""Анализ ошибок, конфликты модальностей, latency/стоимость и графики для отчёта ЛР2."""
import io
import time
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import features as fx

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figs"
CLASSES = ["male", "female", "group", "instrumental"]
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
NAMES = {"clip_zeroshot": "CLIP zero-shot", "only_text": "Только текст", "only_ts": "Только ряд",
         "only_img": "Только изображение", "early_fusion": "Ранняя fusion", "late_fusion": "Поздняя fusion",
         "stacking": "Стекинг", "gated_mlp": "Gated-MLP"}


def bar(df, path, title):
    d = df.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    col = ["#4472C4" if m in ("stacking", "late_fusion", "early_fusion", "gated_mlp") else "#A5A5A5" for m in d.model]
    ax.barh([NAMES[m] for m in d.model], d.macro_f1, color=col,
            xerr=[d.macro_f1 - d.macro_f1_ci_lo, d.macro_f1_ci_hi - d.macro_f1], capsize=3)
    ax.axvline(0.25, color="black", ls="--", lw=1)
    for y, v in enumerate(d.macro_f1):
        ax.text(0.01, y, f"{v:.3f}".replace(".", ","), va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 (95% ДИ, блочный bootstrap по исполнителям)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RES / "comparison_mtat_test.csv")
    bar(comp, FIG / "mtat_comparison.png", "MTAT test, 1 009 клипов")
    tr = pd.read_csv(RES / "transfer_sdd.csv")
    bar(tr[tr.period == "все"], FIG / "sdd_transfer.png", "Перенос на Jamendo (SDD), 281 трек")

    pr = pd.read_csv(RES / "predictions_mtat_test.csv")
    out = ["# Анализ ошибок ЛР2 (MTAT test)\n"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, m in zip(axs, ["only_ts", "stacking"]):
        cm = confusion_matrix(pr.label, pr[m], labels=CLASSES)
        ax.imshow(cm, cmap="Greys", vmax=cm.max() * 1.6)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks(range(4), CLASSES, rotation=20)
        ax.set_yticks(range(4), CLASSES)
        ax.set_title(NAMES[m])
        ax.set_xlabel("предсказание")
        ax.set_ylabel("истина")
        out.append(f"## Матрица ошибок: {NAMES[m]}\n")
        out.append(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_markdown() + "\n")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_mtat.png", dpi=150)
    plt.close(fig)

    # согласие / конфликт модальностей
    ok = {m: pr[m] == pr.label for m in ["only_text", "only_ts", "only_img", "stacking"]}
    n_agree = (pr.only_text == pr.only_ts).astype(int) + (pr.only_ts == pr.only_img).astype(int) + (pr.only_text == pr.only_img).astype(int)
    pr["agreement"] = np.select([(pr.only_text == pr.only_ts) & (pr.only_ts == pr.only_img), n_agree == 1], ["все 3 согласны", "2 из 3"], "все разные")
    rows = []
    for a, g in pr.groupby("agreement"):
        rows.append({"согласие модальностей": a, "n": len(g), "acc текст": (g.only_text == g.label).mean(),
                     "acc ряд": (g.only_ts == g.label).mean(), "acc изображение": (g.only_img == g.label).mean(),
                     "acc стекинг": (g.stacking == g.label).mean()})
    out.append("## Согласие одномодальных моделей и точность\n")
    out.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")
    rescued = pr[~ok["only_ts"] & ok["stacking"]]
    broken = pr[ok["only_ts"] & ~ok["stacking"]]
    out.append(f"Стекинг исправил ошибки модели «только ряд» в {len(rescued)} случаях и внёс новые ошибки в {len(broken)} случаях.\n")
    out.append("## Примеры: стекинг исправил ошибку ряда (подсказала другая модальность)\n")
    out.append(rescued[["id", "artist", "title", "label", "only_text", "only_ts", "only_img", "stacking"]].head(6).to_markdown(index=False) + "\n")
    out.append("## Примеры: ошибки стекинга\n")
    err = pr[~ok["stacking"]]
    out.append(err.groupby(["label", "stacking"]).head(1)[["id", "artist", "title", "label", "only_text", "only_ts", "only_img", "stacking"]].head(10).to_markdown(index=False) + "\n")
    ex = err.groupby(["label", "stacking"]).size().sort_values(ascending=False).head(5)
    out.append("## Самые частые путаницы стекинга\n")
    out.append(ex.rename("n").to_frame().to_markdown() + "\n")

    # SDD: ошибки по классам
    ps = pd.read_csv(RES / "predictions_sdd.csv")
    out.append("## Перенос на SDD: точность по классам\n")
    acc = pd.DataFrame({m: (ps[m] == ps.label) for m in ["only_text", "only_ts", "only_img", "late_fusion", "stacking"]})
    out.append(acc.groupby(ps.label).mean().round(3).to_markdown() + "\n")

    # latency: извлечение признаков на 1 клип (1 ядро CPU) + CLIP
    lab1 = fx.LAB1 / "data" / "raw" / "mp3"
    files = sorted(lab1.glob("*/*.mp3"))[:30]
    t = {"decode": 0, "ts": 0, "spec_png": 0}
    for f in files:
        t0 = time.perf_counter(); y, _ = librosa.load(f, sr=fx.SR, mono=True, duration=fx.DUR); t["decode"] += time.perf_counter() - t0
        t0 = time.perf_counter(); fx.timeseries(y); t["ts"] += time.perf_counter() - t0
        t0 = time.perf_counter(); fx.render_spec(y, "/tmp/_lat.png"); t["spec_png"] += time.perf_counter() - t0
    lat = {k: v / len(files) for k, v in t.items()}
    lat["clip_mps"] = float(np.load(ROOT / "data" / "processed" / "mtat_clip.npz")["sec_per_img"])
    out.append("## Latency извлечения признаков, с на клип (Apple M3 Pro, 1 ядро CPU; CLIP на GPU MPS)\n")
    out.append(pd.Series(lat).round(4).to_frame("сек").to_markdown() + "\n")
    pd.Series(lat).to_csv(RES / "latency_features.csv")
    (RES / "error_analysis.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
