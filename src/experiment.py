"""Единый протокол ЛР2: 8 моделей/схем интеграции, абляции, пропуск модальностей, permutation test, перенос на SDD.

Обучение — MTAT train, подбор гиперпараметров — MTAT val, основной тест — MTAT test (1 009 клипов, артисты не
пересекаются). Перенос — авторский набор SDD/Jamendo (344 трека, другой каталог, релизы 2005-2017).
"""
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RES = ROOT / "results"
LAB1 = ROOT.parents[1] / "1" / "lab1_vocal_llm"
CLASSES = ["male", "female", "group", "instrumental"]
SEED = 42
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
LOG = {"seed": SEED, "models": {}}


# ---------------------------------------------------------------- данные
def ts_stats(seq):
    """(N,20,20) -> (N,120): mean, std, min, max, наклон, средний |Δ| по времени для каждого из 20 рядов."""
    t = np.arange(seq.shape[1]) - (seq.shape[1] - 1) / 2
    slope = (seq * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([seq.mean(1), seq.std(1), seq.min(1), seq.max(1), slope,
                           np.abs(np.diff(seq, axis=1)).mean(1)], 1)


def load(which):
    if which == "mtat":
        df = pd.read_csv(LAB1 / "data" / "processed" / "dataset_final.csv")
        df["id"] = df.clip_id
        df["text"] = df.artist.fillna("") + " | " + df.album.fillna("") + " | " + df.title.fillna("")
        df["group_key"] = df.artist
    else:
        df = pd.read_csv(P / "sdd_labeled.csv")
        df["id"] = df.track_id
        df["split"] = "sdd"
        df["group_key"] = df.artist_id
    ts = np.load(P / f"{which}_ts.npz")
    cl = np.load(P / f"{which}_clip.npz")
    ids = set(ts["id"]) & set(cl["id"])
    df = df[df.id.isin(ids)].reset_index(drop=True)
    tsd = dict(zip(ts["id"], ts["seq"]))
    cld = dict(zip(cl["id"], cl["emb"]))
    zsd = dict(zip(cl["id"], cl["zs"]))
    df["y"] = df.label.map(CLASSES.index)
    return df, np.stack([tsd[i] for i in df.id]), np.stack([cld[i] for i in df.id]), np.stack([zsd[i] for i in df.id])


# ---------------------------------------------------------------- метрики
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return e


def metrics(p, y):
    pred = p.argmax(1)
    Y = np.eye(4)[y]
    return {"macro_f1": f1_score(y, pred, average="macro", labels=range(4), zero_division=0),
            "accuracy": accuracy_score(y, pred), "mcc": matthews_corrcoef(y, pred),
            "auroc_macro": roc_auc_score(Y, p, average="macro", multi_class="ovr") if len(set(y)) == 4 else np.nan,
            "brier": float(np.mean(np.sum((p - Y) ** 2, 1))), "ece": ece(p, y),
            **{f"f1_{c}": v for c, v in zip(CLASSES, f1_score(y, pred, average=None, labels=range(4), zero_division=0))}}


def block_bootstrap(fn, groups, n=1000):
    """Бутстрап по блокам-исполнителям (аналог блочного bootstrap во времени): учитывает зависимость клипов артиста."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(fn(s))
    return np.percentile(vals, [2.5, 97.5])


def perm_test(pa, pb, y, n=5000):
    """Парный permutation test для ΔMacro-F1 = F1(A) - F1(B): случайно меняем местами предсказания A и B по объектам."""
    a, b = pa.argmax(1), pb.argmax(1)
    f = lambda u, v: f1_score(y, u, average="macro", labels=range(4)) - f1_score(y, v, average="macro", labels=range(4))
    obs = f(a, b)
    cnt = 0
    for _ in range(n):
        sw = rng.rand(len(y)) < 0.5
        cnt += abs(f(np.where(sw, b, a), np.where(sw, a, b))) >= abs(obs)
    return obs, (cnt + 1) / (n + 1)


# ---------------------------------------------------------------- энкодеры модальностей
class Encoders:
    def fit(self, df, seq, emb):
        self.tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xt = hstack([self.tf_c.fit_transform(df.text), self.tf_w.fit_transform(df.text)]).tocsr()
        self.svd = TruncatedSVD(128, random_state=SEED).fit(Xt)
        self.sc_ts = StandardScaler().fit(ts_stats(seq))
        return self

    def text_sparse(self, df):
        return hstack([self.tf_c.transform(df.text), self.tf_w.transform(df.text)]).tocsr()

    def transform(self, df, seq, emb):
        return {"text": self.text_sparse(df), "text_dense": self.svd.transform(self.text_sparse(df)),
                "ts": self.sc_ts.transform(ts_stats(seq)), "ts_seq": seq, "img": emb}


def logreg(X, y, Xv, yv):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3, 10]:
        m = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(X, y)
        f = f1_score(yv, m.predict(Xv), average="macro")
        if best is None or f > best[0]:
            best = (f, C, m)
    return best[2], best[1]


UNI = {"text": "text", "ts": "ts", "img": "img"}


def fit_unimodal(F, y, Fv, yv):
    out = {}
    for name, key in UNI.items():
        m, C = logreg(F[key], y, Fv[key], yv)
        out[name] = m
        LOG["models"][f"uni_{name}"] = {"type": "LogisticRegression(balanced)", "C": C, "input": key}
    return out


def early_X(F, mods):
    return np.hstack([{"text": F["text_dense"], "ts": F["ts"], "img": F["img"]}[m] for m in mods])


# ---------------------------------------------------------------- gated fusion MLP (intermediate fusion)
class GatedFusion(nn.Module):
    def __init__(self, d_text=128, d_img=512, d_ts=20, h=128, p_drop_mod=0.3):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(d_text, h), nn.ReLU(), nn.Dropout(0.2))
        self.img = nn.Sequential(nn.Linear(d_img, h), nn.ReLU(), nn.Dropout(0.2))
        self.ts = nn.GRU(d_ts, h // 2, batch_first=True, bidirectional=True)  # временной ряд 20 шагов
        self.gate = nn.Linear(3 * h, 3)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 4))
        self.p = p_drop_mod

    def forward(self, t, i, s, mask=None):
        _, hs = self.ts(s)
        z = torch.stack([self.text(t), self.img(i), torch.cat([hs[0], hs[1]], -1)], 1)  # (B,3,h)
        if mask is None:
            mask = torch.ones(z.shape[0], 3)
            if self.training:  # modality dropout: учим работать при пропавшей модальности
                mask = (torch.rand(z.shape[0], 3) > self.p).float()
                mask[mask.sum(1) == 0, 0] = 1
        z = z * mask[..., None]
        g = self.gate(z.flatten(1)).masked_fill(mask == 0, -1e9).softmax(-1)  # веса модальностей
        return self.head((g[..., None] * z).sum(1)), g


def tensors(F, sc_seq):
    s = (F["ts_seq"] - sc_seq[0]) / sc_seq[1]
    return (torch.tensor(F["text_dense"], dtype=torch.float32), torch.tensor(F["img"], dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32))


def fit_gated(F, y, Fv, yv):
    sc = (F["ts_seq"].mean((0, 1)), F["ts_seq"].std((0, 1)) + 1e-6)
    T, Tv = tensors(F, sc), tensors(Fv, sc)
    yt = torch.tensor(y)
    w = torch.tensor(len(y) / (4 * np.bincount(y, minlength=4)), dtype=torch.float32)
    m = GatedFusion()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_state, patience = -1, None, 0
    for ep in range(80):
        m.train()
        perm = torch.randperm(len(y))
        for b in range(0, len(y), 128):
            ix = perm[b:b + 128]
            out, _ = m(T[0][ix], T[1][ix], T[2][ix])
            loss = nn.functional.cross_entropy(out, yt[ix], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            f = f1_score(yv, m(*Tv)[0].argmax(1).numpy(), average="macro")
        if f > best:
            best, best_state, patience = f, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 12:
                break
    m.load_state_dict(best_state)
    m.eval()
    LOG["models"]["gated_mlp"] = {"type": "GatedFusion (text MLP + CLIP MLP + BiGRU по ряду, softmax-гейт)",
                                  "hidden": 128, "modality_dropout": 0.3, "epochs": ep + 1, "val_macro_f1": best,
                                  "optimizer": "AdamW lr=1e-3 wd=1e-3", "batch": 128}

    def predict(Fx, mask=None):
        with torch.no_grad():
            Tx = tensors(Fx, sc)
            mk = None if mask is None else torch.tensor(np.tile(mask, (len(Tx[0]), 1)), dtype=torch.float32)
            out, g = m(*Tx, mask=mk)
            return out.softmax(-1).numpy(), g.numpy()
    return predict


# ---------------------------------------------------------------- основной эксперимент
def main():
    RES.mkdir(exist_ok=True)
    df, seq, emb, zs = load("mtat")
    sp = {s: (df.split == s).values for s in ["train", "val", "test"]}
    enc = Encoders().fit(df[sp["train"]], seq[sp["train"]], emb[sp["train"]])
    F = {s: enc.transform(df[sp[s]], seq[sp[s]], emb[sp[s]]) for s in sp}
    y = {s: df.y.values[sp[s]] for s in sp}
    groups_test = df.group_key.values[sp["test"]]

    probs, lat = {}, {}
    # M1 CLIP zero-shot (vision-language, без обучения)
    probs["clip_zeroshot"] = torch.tensor(zs[sp["test"]] * 100).softmax(-1).numpy()
    LOG["models"]["clip_zeroshot"] = {"type": "CLIP ViT-B/32 openai, zero-shot, logit_scale=100", "prompts": "см. clip_embed.py"}
    # M2-M4 unimodal
    uni = fit_unimodal(F["train"], y["train"], F["val"], y["val"])
    for k, m in uni.items():
        t0 = time.perf_counter()
        probs[f"only_{k}"] = m.predict_proba(F["test"][UNI[k]])
        lat[f"only_{k}"] = (time.perf_counter() - t0) / len(y["test"])
    # M5 early fusion
    em, C = logreg(early_X(F["train"], ["text", "ts", "img"]), y["train"], early_X(F["val"], ["text", "ts", "img"]), y["val"])
    probs["early_fusion"] = em.predict_proba(early_X(F["test"], ["text", "ts", "img"]))
    LOG["models"]["early_fusion"] = {"type": "LogReg на конкатенации [SVD128(text), ts120, CLIP512]", "C": C}
    # M6 late fusion: веса на симплексе (шаг 0,1) подбираются по val
    pv = {k: uni[k].predict_proba(F["val"][UNI[k]]) for k in uni}
    pt = {k: probs[f"only_{k}"] for k in uni}
    best = max(((a, b, round(1 - a - b, 1)) for a, b in product(np.arange(0, 1.01, 0.1), repeat=2) if a + b <= 1.0001),
               key=lambda w: f1_score(y["val"], (w[0] * pv["text"] + w[1] * pv["ts"] + w[2] * pv["img"]).argmax(1), average="macro"))
    W = dict(zip(["text", "ts", "img"], best))
    probs["late_fusion"] = sum(W[k] * pt[k] for k in W)
    LOG["models"]["late_fusion"] = {"type": "взвешенное среднее вероятностей unimodal", "weights_val": W}
    # M7 stacking: мета-LogReg на out-of-fold вероятностях (GroupKFold по артистам)
    oof = np.zeros((sp["train"].sum(), 12))
    gk = GroupKFold(5)
    Ftr, ytr = F["train"], y["train"]
    for tr, va in gk.split(ytr, ytr, df.group_key.values[sp["train"]]):
        for j, k in enumerate(uni):
            X = Ftr[UNI[k]]
            mm = LogisticRegression(C=LOG["models"][f"uni_{k}"]["C"], max_iter=4000, class_weight="balanced").fit(X[tr], ytr[tr])
            oof[va, 4 * j:4 * j + 4] = mm.predict_proba(X[va])
    meta = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced").fit(oof, ytr)
    stack_in = lambda p: np.hstack([p[k] for k in uni])
    probs["stacking"] = meta.predict_proba(stack_in(pt))
    LOG["models"]["stacking"] = {"type": "LogReg на OOF-вероятностях 3 unimodal (GroupKFold=5 по артистам)"}
    # M8 gated fusion MLP
    gpred = fit_gated(F["train"], y["train"], F["val"], y["val"])
    t0 = time.perf_counter()
    probs["gated_mlp"], gates = gpred(F["test"])
    lat["gated_mlp"] = (time.perf_counter() - t0) / len(y["test"])

    # ---- таблица сравнения
    rows = []
    yt = y["test"]
    for k, p in probs.items():
        r = {"model": k, **metrics(p, yt)}
        r["macro_f1_ci_lo"], r["macro_f1_ci_hi"] = block_bootstrap(
            lambda s: f1_score(yt[s], p[s].argmax(1), average="macro", labels=range(4)), groups_test)
        rows.append(r)
    comp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comp.to_csv(RES / "comparison_mtat_test.csv", index=False)

    # ---- прирост к лучшему unimodal + permutation test + блочный bootstrap Δ
    best_uni = max(["only_text", "only_ts", "only_img"], key=lambda k: f1_score(yt, probs[k].argmax(1), average="macro"))
    gains = []
    for k in ["early_fusion", "late_fusion", "stacking", "gated_mlp"]:
        obs, pval = perm_test(probs[k], probs[best_uni], yt)
        lo, hi = block_bootstrap(lambda s: f1_score(yt[s], probs[k][s].argmax(1), average="macro", labels=range(4))
                                 - f1_score(yt[s], probs[best_uni][s].argmax(1), average="macro", labels=range(4)), groups_test)
        gains.append({"model": k, "vs": best_uni, "delta_macro_f1": obs, "ci_lo": lo, "ci_hi": hi, "perm_p": pval})
    gains = pd.DataFrame(gains)
    gains.to_csv(RES / "gain_vs_best_unimodal.csv", index=False)

    # ---- абляции: early fusion без каждой модальности + пары
    abl = []
    for mods in [["text", "ts", "img"], ["ts", "img"], ["text", "img"], ["text", "ts"], ["text"], ["ts"], ["img"]]:
        m, _ = logreg(early_X(F["train"], mods), y["train"], early_X(F["val"], mods), y["val"])
        abl.append({"modalities": "+".join(mods), **metrics(m.predict_proba(early_X(F["test"], mods)), yt)})
    pd.DataFrame(abl).to_csv(RES / "ablation_early_fusion.csv", index=False)

    # ---- пропуск модальности на тесте (модель обучена со всеми модальностями)
    miss = []
    for name, mask in [("все", [1, 1, 1]), ("нет текста", [0, 1, 1]), ("нет изображения", [1, 0, 1]),
                       ("нет ряда", [1, 1, 0]), ("только текст", [1, 0, 0]), ("только изображение", [0, 1, 0]),
                       ("только ряд", [0, 0, 1])]:
        pg, _ = gpred(F["test"], mask=np.array(mask, dtype=np.float32))
        wm = {"text": mask[0], "img": mask[1], "ts": mask[2]}
        ws = sum(W[k] * wm[k] for k in W)
        pl = sum(W[k] * wm[k] * pt[k] for k in W) / ws if ws > 0 else np.full_like(pt["text"], 0.25)
        miss.append({"режим": name, "gated_mlp": metrics(pg, yt)["macro_f1"], "late_fusion": metrics(pl, yt)["macro_f1"]})
    pd.DataFrame(miss).to_csv(RES / "missing_modality.csv", index=False)

    # ---- permutation importance модальностей для gated MLP (перемешиваем модальность между объектами)
    imp = []
    base = metrics(probs["gated_mlp"], yt)["macro_f1"]
    for mod, keys in [("text", ["text_dense"]), ("img", ["img"]), ("ts", ["ts_seq"])]:
        drops = []
        for r in range(20):
            Fp = dict(F["test"])
            pi = rng.permutation(len(yt))
            for key in keys:
                Fp[key] = F["test"][key][pi]
            drops.append(base - metrics(gpred(Fp)[0], yt)["macro_f1"])
        imp.append({"modality": mod, "drop_mean": np.mean(drops), "drop_std": np.std(drops)})
    pd.DataFrame(imp).to_csv(RES / "modality_permutation_importance.csv", index=False)
    pd.DataFrame(gates, columns=["gate_text", "gate_img", "gate_ts"]).assign(y=[CLASSES[i] for i in yt]).groupby("y").mean() \
        .to_csv(RES / "gated_weights_by_class.csv")

    # ---- сохраняем предсказания для анализа ошибок
    pred_df = df[sp["test"]][["id", "label", "artist", "album", "title"]].copy()
    for k, p in probs.items():
        pred_df[k] = [CLASSES[i] for i in p.argmax(1)]
    pred_df.to_csv(RES / "predictions_mtat_test.csv", index=False)

    # ---- перенос на авторский набор SDD (Jamendo): модели обучены на MTAT train
    try:
        dfs, seqs, embs, zss = load("sdd")
        Fs = enc.transform(dfs, seqs, embs)
        ys = dfs.y.values
        ps = {"clip_zeroshot": torch.tensor(zss * 100).softmax(-1).numpy()}
        for k, m in uni.items():
            ps[f"only_{k}"] = m.predict_proba(Fs[UNI[k]])
        ps["early_fusion"] = em.predict_proba(early_X(Fs, ["text", "ts", "img"]))
        ps["late_fusion"] = sum(W[k] * ps[f"only_{k}"] for k in W)
        ps["stacking"] = meta.predict_proba(stack_in({k: ps[f"only_{k}"] for k in uni}))
        ps["gated_mlp"] = gpred(Fs)[0]
        period = np.where(dfs.release_year.values >= 2010, "2010-2017", "2005-2009")
        srows = []
        for k, p in ps.items():
            for per in ["все", "2005-2009", "2010-2017"]:
                s = np.ones(len(ys), bool) if per == "все" else period == per
                r = {"model": k, "period": per, "n": int(s.sum()), **metrics(p[s], ys[s])}
                if per == "все":
                    r["macro_f1_ci_lo"], r["macro_f1_ci_hi"] = block_bootstrap(
                        lambda ix: f1_score(ys[ix], p[ix].argmax(1), average="macro", labels=range(4)), dfs.group_key.values)
                srows.append(r)
        pd.DataFrame(srows).to_csv(RES / "transfer_sdd.csv", index=False)
        sd = dfs[["id", "label", "text", "release_year"]].copy()
        for k, p in ps.items():
            sd[k] = [CLASSES[i] for i in p.argmax(1)]
        sd.to_csv(RES / "predictions_sdd.csv", index=False)
        sg = []
        bu = max(["only_text", "only_ts", "only_img"], key=lambda k: f1_score(ys, ps[k].argmax(1), average="macro"))
        for k in ["early_fusion", "late_fusion", "stacking", "gated_mlp"]:
            obs, pval = perm_test(ps[k], ps[bu], ys)
            sg.append({"model": k, "vs": bu, "delta_macro_f1": obs, "perm_p": pval})
        pd.DataFrame(sg).to_csv(RES / "transfer_sdd_gain.csv", index=False)
    except FileNotFoundError as e:
        print("SDD пропущен:", e)

    LOG["latency_s_per_item_inference"] = lat
    LOG["late_fusion_weights"] = W
    LOG["best_unimodal_mtat"] = best_uni
    (RES / "params_log.json").write_text(json.dumps(LOG, ensure_ascii=False, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(comp.round(3).to_string(index=False))
    print(gains.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
