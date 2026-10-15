"""Авторский набор: Song Describer Dataset (Manco et al., 2023) + метаданные MTG-Jamendo.

Метка типа вокала извлекается из текстовых описаний слушателей (captions) правилами по ключевым словам.
Сами описания НЕ подаются в модели (иначе утечка): текстовая модальность = исполнитель | альбом | название трека,
как и в MTAT. Дата релиза берётся из raw.meta.tsv MTG-Jamendo.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sdd"
OUT = ROOT / "data" / "processed"

PAT = {
    "male": r"\bmale\b|\bman\b|\bmen\b|\bhis voice\b|\bboy\b|\bbaritone\b|\btenor\b",
    "female": r"\bfemale\b|\bwoman\b|\bwomen\b|\bgirl\b|\bher voice\b|\bsoprano\b|\blady\b",
    "group": r"\bchoir\b|\bchoral\b|\bduet\b|\bgroup of (singers|voices)\b|\bchorus of\b|\bvoices\b",
    "instr": r"\binstrumental\b|\bno vocals?\b|without (any )?(vocals?|lyrics|singing|voices?)|\bno (singing|lyrics|voices?)\b",
}


def read_jamendo_meta(path):
    """В raw.meta.tsv встречаются табуляции внутри названий: id — первые 3 поля, дата и URL — последние 2."""
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        f = ln.split("\t")
        mid = f[3:-2]
        if len(mid) != 3:  # лишние табы -> склеиваем хвост в название альбома
            mid = [mid[0], mid[1] if len(mid) > 1 else "", " ".join(mid[2:])]
        rows.append([f[0], f[1], f[2], *mid, f[-2], f[-1]])
    return pd.DataFrame(rows, columns=["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TRACK_NAME", "ARTIST_NAME",
                                       "ALBUM_NAME", "RELEASEDATE", "URL"])


def label(caps):
    s = " ".join(caps).lower()
    h = {k for k, p in PAT.items() if re.search(p, s)}
    voc = h & {"male", "female", "group"}
    if "instr" in h:
        return "instrumental" if not voc else None  # противоречие -> исключаем
    if "group" in h or {"male", "female"} <= h:
        return "group"
    if h == {"male"}:
        return "male"
    if h == {"female"}:
        return "female"
    return None


def main():
    cap = pd.read_csv(RAW / "sdd.csv")
    meta = read_jamendo_meta(RAW / "raw.meta.tsv")
    meta["track_id"] = meta.TRACK_ID.str.replace("track_", "").astype(int)
    g = cap.groupby("track_id").agg(captions=("caption", list), n_captions=("caption", "size"),
                                    artist_id=("artist_id", "first"), path=("path", "first")).reset_index()
    g["label"] = g.captions.map(label)
    stats = {"tracks_total": int(len(g)), "captions_total": int(len(cap)),
             "no_vocal_info": int(g.label.isna().sum())}
    g = g[g.label.notna()].merge(meta[["track_id", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME", "RELEASEDATE"]],
                                 on="track_id", how="left")
    stats["missing_meta"] = int(g.TRACK_NAME.isna().sum())
    g["text"] = (g.ARTIST_NAME.fillna("") + " | " + g.ALBUM_NAME.fillna("") + " | " + g.TRACK_NAME.fillna(""))
    g["release_year"] = pd.to_datetime(g.RELEASEDATE, errors="coerce").dt.year
    stats["labeled_tracks"] = int(len(g))
    stats["class_counts"] = g.label.value_counts().to_dict()
    stats["release_year"] = g.release_year.describe()[["min", "25%", "50%", "75%", "max"]].to_dict()
    stats["unique_artists"] = int(g.artist_id.nunique())
    OUT.mkdir(parents=True, exist_ok=True)
    g.drop(columns=["captions"]).to_csv(OUT / "sdd_labeled.csv", index=False)
    g[["track_id", "label", "captions"]].to_json(OUT / "sdd_label_evidence.jsonl", orient="records", lines=True,
                                                force_ascii=False)
    (OUT / "sdd_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
