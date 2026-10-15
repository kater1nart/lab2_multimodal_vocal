"""Мультимодальный объект трека и извлечение модальностей.

Объект = (text, timeseries, image), все модальности строятся из информации, доступной в момент загрузки
трека в каталог (метаданные + аудио), и из ОДНОГО и того же 10-секундного окна аудио:
  - text: "исполнитель | альбом | название" (пользовательские теги MTAT и описания SDD НЕ используются);
  - timeseries: покадровые признаки окна (RMS, спектральные центроид/полоса/спад/плоскостность, ZCR,
    основной тон YIN, MFCC 1-13), усреднённые в 20 шагов по 0,5 с -> матрица 20x20;
  - image: PNG лог-мел спектрограммы окна (как в ЛР1) -> эмбеддинг CLIP ViT-B/32.
Окно: MTAT — первые 10 с 29-секундного клипа (как в ЛР1); SDD — 30-40 с 2-минутного фрагмента (пропуск интро).
"""
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
LAB1 = ROOT.parents[1] / "1" / "lab1_vocal_llm"
SR, DUR, N_STEPS = 22050, 10.0, 20
SDD_OFFSET = 30.0
TS_NAMES = ["rms_db", "centroid", "bandwidth", "rolloff", "flatness", "zcr", "f0_yin"] + [f"mfcc{i}" for i in range(1, 14)]


def timeseries(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    rms = librosa.amplitude_to_db(librosa.feature.rms(S=S)[0], ref=np.max)
    cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0] / 1000
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0] / 1000
    ro = librosa.feature.spectral_rolloff(S=S, sr=SR)[0] / 1000
    fl = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    f0 = librosa.yin(y, fmin=60, fmax=1000, sr=SR, frame_length=2048, hop_length=512) / 1000
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=14)[1:]
    n = min(len(rms), len(f0), mfcc.shape[1])
    X = np.vstack([rms[:n], cen[:n], bw[:n], ro[:n], fl[:n], zcr[:n], f0[:n], mfcc[:, :n]]).T  # (frames, 20)
    steps = np.array_split(np.arange(n), N_STEPS)
    return np.stack([X[s].mean(0) for s in steps]).astype(np.float32)  # (20, 20)


def render_spec(y, png):
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, fmax=8000), ref=np.max)
    fig = plt.figure(figsize=(4.48, 2.24), dpi=100)
    ax = fig.add_axes([0.11, 0.1, 0.88, 0.88])
    librosa.display.specshow(S_db, sr=SR, x_axis=None, y_axis="mel", fmax=8000, cmap="magma", ax=ax)
    ax.set_ylabel("")
    ax.tick_params(labelsize=6)
    fig.savefig(png)
    plt.close(fig)


def mtat_job(args):
    clip_id, mp3 = args
    try:
        y, _ = librosa.load(LAB1 / "data" / "raw" / "mp3" / mp3, sr=SR, mono=True, duration=DUR)
        return clip_id, timeseries(y)
    except Exception as e:
        print("ERR", clip_id, e, file=sys.stderr)
        return clip_id, None


def sdd_job(args):
    track_id, audio_bytes = args
    try:
        # Гармонизация с MTAT: аудио MTAT хранится в 16 кГц (нет энергии выше 8 кГц), поэтому Jamendo сначала
        # приводится к 16 кГц, затем, как и MTAT, ресемплируется в 22,05 кГц.
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, offset=SDD_OFFSET, duration=DUR)
        if len(y) < 16000 * 5:  # трек короче 35 с -> берём начало
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=DUR)
        y = librosa.resample(y, orig_sr=16000, target_sr=SR)
        png = P / "sdd_spec" / f"{track_id}.png"
        render_spec(y, png)
        return track_id, timeseries(y)
    except Exception as e:
        print("ERR", track_id, e, file=sys.stderr)
        return track_id, None


def run(jobs, fn, out):
    ids, seqs, bad = [], [], []
    with ProcessPoolExecutor(10) as ex:
        for k, (i, s) in enumerate(ex.map(fn, jobs, chunksize=8)):
            (bad.append(int(i)) if s is None else (ids.append(i), seqs.append(s)))
            if k % 1000 == 0:
                print(k, flush=True)
    np.savez(out, id=np.array(ids), seq=np.stack(seqs), bad=np.array(bad))
    print("saved", out, len(ids), "bad", len(bad))


def main(which):
    if which == "mtat":
        df = pd.read_csv(LAB1 / "data" / "processed" / "dataset_final.csv")
        run(list(zip(df.clip_id, df.mp3_path)), mtat_job, P / "mtat_ts.npz")
    else:
        import pyarrow.parquet as pq
        (P / "sdd_spec").mkdir(parents=True, exist_ok=True)
        lab = pd.read_csv(P / "sdd_labeled.csv")
        need = set(lab.track_id)
        jobs = []
        for f in sorted((ROOT / "data" / "raw" / "sdd").glob("train-*.parquet")):
            t = pq.read_table(f, columns=["track_id", "path"]).to_pandas()
            for tid, a in zip(t.track_id, t.path):
                if tid in need and tid not in {j[0] for j in jobs}:
                    jobs.append((tid, a["bytes"]))
        print("sdd jobs", len(jobs), "of", len(need))
        run(jobs, sdd_job, P / "sdd_ts.npz")


if __name__ == "__main__":
    main(sys.argv[1])
