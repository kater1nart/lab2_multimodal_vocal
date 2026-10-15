# Лабораторная работа 2. Мультимодальные модели для определения типа фронт-вокала

Тема ВКР: *Разработка системы автоматической разметки музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.*
Продолжение ЛР1: https://github.com/kater1nart/lab1_vocal_llm

Модальности одного трека: **текст** (исполнитель | альбом | название), **временной ряд** (20 покадровых признаков звука × 20 шагов по 0,5 с),
**изображение** (лог-мел спектрограмма → CLIP ViT-B/32). Классы: male / female / group / instrumental.

## Результаты (MagnaTagATune test, 1 009 клипов, исполнители не пересекаются с train)

| Схема | Macro-F1 | Δ к лучшему unimodal | p (permutation) |
|---|---|---|---|
| Стекинг | **0.760** | +0.079 [0.007, 0.131] | 0.0002 |
| Поздняя fusion | 0.723 | +0.043 [0.018, 0.079] | 0.0006 |
| Ранняя fusion | 0.687 | +0.007 | 0.40 |
| Только временной ряд | 0.681 | — | — |
| Только изображение (CLIP) | 0.637 | | |
| Gated-MLP (intermediate) | 0.604 | −0.077 | 0.0002 |
| Только текст | 0.601 | | |
| CLIP zero-shot | 0.229 | | |

Перенос на Jamendo (Song Describer, 281 трек, 2005–2017): изображение 0.437, поздняя fusion 0.424 — прирост fusion исчезает.
Отчёт: `report/Отчет_ЛР2.docx`.

## Данные

- MagnaTagATune (CC BY-NC-SA 3.0): https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset — метки и split из ЛР1.
- Song Describer Dataset (CC BY-SA 4.0): https://zenodo.org/records/10072001, аудио — https://huggingface.co/datasets/renumics/song-describer-dataset
- MTG-Jamendo метаданные: https://github.com/MTG/mtg-jamendo-dataset (`data/raw.meta.tsv`)

## Воспроизведение

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt
# нужен соседний репозиторий ЛР1 (../../1/lab1_vocal_llm) с аудио MTAT и спектрограммами
.venv/bin/python src/sdd_labels.py
.venv/bin/python src/features.py mtat && .venv/bin/python src/features.py sdd
.venv/bin/python src/clip_embed.py mtat && .venv/bin/python src/clip_embed.py sdd
.venv/bin/python src/experiment.py      # 8 схем, абляции, permutation test, перенос
.venv/bin/python src/analysis.py        # анализ ошибок, latency, графики
.venv/bin/python src/build_report.py
```
Seed = 42.
