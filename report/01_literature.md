## 1. Исследовательский обзор

**Тема ВКР:** разработка системы автоматической разметки музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.

**Постановка ЛР2 (единая для всех моделей):** классификация трека по типу ведущего вокала — `male`, `female`, `group` (хор / дуэт / смешанный состав), `instrumental`. Решение принимается по информации, доступной в момент загрузки трека в каталог: метаданные и аудио.

**Исследовательский вопрос:** даёт ли объединение трёх модальностей — текста (метаданные), числового временного ряда (покадровая динамика звука) и изображения (спектрограмма) — значимый прирост качества по сравнению с лучшей одномодальной моделью? Какая модальность создаёт прирост и сохраняется ли он на другом каталоге и в более позднем периоде?

**Поисковый запрос** (Google Scholar, arXiv, ISMIR, ACL Anthology, 2017–2026):
`("multimodal" OR "fusion") AND ("music" OR "audio" OR "vision-language" OR "time series") AND ("classification" OR "tagging" OR "missing modality")`

### Аннотированный обзор

| № | Источник | Суть | Значение для работы |
|---|---|---|---|
| 1 | Baltrušaitis T., Ahuja C., Morency L.-P. *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI, 2019. [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) | Таксономия мультимодального обучения: представление, выравнивание, слияние (early / late / hybrid), co-learning | Терминология и классификация схем fusion, использованных в работе |
| 2 | Radford A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Общее пространство изображений и текста через контрастивное обучение, zero-shot классификация | Энкодер изображения-спектрограммы и zero-shot baseline |
| 3 | Alayrac J.-B. et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) | Cross-attention между замороженным визуальным энкодером и LLM, few-shot | Пример глубокой (intermediate) интеграции модальностей |
| 4 | Li J. et al. *BLIP-2.* ICML 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) | Q-Former связывает замороженные визуальный и языковой энкодеры | Экономичная адаптация без обучения энкодеров — так же устроены наши схемы на замороженном CLIP |
| 5 | Liu H. et al. *Visual Instruction Tuning (LLaVA).* NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) | Instruction tuning VLM | Контекст для VLM из ЛР1, которые на спектрограммах работали на уровне угадывания |
| 6 | Wang W., Tran D., Feiszli M. *What Makes Training Multi-Modal Classification Networks Hard?* CVPR 2020. [arXiv:1905.12681](https://arxiv.org/abs/1905.12681) | Совместно обучаемая мультимодальная сеть может быть хуже лучшей одномодальной из-за переобучения и разной скорости обучения модальностей | Объясняет провал gated-MLP и доминирование одной модальности |
| 7 | Ma M. et al. *Are Multimodal Transformers Robust to Missing Modality?* CVPR 2022. [arXiv:2204.05454](https://arxiv.org/abs/2204.05454) | Качество мультимодальных моделей резко падает при пропуске модальности; стратегия слияния влияет на устойчивость | Обоснование эксперимента с пропуском модальности и modality dropout |
| 8 | Elizalde B. et al. *CLAP: Learning Audio Concepts From Natural Language Supervision.* ICASSP 2023. [arXiv:2206.04769](https://arxiv.org/abs/2206.04769) | CLIP-подобное пространство «аудио — текст» | Альтернатива спектрограмме-изображению для аудио-модальности |
| 9 | Huang Q. et al. *MuLan: A Joint Embedding of Music Audio and Natural Language.* ISMIR 2022. [arXiv:2208.12415](https://arxiv.org/abs/2208.12415) | Совместное пространство музыки и текста на 44 млн треков | Показывает пользу связи музыки и текста, но требует закрытых данных масштаба индустрии |
| 10 | Oramas S. et al. *Multi-label Music Genre Classification from Audio, Text, and Images Using Deep Features.* ISMIR 2017. [arXiv:1707.04916](https://arxiv.org/abs/1707.04916) | Аудио + тексты рецензий + обложки; мультимодальная модель лучше одномодальных | Ближайший музыкальный аналог тройного слияния модальностей |
| 11 | Delbouys R. et al. *Music Mood Detection Based on Audio and Lyrics with Deep Neural Net.* ISMIR 2018. [arXiv:1809.07276](https://arxiv.org/abs/1809.07276) | Mid-level fusion аудио и текста песни для настроения | Показывает, что выигрыш fusion зависит от задачи и модальности |
| 12 | Jin M. et al. *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.* ICLR 2024. [arXiv:2310.01728](https://arxiv.org/abs/2310.01728) | Временной ряд переводится в пространство LLM с текстовым префиксом | Представитель направления text + time series |
| 13 | Law E. et al. *Evaluation of algorithms using games: the case of music tagging (MagnaTagATune).* ISMIR 2009 | Набор 25,9 тыс. клипов с краудсорсинговыми тегами | Основной набор данных |
| 14 | Manco I. et al. *The Song Describer Dataset.* NeurIPS 2023 Workshop on ML for Audio. [arXiv:2311.10057](https://arxiv.org/abs/2311.10057); Bogdanov D. et al. *The MTG-Jamendo Dataset.* ICML 2019 Workshop ML4MD | Описания треков Jamendo слушателями; метаданные и даты релиза Jamendo | Авторский набор для переноса: другой каталог, релизы 2005–2017 |
| 15 | Dixit S., Heller L., Donahue C. *Vision Language Models Are Few-Shot Audio Spectrogram Classifiers.* 2024. [arXiv:2411.12058](https://arxiv.org/abs/2411.12058) | VLM классифицируют звуки по картинке спектрограммы | Обоснование модальности «изображение-спектрограмма»; в ЛР1 подход без обучения не сработал |

### Сравнительная матрица

| Работа | Модальности | Кодирование | Fusion | Задача | Данные | Метрики | Код |
|---|---|---|---|---|---|---|---|
| CLIP | изображение + текст | ViT / Transformer | контрастивное выравнивание | zero-shot классификация | 400 млн пар | accuracy | да |
| Flamingo | изображение + текст | NFNet + LM | cross-attention (intermediate) | VQA, captioning | M3W, ALIGN | CIDEr, acc | нет |
| BLIP-2 | изображение + текст | ViT + Q-Former + LLM | запросы Q-Former | VQA, captioning | 129 млн | VQA acc | да |
| LLaVA | изображение + текст | CLIP + LLM | проекция в токены LLM | instruction following | 158 тыс. инструкций | GPT-оценка | да |
| Wang et al. | видео + аудио | CNN | late / joint | классификация видео | Kinetics | accuracy | частично |
| Ma et al. | изображение + текст | Transformer | early / mid / late | классификация | MM-IMDb и др. | F1, acc | да |
| CLAP | аудио + текст | CNN14 + BERT | контрастивное выравнивание | zero-shot аудио | 128 тыс. пар | acc, mAP | да |
| MuLan | музыка + текст | ResNet / AST + BERT | контрастивное выравнивание | поиск, тегирование | 44 млн треков | AUC, mAP | нет |
| Oramas et al. | аудио + текст + обложка | CNN + VSM + ResNet | конкатенация эмбеддингов | жанры (multi-label) | MuMu, 31 тыс. альбомов | AUC | да |
| Delbouys et al. | аудио + лирика | CNN + embeddings | mid-level | настроение (регрессия) | 18 тыс. треков Deezer | R² | частично |
| Time-LLM | ряд + текст | патчи + LLM | репрограммирование | прогноз рядов | ETT, Weather | MSE | да |
| Dixit et al. | спектрограмма + текст | VLM | промпт | классификация звуков | ESC-10 | accuracy | да |
| **Эта работа** | **текст + ряд + изображение** | **TF-IDF / статистики + BiGRU / CLIP** | **early, late, stacking, gated + zero-shot** | **тип фронт-вокала** | **MTAT + SDD** | **Macro-F1, MCC, AUROC, Brier, ECE** | **да** |

### Исследовательский разрыв

В музыкальных работах (10, 11) мультимодальность показывает прирост для жанра и настроения, но:
1. тип фронт-вокала отдельно не изучался;
2. одновременно три модальности — метаданные, динамика звука и спектрограмма — не сравнивались;
3. прирост, как правило, не проверяется на другом каталоге и периоде, а устойчивость к пропущенной модальности (7) в музыке не изучена.

Работа 6 предупреждает, что совместно обучаемая сеть может проиграть одномодальной. Это требует обязательных unimodal baseline и статистической проверки прироста.

### Гипотезы

- **H1.** Лучшая схема fusion повысит Macro-F1 на тесте MTAT не менее чем на 0,03 относительно лучшего unimodal baseline, причём 95% интервал прироста не включает 0.
- **H2.** Основной вклад даст аудио-модальность (ряд или изображение); текст метаданных будет вспомогательным.
- **H3.** Прирост сохранится при переносе на другой каталог (Jamendo) и на более поздний период (релизы 2010–2017).
- **H4.** Поздние схемы (late fusion, stacking) устойчивее к пропуску модальности, чем совместно обучаемая сеть.
