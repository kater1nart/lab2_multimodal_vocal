# Анализ ошибок ЛР2 (MTAT test)

## Матрица ошибок: Только ряд

|              |   male |   female |   group |   instrumental |
|:-------------|-------:|---------:|--------:|---------------:|
| male         |    186 |       36 |      16 |             23 |
| female       |     65 |      142 |      12 |             35 |
| group        |     21 |       15 |      99 |              9 |
| instrumental |     33 |       34 |      15 |            268 |

## Матрица ошибок: Стекинг

|              |   male |   female |   group |   instrumental |
|:-------------|-------:|---------:|--------:|---------------:|
| male         |    191 |       38 |      12 |             20 |
| female       |     26 |      201 |       8 |             19 |
| group        |     16 |       17 |     106 |              5 |
| instrumental |     25 |       48 |       6 |            271 |

## Согласие одномодальных моделей и точность

| согласие модальностей   |   n |   acc текст |   acc ряд |   acc изображение |   acc стекинг |
|:------------------------|----:|------------:|----------:|------------------:|--------------:|
| 2 из 3                  | 503 |       0.457 |     0.571 |             0.525 |         0.712 |
| все 3 согласны          | 380 |       0.934 |     0.934 |             0.934 |         0.934 |
| все разные              | 126 |       0.238 |     0.421 |             0.254 |         0.444 |

Стекинг исправил ошибки модели «только ряд» в 130 случаях и внёс новые ошибки в 56 случаях.

## Примеры: стекинг исправил ошибку ряда (подсказала другая модальность)

|   id | artist       | title          | label        | only_text    | only_ts      | only_img     | stacking     |
|-----:|:-------------|:---------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| 1726 | Psychetropic | Futurengine    | instrumental | instrumental | group        | instrumental | instrumental |
| 2069 | Processor    | Insomnia       | male         | instrumental | instrumental | male         | male         |
| 2536 | Processor    | Mad bull       | male         | instrumental | instrumental | male         | male         |
| 2844 | Norine Braun | Modern Anguish | female       | female       | male         | male         | female       |
| 2845 | Norine Braun | Modern Anguish | female       | female       | male         | female       | female       |
| 3247 | Norine Braun | Persephone     | female       | female       | male         | male         | female       |

## Примеры: ошибки стекинга

|   id | artist                                      | title                                       | label        | only_text    | only_ts      | only_img     | stacking     |
|-----:|:--------------------------------------------|:--------------------------------------------|:-------------|:-------------|:-------------|:-------------|:-------------|
|  178 | Burning Babylon                             | 7 Nine Skank                                | male         | instrumental | instrumental | male         | instrumental |
|  181 | Burning Babylon                             | 7 Nine Skank                                | instrumental | instrumental | male         | female       | female       |
|  696 | Kiev Theological Academy and Seminary Choir | Blessed is the man                          | male         | group        | male         | group        | group        |
|  813 | Wicked boy                                  | Buy my pullovers                            | male         | instrumental | female       | instrumental | female       |
| 1473 | Philharmonia Baroque                        | Eroica 1st                                  | instrumental | group        | instrumental | group        | group        |
| 1718 | Norine Braun                                | Frontliner Blues                            | female       | female       | male         | male         | male         |
| 2848 | Norine Braun                                | Modern Anguish                              | group        | female       | male         | male         | female       |
| 4111 | Solace                                      | Saalik                                      | female       | instrumental | female       | group        | group        |
| 5207 | Minstrel Spirit                             | The lonely minstrel part 1 (edited version) | female       | instrumental | instrumental | instrumental | instrumental |
| 7046 | Indidginus                                  | Dusty lands                                 | instrumental | instrumental | male         | male         | male         |

## Самые частые путаницы стекинга

|                            |   n |
|:---------------------------|----:|
| ('instrumental', 'female') |  48 |
| ('male', 'female')         |  38 |
| ('female', 'male')         |  26 |
| ('instrumental', 'male')   |  25 |
| ('male', 'instrumental')   |  20 |

## Перенос на SDD: точность по классам

| label        |   only_text |   only_ts |   only_img |   late_fusion |   stacking |
|:-------------|------------:|----------:|-----------:|--------------:|-----------:|
| female       |       0.029 |     0.657 |      0.286 |         0.6   |      0.571 |
| group        |       0     |     0.034 |      0     |         0     |      0     |
| instrumental |       0.772 |     0.693 |      0.98  |         0.901 |      0.891 |
| male         |       0.095 |     0.302 |      0.612 |         0.44  |      0.397 |

## Latency извлечения признаков, с на клип (Apple M3 Pro, 1 ядро CPU; CLIP на GPU MPS)

|          |    сек |
|:---------|-------:|
| decode   | 0.0197 |
| ts       | 0.0232 |
| spec_png | 0.0286 |
| clip_mps | 0.0049 |
