# Avaliação do modelo v3

Gerado por `python scripts/train.py` — não editar à mão.

- Gravações: 851 | pacientes: 104 | features: 77 (áudio cortado a 20 s, sem `duration_s`)
- Validação: 5×5-fold estratificado por paciente; IC95% por bootstrap de pacientes (2000 reamostragens).
- Decisão por paciente: média das probabilidades calibradas das suas gravações.

## Todos os pacientes

Pacientes: COPD 64, Healthy 26, URTI 14

| Estratégia | Recall doente | Recall COPD | Recall URTI | Especificidade | F1-macro | FN clínicos |
|---|---|---|---|---|---|---|
| argmax | 0.887 [0.854, 0.921] | 1.000 [1.000, 1.000] | 0.243 [0.086, 0.414] | 0.523 [0.377, 0.677] | 0.591 [0.511, 0.664] | 8.8 [6.2, 11.4] |
| safe θ=0.30 | 0.910 [0.872, 0.949] | 1.000 [1.000, 1.000] | 0.329 [0.143, 0.529] | 0.431 [0.284, 0.585] | 0.589 [0.502, 0.670] | 7.0 [4.0, 10.0] |
| safe θ=0.10 | 0.938 [0.903, 0.969] | 1.000 [1.000, 1.000] | 0.457 [0.257, 0.657] | 0.277 [0.138, 0.423] | 0.563 [0.478, 0.641] | 4.8 [2.4, 7.6] |
| safe θ=0.05 | 0.962 [0.928, 0.990] | 1.000 [1.000, 1.000] | 0.586 [0.386, 0.772] | 0.231 [0.100, 0.369] | 0.572 [0.487, 0.650] | 3.0 [0.8, 5.6] |

## Só gravações Meditron (único equipamento com as 3 classes)

Pacientes: COPD 8, Healthy 26, URTI 14

| Estratégia | Recall doente | Recall COPD | Recall URTI | Especificidade | F1-macro | FN clínicos |
|---|---|---|---|---|---|---|
| argmax | 0.600 [0.482, 0.718] | 1.000 [1.000, 1.000] | 0.243 [0.086, 0.415] | 0.523 [0.369, 0.677] | 0.538 [0.455, 0.632] | 8.8 [6.2, 11.4] |
| safe θ=0.30 | 0.682 [0.555, 0.809] | 1.000 [1.000, 1.000] | 0.329 [0.143, 0.543] | 0.431 [0.284, 0.600] | 0.523 [0.429, 0.624] | 7.0 [4.2, 9.8] |
| safe θ=0.10 | 0.782 [0.664, 0.891] | 1.000 [1.000, 1.000] | 0.457 [0.257, 0.657] | 0.277 [0.146, 0.431] | 0.489 [0.393, 0.588] | 4.8 [2.4, 7.4] |
| safe θ=0.05 | 0.864 [0.745, 0.964] | 1.000 [1.000, 1.000] | 0.586 [0.386, 0.786] | 0.231 [0.108, 0.377] | 0.498 [0.399, 0.594] | 3.0 [0.8, 5.6] |

Recall doente = doentes (COPD ou URTI) não classificados como Healthy. FN clínicos = n.º médio de pacientes doentes dados como Healthy.

## Calibração (aninhada, por gravação)

| Método | Log loss | Brier |
|---|---|---|
| nenhuma | 0.228 | 0.073 |
| isotonica | 0.196 | 0.067 |
| sigmoide ← usado | 0.150 | 0.065 |

## Viés de equipamento — sonda que tenta adivinhar o equipamento

| Sonda | Features cruas | Features residualizadas |
|---|---|---|
| Linear (LogReg) | 0.895 | 0.216 |
| Não-linear (RandomForest) | 0.897 | 0.869 |

A residualização linear só remove a componente linear do viés.

## Detetor de outlier

- Limiar (p99 empírico no treino): 346.0
- Falso alarme medido fora do treino (CV): 5.5%

## Custos — safe θ=0.10 vs argmax, 50.000 pacientes/ano

As prevalências são **hipóteses**, não medições. Valores negativos = o modo safe sai mais caro.

| Cenário (COPD/Healthy/URTI) | €/paciente argmax | €/paciente safe | Poupança anual [IC95%] |
|---|---|---|---|
| dataset ICBHI (não realista) | 15.48 | 17.77 | −€114.423 [−€208.654, −€13.462] |
| rastreio população geral (5/85/10) | 33.68 | 46.82 | −€657.308 [−€941.923, −€390.505] |
| cuidados primários (10/70/20) | 33.97 | 43.03 | −€453.077 [−€695.798, −€218.077] |
| consulta de pneumologia (40/40/20) | 23.95 | 27.85 | −€194.615 [−€343.846, −€38.462] |

## Limitações que estes números não resolvem

- Healthy e URTI só existem em gravações Meditron; fora dele o modelo só viu COPD.
- 14 pacientes URTI e 26 Healthy: os intervalos são largos e devem ser reportados.
- Não há conjunto de teste independente: todos os dados participaram na escolha do modelo.
- Desempenho pediátrico fraco (nb15): não usar em crianças.
