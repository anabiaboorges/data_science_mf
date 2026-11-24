# Relatório da frente de machine learning – Random Forest

> Responsável: Ana Beatriz · Notebooks: [`frentes/ml/`](../../frentes/ml/) · Resumo de 5 linhas: [`resumo.md`](resumo.md) · Configuração congelada: [`config_final.json`](config_final.json) · Plano: [`planos/PLANO_FRENTES_02.11.md`](../../planos/PLANO_FRENTES_02.11.md)

Este relatório explica **o que a frente quis responder, o que foi feito, como, por que cada decisão foi tomada e o que os resultados querem dizer**. O `resumo.md` segue o formato fixo de 5 linhas do plano; aqui está o raciocínio completo, com todos os números tirados dos CSVs desta pasta.

---

## Sumário

1. [Em uma frase](#1-em-uma-frase)
2. [Objetivo e perguntas](#2-objetivo-e-perguntas)
3. [O problema: alvo, dados e features](#3-o-problema-alvo-dados-e-features)
4. [O Random Forest no nosso problema](#4-o-random-forest-no-nosso-problema)
5. [O protocolo de avaliação](#5-o-protocolo-de-avaliação)
6. [O pipeline passo a passo (Etapas A–F)](#6-o-pipeline-passo-a-passo-etapas-af)
7. [Ajustes desta frente em relação ao plano comum](#7-ajustes-desta-frente-em-relação-ao-plano-comum)
8. [Resultado oficial no teste](#8-resultado-oficial-no-teste)
9. [Investigações complementares](#9-investigações-complementares)
10. [Por que o modelo prevê, mas não ganha dinheiro](#10-por-que-o-modelo-prevê-mas-não-ganha-dinheiro)
11. [Quadro de decisões](#11-quadro-de-decisões)
12. [Limitações](#12-limitações)
13. [Para a apresentação](#13-para-a-apresentação)
14. [Arquivos e como reproduzir](#14-arquivos-e-como-reproduzir)

---

## 1. Em uma frase

O Random Forest encontra uma **previsibilidade real, mas minúscula**, no retorno do minuto seguinte do WIN: é estatisticamente melhor que prever zero (Clark-West t = 3,72 no teste), mas o ganho médio por operação, **+0,59 ponto, é cerca de 8 vezes menor que o custo de operar (5 pontos)**. Nenhuma das formas testadas de usar essa previsão (limiar alto, candles maiores, segurar a posição, alvo de 10–20 minutos) deu lucro com evidência estatística depois dos custos. A melhor delas, segurar a posição até a previsão virar de sinal, chega perto do empate.

---

## 2. Objetivo e perguntas

O projeto prevê o **T(t+1)**: o retorno do próximo candle do mini-índice (WIN). Cada frente usa uma família de modelos, sob o mesmo protocolo, para responder às mesmas quatro perguntas do plano:

| Pergunta | O que significa | Resposta da frente de ML |
|---|---|---|
| **Q1** – Qual candle e qual janela de treino? | 1, 15 ou 60 min; 1, 3, 6 ou 12 meses de histórico | **1 min, 12 meses** (menor Theil U; nenhuma combinação lucrativa com significância) |
| **Q2** – O modelo bate o naive? | Prevê melhor do que "o retorno será zero"? | **Sim, estatisticamente, mas por muito pouco**: Theil U 0,9999, R²_oos +0,0002, Clark-West t = 3,72 |
| **Q3** – Acertar mais é ganhar mais? | Acurácia direcional se traduz em lucro? | Sim na direção (correlação 0,85 na grade), mas acertar 51% das direções não paga o custo |
| **Q4** – Qual regra de operação e qual o resultado real? | Simulação com limiar, take profit, stop loss e custo | Opera sempre, sem TP/SL; **−4,41 pts por operação** no teste com custo de 5 pts |

**Por que Random Forest nesta frente:** ele captura relações **não lineares** e **interações** entre variáveis (por exemplo, "um candle de corpo grande *e* volume alto no fim do pregão") que o ARIMA, linear, não enxerga. Se houvesse estrutura não linear no retorno de curto prazo, esta frente deveria encontrá-la.

---

## 3. O problema: alvo, dados e features

### Alvo

O alvo é o mesmo de todas as frentes (`tt1_comum.montar_base`): o **log-retorno do próximo candle, em %**.

```
alvo(t) = 100 · ln( P(t+1) / P(t) )
```

O modelo devolve um número contínuo (quanto sobe ou desce), e não só a direção. Por isso o RF usado é o `RandomForestRegressor` (regressão), não o classificador. O último candle de cada pregão fica sem alvo, porque nenhuma operação atravessa a noite.

**Escala do problema:** no candle de 1 min, o retorno típico é de ~0,03% (desvio-padrão), cerca de **40 pontos** com o WIN perto de 140 mil. As previsões do modelo são muito menores, entre 0,4 e 1,5 ponto. Isso já antecipa o problema do custo (Seção 10).

### Dados

| Item | Valor |
|---|---|
| Ativo previsto | WIN (mini-índice), série contínua `WIN$N`, candles de 1 min do MetaTrader 5 |
| Candles | 1 min (original), 15 min e 60 min (reamostrados por pregão) |
| Base comum | WIN + WDO por *inner join*, termina em **01/04/2026** porque o WDO acaba aí |
| Base só WIN | Usada no período novo (abr–set/2026); o WIN vai até 25/09/2026 |
| Buraco nos dados | jun–out/2023 não existe na base; por isso as janelas de 3 e 6 meses têm o mesmo tamanho no 1º trimestre de 2024 |

### Features (conjunto `COMPLETO` do código comum)

Todas são conhecidas no fechamento do candle *t* (nenhuma usa informação de *t+1*).

| Feature | Definição | Intuição |
|---|---|---|
| `ret_lag1` … `ret_lag10` | Retornos dos 10 últimos candles (`ret_lag1` = retorno do próprio candle *t*) | Momento ou reversão de curto prazo |
| `amplitude` | (máxima − mínima) / fechamento, em % | Quanto o preço oscilou no candle |
| `corpo` | (fechamento − abertura) / fechamento, em % | Direção e força do candle |
| `vol_rel` | Volume / média de volume do último pregão | Atividade anormal |
| `volat` | Desvio-padrão dos retornos do último pregão (s candles) | Regime de volatilidade |
| `hora` | Hora do dia (decimal) | Padrões intradiários (abertura, almoço, fechamento) |
| `candle_do_dia` | Posição do candle no pregão | Idem, em contagem |

**O RF não usa o dólar (WDO).** O conjunto `COMPLETO` não inclui `ret_wdo_lag1`. A EDA e a frente estatística mostraram que a relação WIN × WDO é **instantânea** (correlação −0,47 no mesmo candle e ≈ 0,01 com um candle de defasagem, em 60 min): o dólar de *t* não antecipa o índice de *t+1*. Isso também permitiu rodar o modelo no período abr–set/2026, em que só existe o WIN.

---

## 4. O Random Forest no nosso problema

### Como funciona

1. **Árvore de decisão:** divide os dados com perguntas do tipo "`corpo` > 0,02%?" e "`hora` < 10,5?". Cada divisão é escolhida para deixar os grupos resultantes com retornos mais parecidos (menor variância). Na folha, a previsão é a **média dos retornos** dos candles de treino que caíram ali.
2. **Floresta:** treina muitas árvores, cada uma em uma amostra *bootstrap* (sorteio com reposição) das linhas e, em cada divisão, olhando só uma fração das features (`max_features = 0,5`). A previsão final é a **média das árvores**.
3. **Por que funciona:** uma árvore sozinha decora ruído (alta variância). Árvores diferentes erram de formas diferentes; a média cancela parte desse erro.

### Hiperparâmetros e o que cada um controla

| Hiperparâmetro | Valor final | O que controla | Como foi decidido |
|---|---|---|---|
| `n_estimators` (nº de árvores) | **100** | Estabilidade da média; mais árvores não causam overfitting, só custam tempo | Etapa A2: menor nº cujo Theil U fica a < 0,0005 do resultado com 400 árvores |
| `max_depth` (profundidade) | **4** | Quantas perguntas em sequência; mais profundo = mais complexo | Etapa D: testados 4, 8 e 12 |
| `min_samples_leaf` (folha mínima) | **2% das linhas de treino** (mín. 10) | Quantos candles, no mínimo, cada previsão média usa; maior = mais suave | Etapa D: testados 0,5%, 1% e 2% |
| `max_features` | 0,5 | Diversidade entre as árvores | Fixo (valor usual para regressão) |
| `random_state` | 42 | Reprodutibilidade | Fixo |

Com janela de 12 meses em 1 min (~85 mil candles no treino), a folha mínima de 2% significa que **cada previsão é a média de pelo menos ~1.700 candles**. O modelo final é deliberadamente simples (profundidade 4 = no máximo 16 folhas por árvore). Isso faz sentido num alvo com muito ruído: um modelo mais flexível só aprenderia ruído.

### Importância das features

Medida pela redução média de impureza (variância) no modelo final, treinado na última janela da validação (`importancia_features.png`). As três mais usadas foram **`corpo`, `ret_lag1` e `amplitude`**: tudo informação do **próprio candle *t***. Os retornos mais antigos (lags 2–10), a hora e o volume pesaram pouco. Esse padrão é importante para a interpretação da Seção 10.

---

## 5. O protocolo de avaliação

### Períodos

| Período | Datas | Uso |
|---|---|---|
| Treino | Os *N* meses anteriores a cada trimestre (*N* = 1, 3, 6 ou 12) | Ajustar o modelo |
| **Validação** | 01/2024 – 12/2024 | **Todas as escolhas**: candle, janela, nº de árvores, hiperparâmetros, regra |
| **Teste oficial** | 01/2025 – 03/2026 | Rodado **uma única vez** com tudo congelado |
| **Período novo** | 04/2026 – 09/2026 | Nunca usado por nenhum modelo do projeto; usado só para hipóteses registradas antes de rodar |

### Walk-forward com re-treino trimestral

O modelo não é treinado uma vez só. No início de cada trimestre (jan, abr, jul, out), ele é **re-treinado com os últimos N meses** (janela deslizante) e prevê os 3 meses seguintes. Exemplo com janela de 12 meses no teste: o modelo treinado em jan–dez/2024 prevê jan–mar/2025; o treinado em abr/2024–mar/2025 prevê abr–jun/2025; e assim por diante. Um **embargo** de 1 candle entre treino e previsão garante que o alvo da última linha de treino não use o 1º candle previsto.

Isso simula o uso real (o modelo só conhece o passado) e permite ver se as relações aprendidas se mantêm ao longo do tempo.

### Métricas da previsão

| Métrica | Definição | Como ler |
|---|---|---|
| **Theil U** | RMSE do modelo ÷ RMSE do naive (previsão = 0) | < 1: melhor que o naive; = 1: igual; > 1: pior |
| **R²_oos** | 1 − (erro² do modelo ÷ erro² do naive), fora da amostra | > 0: melhor que o naive |
| **Acurácia direcional** | % de candles em que o sinal previsto = sinal real | 50% = moeda |
| **Clark-West** (teste t) | Testa se o modelo é melhor que o naive, corrigindo o viés de comparar um modelo com parâmetros com um sem (Clark & West, 2007); erro-padrão HAC com 10 defasagens | t > 1,645: melhor que o naive a 5% (unicaudal) |

**Por que o Clark-West foi incluído:** com 175 mil previsões, um Theil U de 0,9999 pode ser ruído ou sinal; só o número não diz. O Clark-West responde isso.

### Simulação das operações

Usa a função comum `simular` (detalhes em [`resultados/README.md`](../README.md)):

- **Entrada:** compra se previsão > +limiar; vende se previsão < −limiar; entra no fechamento de *t*.
- **Saída:** fechamento de *t+1*, ou antes se tocar take profit/stop loss (se os dois forem tocados no mesmo candle, conta o stop).
- **Custo:** 5 pts por operação completa (sensibilidade: 0 e 10). O ponto do WIN vale R$ 0,20 por contrato.
- **Limiar:** quantil de |previsão| na validação (0%, 50%, 75%, 90%), congelado em % para o teste.
- **Referência aleatória:** mesmos candles e mesma regra, mas com a direção sorteada. Serve para separar o efeito do modelo do efeito da regra e do mercado.

Para cada simulação, além da **esperança por operação** (lucro médio, em pontos), a frente calcula o **erro-padrão** e a **estatística t** (esperança ÷ erro-padrão). A esperança sozinha não basta: +3 pts em 100 operações com desvio de 200 pts é indistinguível de zero.

---

## 6. O pipeline passo a passo (Etapas A–F)

Tudo em [`frentes/ml/frente_ml.ipynb`](../../frentes/ml/frente_ml.ipynb), sem alterar o `tt1_comum.py`.

### Etapa A – Preparação

Monta as bases de 1, 15 e 60 min com `montar_base` e define o RF com a **folha proporcional** (classe `FolhaProporcional`, Seção 7).

### Etapa A2 – Quantas árvores? (`arvores_val.csv`)

Teste em 15 min, janela de 6 meses, validação 2024:

| Árvores | Theil U | Tempo |
|---|---|---|
| 25 | 1,0060 | 0,3 s |
| 50 | 1,0044 | 0,4 s |
| **100** | **1,0041** | 0,8 s |
| 200 | 1,0039 | 1,4 s |
| 400 | 1,0038 | 2,5 s |

**Critério:** o menor número de árvores cujo Theil U fica a menos de 0,0005 do resultado com 400. Com 100, a diferença é 0,0003; com 50, 0,0006. **Escolha: 100 árvores.** Acima disso, o erro não melhora e o tempo dobra a cada passo, o que pesa no candle de 1 min. O plano sugeria 200; a verificação mostrou que 100 bastam.

### Etapa B – Grade candle × janela na validação (`grade_val.csv`)

RF base (100 árvores, profundidade 8, folha 1%) nas 12 combinações, operando em todo candle, custo 5 pts:

| Candle | Janela | Theil U | Acurácia | Esperança (pts/op) | t |
|---|---|---|---|---|---|
| 1 min | 1 m | 1,0015 | 50,3% | −4,88 | −43,5 |
| 1 min | 3 m | 1,0006 | 50,6% | −4,81 | −42,9 |
| 1 min | 6 m | 1,0004 | 50,8% | −4,67 | −41,6 |
| **1 min** | **12 m** | **1,0000** | 50,8% | −4,79 | −42,7 |
| 15 min | 1 m | 1,0133 | 50,7% | −4,74 | −2,9 |
| 15 min | 3 m | 1,0063 | 50,3% | −4,63 | −2,8 |
| 15 min | 6 m | 1,0041 | 50,5% | −4,40 | −2,7 |
| 15 min | 12 m | 1,0025 | 49,8% | −5,46 | −3,3 |
| 60 min | 1 m | 1,0237 | 51,3% | +1,03 | +0,17 |
| 60 min | 3 m | 1,0180 | 48,7% | −10,97 | −1,8 |
| 60 min | 6 m | 1,0164 | 49,4% | −14,33 | −2,3 |
| 60 min | 12 m | 1,0090 | 50,9% | +1,06 | +0,17 |

**Leitura:**
- **Mais histórico ajuda sempre:** em todos os candles, o Theil U cai de 1 para 12 meses. O sinal (se existe) é fraco e precisa de muitos dados para ser estimado.
- **Candle menor prevê melhor:** 1 min fica colado em 1; 60 min é claramente pior que o naive.
- **As duas esperanças positivas (60 min) são ruído:** +1 pt com erro-padrão de 6,2 pts dá t = 0,17. Pelo critério do plano (maior esperança), 60 min × 12 m seria escolhida; pelo critério com t ≥ 2, nenhuma passa.

**Escolha: 1 min × 12 meses**, pelo menor Theil U (nenhuma combinação com esperança positiva e t ≥ 2).

### Etapa C – Regra de operação na validação (`regras_val.csv`)

As 36 regras (4 limiares × 3 TP × 3 SL) em 1 min × 12 m, custo 5 pts. **Todas negativas.** As melhores:

| Limiar | TP | SL | Operações | Acerto | Esperança | t |
|---|---|---|---|---|---|---|
| 90% | 1× | – | 14.122 | 53,9% | −3,50 | −8,1 |
| 90% | – | – | 14.122 | 45,3% | −3,88 | −7,3 |
| 90% | 2× | – | 14.122 | 46,2% | −3,98 | −8,0 |
| 75% | 1× | – | 35.304 | 50,3% | −4,08 | −16,8 |
| … | | | | | | |
| 0% (sempre) | – | – | 141.216 | 40,6% | −4,79 | −42,7 |

**Escolha: regra padrão (opera sempre, sem TP/SL)**, porque nenhuma regra teve t ≥ 2. A regra padrão não é escolhida olhando o resultado, então não carrega viés de seleção.

Dois padrões aparecem:
- **Limiar maior melhora a esperança** (−4,8 → −3,5): as previsões mais fortes são um pouco mais confiáveis. Mas nenhum limiar paga o custo.
- **Take profit curto aumenta a taxa de acerto e reduz o payoff** (53,9% de acerto, payoff 0,67): corta os ganhos e deixa as perdas correrem. Acertar mais não significou ganhar mais.

### Etapa D – Variantes de hiperparâmetros (`variante_val.csv`)

Em 1 min × 12 m, validação 2024:

| Variante | Profundidade | Folha | Theil U | R²_oos | Tempo |
|---|---|---|---|---|---|
| **`rf_d4_f2`** | **4** | **2%** | **0,99995** | **+0,0001** | 14,8 s |
| `rf_d4_f0.5` | 4 | 0,5% | 1,00001 | −0,0000 | 14,7 s |
| `rf_d8_f2` | 8 | 2% | 1,00001 | −0,0000 | 18,1 s |
| `rf_base_d8_f1` | 8 | 1% | 1,00003 | −0,0001 | 26,4 s |
| `rf_d12_f2` | 12 | 2% | 1,00004 | −0,0001 | 27,0 s |
| `rf_d8_f0.5` | 8 | 0,5% | 1,00008 | −0,0002 | 24,0 s |
| `rf_d12_f0.5` | 12 | 0,5% | 1,00010 | −0,0002 | 37,8 s |
| `arvore_d8_f1` (árvore única) | 8 | 1% | 1,00117 | −0,0024 | 6,3 s |

**Leitura:**
- **Quanto mais simples, melhor:** o ranking é quase perfeitamente "menos profundo e folha maior vence". `rf_d4_f2` é a **única variante com Theil U < 1 na validação**.
- **A floresta vale a pena em relação à árvore única:** a árvore sozinha é 4× mais rápida, mas claramente pior (1,0012). A média de 100 árvores é o que tira o modelo de "pior que o naive".
- Nenhuma variante teve esperança positiva com t ≥ 2 (todas em torno de −4,7 pts/op).

**Escolha: `rf_d4_f2`**, pelo menor Theil U. É também a mais rápida entre as florestas.

### Etapa E – Congelamento e teste

O `config_final.json` foi gravado **antes** de gerar qualquer previsão do teste. O teste (01/2025–03/2026) rodou uma vez. Resultados na Seção 8.

> Nota sobre o `config_final.json`: o campo `folha_frac: 0.01` é o valor padrão da grade (Etapas B e C). O modelo final usa 2%, como indica o nome `rf_d4_f2`.

### Etapa F – Gráficos e resumo

`curva_capital_teste.png`, `previsto_vs_real_teste.png`, `importancia_features.png`, heatmaps e o `resumo.md`.

---

## 7. Ajustes desta frente em relação ao plano comum

Os três ajustes abaixo foram aplicados **só na frente de ML**, para o grupo avaliar se adota no projeto todo.

### 7.1 O problema encontrado na primeira rodada

Na primeira execução, seguindo o plano à risca (folha mínima fixa de 100 e escolha pela maior esperança), a combinação vencedora foi **60 min × 1 mês**: +17,8 pts/op na validação. Investigando:

- Com janela de 1 mês em 60 min, o treino tem **~170–200 candles**. Com folha mínima de 100, cada árvore conseguia fazer **uma única divisão**, e o modelo inteiro gerava **só 4 valores diferentes de previsão**.
- O erro-padrão daquela esperança era 12,3 pts: **+17,8 ± 12,3 não é distinguível de zero**.
- No teste, a mesma combinação deu **−31,3 pts/op**.

Ou seja, a escolha foi feita sobre ruído: a **maldição do vencedor**. Ao testar 12 combinações × 36 regras, alguma vai parecer boa por sorte, principalmente onde há poucas operações (60 min) e, portanto, mais variância. A frente estatística viveu o mesmo efeito: a ARIMAX venceu na validação com +10,2 ± 18,5 pts/op e deu −41,1 no teste.

### 7.2 Os ajustes

| Ajuste | O que mudou | Por quê |
|---|---|---|
| **1. Folha proporcional** | `min_samples_leaf = max(10, 1% das linhas de treino)` em vez de 100 fixo | A regularização passa a ter o mesmo peso em janelas de 170 ou de 85 mil linhas. Com 100 fixo, a janela pequena virava um modelo degenerado e a grande ficava sem regularização |
| **2. Escolha com significância** | Escolhe pela maior esperança **só entre as opções com esperança > 0 e t ≥ 2**; se nenhuma passar, usa o menor Theil U (grade) e a regra padrão (opera sempre, sem TP/SL) | Impede que ruído vire decisão. A regra padrão não depende do resultado, então não carrega viés de seleção |
| **3. Clark-West** | Teste de significância da previsão contra o naive | Diz se o Theil U ligeiramente abaixo de 1 é sinal ou acaso |

**Consequência prática:** com esses ajustes, a frente de ML não escolheu nada "porque deu lucro na validação", porque nada deu lucro com significância. A escolha caiu no critério estatístico (menor erro de previsão), que se mostrou estável entre validação e teste (Seção 9.1).

---

## 8. Resultado oficial no teste

Configuração congelada: `rf_d4_f2`, 100 árvores, 1 min, janela de 12 meses, re-treino trimestral, opera em todo minuto, sem TP/SL. Teste de 01/2025 a 03/2026 (`teste_final.csv`).

### Qualidade da previsão

| Métrica | Valor | Leitura |
|---|---|---|
| Previsões | 174.579 candles | |
| Theil U | **0,99992** | Erro 0,008% menor que o do naive |
| R²_oos | **+0,00017** | Positivo, mas minúsculo |
| Acurácia direcional | **51,4%** | 1,4 ponto percentual acima de uma moeda |
| Correlação previsto × real | 0,013 | |
| **Clark-West t** | **3,72** | **Melhor que o naive com significância** (p < 0,001) |

### Simulação

| Custo | Operações | Acerto | Payoff | Esperança (pts/op) | t | Fator de lucro | Resultado (1 contrato) |
|---|---|---|---|---|---|---|---|
| 0 pts | 174.579 | 48,4% | 0,97 | **+0,59** | +3,85 | 1,03 | **+R$ 20.587** |
| **5 pts** | 174.579 | 42,6% | 0,98 | **−4,41** | −28,8 | 0,81 | **−R$ 153.992** |
| 10 pts | 174.579 | 37,2% | 0,98 | −9,41 | −61,5 | 0,63 | −R$ 328.571 |

| Aleatório (mesmos candles, direção sorteada) | Esperança | t |
|---|---|---|
| Custo 0 | +0,17 | +1,10 |
| Custo 5 | −4,83 | −31,6 |

**Leitura:**
- **Sem custo, o modelo ganha de verdade:** +0,59 pt por operação com t = 3,85, contra +0,17 (não significativo) do aleatório. É a mesma conclusão do Clark-West, agora em pontos.
- **Com custo, perde de verdade:** o ganho de +0,59 pt não cobre os 5 pts. Operar todo minuto por 15 meses daria cerca de −R$ 154 mil por contrato.
- O modelo perde um pouco **menos** que o aleatório (−4,41 contra −4,83): a diferença, ~0,4 pt, é o valor da previsão.

---

## 9. Investigações complementares

Depois do teste oficial, a frente testou caminhos para transformar a previsibilidade em lucro. Cada um foi **registrado em JSON antes de rodar** (`hipotese_*.json`), com regra, critério de sucesso (esperança > 0 com t ≥ 2 e fator de lucro > 1, custo 5 pts) e o compromisso de reportar tudo. Nenhum altera o resultado oficial.

### 9.1 A grade completa no teste (`grade_teste_complementar.csv`)

**Pergunta:** a escolha feita na validação seria a melhor no teste? As 12 combinações foram rodadas no teste **só para comparação**.

| Candle | Theil U no teste (1 / 3 / 6 / 12 meses) | Clark-West t (1 / 3 / 6 / 12 meses) |
|---|---|---|
| 1 min | 1,0017 / 1,0002 / 1,0000 / **0,9999** | 2,66 / 3,62 / 4,07 / **4,26** |
| 15 min | 1,0124 / 1,0051 / 1,0025 / 1,0020 | −0,11 / 1,31 / 0,71 / −0,11 |
| 60 min | 1,0244 / 1,0150 / 1,0106 / 1,0083 | −0,80 / −0,75 / 0,03 / −0,45 |

**Leitura:**
- **O ranking por Theil U se repete** entre validação e teste (correlação de Spearman 0,99): a combinação escolhida (1 min × 12 m) foi também a melhor no teste. **O ranking por lucro não se repete** (Spearman −0,24 operando sempre; 0,43 com limiar de 90%, em 12 pontos, sem significância). Isso justifica o ajuste 2: escolher pelo erro de previsão é estável; escolher pelo lucro da validação é sortear.
- **Só o candle de 1 min tem sinal:** Clark-West significativo em todas as janelas de 1 min e em nenhuma de 15 ou 60 min.

### 9.2 Limiar alto no candle de 1 min (`segurar_posicao.csv`, regra `por_minuto`)

**Pergunta:** operando só quando a previsão é forte, o ganho por operação aumenta?

| Limiar | Operações no teste | Ganho bruto (pts/op) | Com custo de 5 pts |
|---|---|---|---|
| 0% (oficial) | 174.579 | +0,59 | −4,41 |
| 50% | 100.057 | +0,94 | −4,06 |
| 75% | 63.878 | +1,07 | −3,93 |
| 90% | 28.409 | +1,99 | −3,01 |

**Leitura:** sim, o ganho bruto cresce de forma monótona com o limiar (as previsões mais fortes são mais confiáveis), mas mesmo no limiar de 90% ele fica em ~2 pts, abaixo dos 5 de custo. O mesmo padrão já aparecia na validação (Etapa C), por isso não muda a escolha.

### 9.3 Candles maiores com limiar alto, no período novo (`hipotese_abr_set_2026.json`)

**Pergunta:** em candles de 15 e 60 min o movimento é maior (191 e 361 pts por candle, em média). Com limiar de 90%, o custo de 5 pts pesaria menos?

| Hipótese | Operações | Esperança (custo 5) | t | Aleatório (mesmos candles) | Clark-West t |
|---|---|---|---|---|---|
| H15: 15 min, 12 m, limiar 90% (≈ 25 pts) | 460 | **−22,2** | −1,13 | +8,8 | −1,75 |
| H60: 60 min, 12 m, limiar 90% (≈ 95 pts) | 120 | **−109,5** | −1,49 | +16,6 | −1,59 |

**Leitura:** o custo pesa menos, mas o problema é outro: **nos candles maiores, o modelo não tem sinal** (Clark-West negativo, perde até para o aleatório). As operações ficam muito voláteis (erro-padrão de 20 e 73 pts), e a esperança fica dominada por ruído. **Hipótese rejeitada.**

### 9.4 Segurar a posição até a previsão virar de sinal (`hipotese_segurar_posicao.json`)

**Pergunta:** o custo é pago a cada operação. Se a posição for mantida enquanto a previsão continua do mesmo lado, o número de operações cai e cada uma acumula vários minutos de ganho. Isso paga o custo?

**Regra:** compra quando a previsão passa +limiar; vende quando passa −limiar; mantém a posição até a previsão passar o limiar **do lado oposto** (vira a mão); zera no fim do pregão; custo de 5 pts por operação completa.

**Resultado com custo de 5 pts (esperança em pts/op e t):**

| Limiar | Validação 2024 | Teste 01/2025–03/2026 | Período novo 04–09/2026 | Duração média |
|---|---|---|---|---|
| 0% | −4,22 (t −13,9) | −3,77 (t −11,8) | −3,13 (t −3,7) | 2–3 min |
| 50% | −3,52 (t −4,8) | −2,87 (t −5,2) | −1,24 (t −0,8) | 4–7 min |
| 75% | −2,54 (t −2,2) | −2,99 (t −3,4) | +0,38 (t +0,2) | 6–10 min |
| 90% | −2,50 (t −1,0) | −0,73 (t −0,4) | **+1,93 (t +0,6)** | 10–22 min |

| Comparação no limiar de 90%, custo 5 | Teste | Período novo |
|---|---|---|
| Segurar (modelo) | −0,73 | +1,93 |
| Segurar com direção aleatória (mesmas operações e durações) | −7,28 | +0,27 |
| Por minuto (regra oficial com o mesmo limiar) | −3,01 | −1,96 |

**Leitura:**
- **É a melhor forma encontrada de usar o modelo.** Em todos os períodos e limiares, segurar perde menos que operar por minuto, e **o modelo bate claramente o aleatório com as mesmas operações** (no teste, −0,73 contra −7,28). Ou seja, a direção da previsão continua valendo por vários minutos.
- **Mas não há evidência de lucro.** Pelo critério registrado, o limiar seria escolhido na validação entre os que têm esperança > 0 e t ≥ 2: **nenhum passou** (todos negativos em 2024). A variante é, portanto, classificada como **"sem evidência"**, e os testes valem só como sensibilidade. O +1,93 do período novo tem t = 0,6: é compatível com zero.
- **Ressalva de método:** a regra foi pensada **depois** de ver o resultado oficial do teste 2025–26 (as previsões já existiam). Por isso, o resultado mais limpo é o do período novo (abr–set/2026), que ninguém tinha visto. Na comparação entre frentes, ela deve aparecer como **variante identificada**, ao lado do resultado oficial, com o intervalo dos 4 limiares, e não só com o de 90%.

### 9.5 Alvo de 10 e 20 minutos (`hipotese_horizonte.json`)

**Pergunta:** na variante anterior, o modelo era treinado para prever 1 minuto, mas a posição ficava aberta 10–22 minutos. E se o alvo for o retorno dos próximos H minutos (H = 10 ou 20), com a posição mantida por H minutos?

Mesmo modelo (`rf_d4_f2`, 1 min, 12 meses), alvo = retorno de *t* a *t+H*, embargo de H candles, Clark-West com H+5 defasagens.

| H | Limiar | Validação | Teste | Período novo |
|---|---|---|---|---|
| 10 | 0% | −4,21 | −4,23 | −6,90 |
| 10 | 90% | −10,39 | −6,27 | −7,71 |
| 20 | 0% | −7,76 | −3,54 | −13,27 |
| 20 | 90% | −8,23 | −6,24 | −7,46 |

(esperança em pts/op com custo de 5; todas as 24 combinações H × limiar × período foram negativas)

**Leitura:** **a previsibilidade some em horizontes maiores.** O ganho bruto fica em torno de zero, de −8,6 a +1,6 pts, sem padrão entre limiares e períodos, e muitas vezes perde para o aleatório. A sensibilidade "segurar até virar o sinal" com as previsões de H minutos também ficou negativa ou insignificante. Conclusão: **o sinal existe só no minuto seguinte**; o que a variante 9.4 aproveitava era a persistência da previsão de 1 minuto, não um sinal de prazo mais longo.

---

## 10. Por que o modelo prevê, mas não ganha dinheiro

### A aritmética do custo

| Quantidade | Valor (1 min) |
|---|---|
| Movimento típico de um candle | ~40 pts (desvio-padrão) |
| Tamanho das previsões do modelo | 0,4 a 1,5 pt |
| Ganho médio do modelo por operação | **+0,59 pt** |
| Custo por operação | **5 pts** |

Com o ganho médio (42,7 pts) e a perda média (43,8 pts) das operações no teste, pagar 5 pts por operação exigiria que cerca de **56%** das operações fossem vencedoras. Antes do custo, são **48,4%** (as demais perdem ou empatam), e a acurácia direcional é de 51,4%. A previsão é real, mas está em uma escala muito menor que a do custo.

### Acurácia × lucro (Q3)

Na grade de validação, a correlação entre acurácia direcional e esperança foi de 0,85: **acertar mais a direção ajuda**. Mas há dois cuidados:
- A **taxa de acerto das operações** cai com o custo (48,4% sem custo → 42,6% com custo de 5 pts), porque operações com ganho de 1 a 4 pts viram perdas.
- O **take profit curto** aumenta a taxa de acerto (até 53,9%) e piora o resultado, porque corta os ganhos (payoff 0,67).

### Hipótese para a origem do sinal: *bid-ask bounce*

As features mais importantes são **`corpo` e `ret_lag1`** (o movimento do próprio candle *t*), e o sinal desaparece em 10–20 minutos. Esse é o padrão de um efeito de microestrutura conhecido: o *bid-ask bounce*. Se o último negócio do candle saiu no preço de venda (ask), o fechamento fica "alto" e o próximo tende a "voltar" ao preço de compra (bid), e vice-versa. Isso gera uma pequena **reversão** previsível de 1 candle, mas é exatamente o spread: para capturá-la, seria preciso comprar no bid e vender no ask, o que é o custo que estamos pagando.

**É uma hipótese, não uma conclusão testada.** Para confirmá-la seria preciso dados de book/negócios (bid e ask), que não temos. Mas ela explica de forma coerente os três fatos: sinal significativo só em 1 min, concentrado no último candle, e menor que o custo.

---

## 11. Quadro de decisões

| # | Decisão | Alternativas consideradas | Motivo |
|---|---|---|---|
| 1 | RF de **regressão** | Classificador (sobe/desce) | Mesmo alvo e métricas das outras frentes; o tamanho previsto alimenta o limiar |
| 2 | Features `COMPLETO` **sem WDO** | Incluir `ret_wdo_lag1` | Relação WIN × WDO é instantânea; o dólar de *t* não antecipa o índice de *t+1*; permite usar o período novo |
| 3 | **100 árvores** | 200 (sugerido no plano), 400 | Theil U estabiliza em 100 (diferença 0,0003 para 400); tempo 2× menor que 200 |
| 4 | **Folha proporcional** (1–2% do treino) | Folha fixa de 100 | Com 100 fixo, janela pequena virava modelo degenerado de 4 valores (Seção 7.1) |
| 5 | **Critério t ≥ 2** | Maior esperança (plano) | A maior esperança da validação era ruído e virou prejuízo no teste |
| 6 | **1 min × 12 meses** | 60 min × 12 m (maior esperança, t = 0,17) | Menor Theil U; nenhuma combinação com t ≥ 2; o Theil U é estável entre validação e teste |
| 7 | **Opera sempre, sem TP/SL** | 90% + TP 1× (melhor esperança, −3,50) | Nenhuma regra com t ≥ 2; a regra padrão não é escolhida pelo resultado |
| 8 | **`rf_d4_f2`** | 6 combinações de profundidade × folha + árvore única | Único com Theil U < 1 na validação; mais rápido |
| 9 | **Clark-West** como teste do Q2 | Só Theil U | Com 175 mil previsões, só o teste diz se 0,9999 é sinal |
| 10 | Hipóteses complementares **registradas antes de rodar** | Testar livremente | Evita ajustar a regra olhando o teste |
| 11 | Segurar a posição como **variante identificada**, não oficial | Substituir o resultado oficial | Foi desenhada depois do teste oficial e não passou o critério na validação |

---

## 12. Limitações

- **Custo estimado, não confirmado:** 5 pts por operação (≈ 1 tick de spread, corretagem zero) não foi validado com corretora/B3. A conclusão é robusta a isso: mesmo com custo zero o ganho é de 0,59 pt; qualquer custo acima de ~0,6 pt elimina o lucro.
- **Execução idealizada:** a simulação entra e sai no fechamento do candle, sem *slippage* nem fila. Na prática, o resultado seria pior.
- **Candles, não ticks:** quando TP e SL são tocados no mesmo candle, conta o stop (hipótese conservadora).
- **Um só ativo e uma só família de modelos:** só WIN e só RF. Outras features (book, fluxo de agressão) poderiam ter mais sinal.
- **Buraco jun–out/2023:** reduz o treino do 1º trimestre de 2024 e torna as janelas de 3 e 6 meses iguais nesse trimestre.
- **Variantes testadas após o teste oficial:** a frente testou 5 caminhos além do protocolo. Mesmo com registro prévio, mais tentativas aumentam a chance de um resultado bom por acaso. Por isso, todos são reportados, inclusive os negativos.
- **Arquivos de previsão de 1 min não versionados** (~280 MB): são recriados rodando os notebooks.

---

## 13. Para a apresentação

**Mensagens principais:**
1. O RF **prevê melhor que o naive com significância estatística** (Clark-West t = 3,72), o que nenhum ARIMA conseguiu. Mas só no candle de 1 min.
2. O ganho é de **0,59 pt por operação contra 5 pts de custo**. Previsão melhor que o naive não basta: o custo decide.
3. **Escolher pelo lucro da validação é sortear** (Spearman −0,24 entre validação e teste); escolher pelo erro de previsão é estável (0,99). Daí o critério t ≥ 2.
4. A melhor forma de uso (segurar a posição) **chega perto do empate**, sem evidência de lucro.
5. Hipótese: o sinal é em boa parte **bid-ask bounce**, que não é capturável pagando o spread.

**Figuras sugeridas:** `heatmap_theilU.png` (Q1), `validacao_vs_teste.png` (estabilidade do Theil U × lucro), `curva_capital_teste.png` (Q4, efeito do custo), `importancia_features.png` (origem do sinal), `segurar_posicao_curvas.png` (melhor variante).

---

## 14. Arquivos e como reproduzir

Rodar os notebooks de `frentes/ml/` nesta ordem (cada um usa saídas do anterior):

| Notebook | Saídas nesta pasta |
|---|---|
| `frente_ml.ipynb` | `arvores_val.*`, `grade_val.csv`, heatmaps, `regras_val.csv`, `variante_val.csv`, `importancia_features.png`, `config_final.json`, `teste_final.csv`, `curva_capital_teste.png`, `previsto_vs_real_teste.png`, `resumo.md` |
| `teste_complementar_grade.ipynb` | `grade_teste_complementar.csv`, `validacao_vs_teste.png` |
| `teste_abr_set_2026.ipynb` | `hipotese_abr_set_2026.json`, `resultado_abr_set_2026.csv`, `curva_capital_abr_set_2026.png` |
| `variante_segurar_posicao.ipynb` | `hipotese_segurar_posicao.json`, `segurar_posicao.csv`, `segurar_posicao_curvas.png` |
| `variante_horizonte.ipynb` | `hipotese_horizonte.json`, `horizonte_resultados.csv`, `horizonte_segurar_sensibilidade.csv`, `horizonte_curvas.png` |

No Windows, se o `jupyter execute` der erro de codificação, defina `PYTHONUTF8=1` antes de rodar.
