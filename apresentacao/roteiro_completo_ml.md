# Roteiro completo — Frente de Machine Learning (Random Forest)

Slides 13 a 26 de `slides.html` · 16 a 18 minutos · atualizado em 23/11/2026

Este arquivo substitui, para a fala, os roteiros antigos, que ainda tratam as regras de compra e venda como próximo passo.

**Como ler cada slide:**
- **Está no slide:** o que o público vê.
- **Fala:** texto pronto.
- **Como foi testado:** o método.
- **Outros cenários bons:** o que quase foi escolhido.
- **Se perguntarem / Atenção:** apoio e pontos sensíveis.

| Slides | Tema | Tempo |
|---|---|---|
| 13 | Roteiro dos 6 passos | 1 min |
| 14 | Candle | 1 min 30 s |
| 15 | Features | 2 min 30 s |
| 16–17 | Janela e retreino | 2 min |
| 18–20 | Hiperparâmetros | 3 min |
| 21 | Ganho acumulado | 1 min |
| 22–25 | Etapa 5 | 4 min |
| 26 | Outras abordagens | 1 min 30 s |

---

## Antes de começar: 6 ideias para ter na ponta da língua

1. **Alvo:** retorno do próximo candle, `100 · ln(C(t+1)/C(t))`, só dentro do mesmo pregão.
2. **Custo:** 5 pontos por operação completa, entrada e saída (R$ 1,00 por contrato). Vale para tudo.
3. **Walk-forward:** o modelo treina só com o passado, prevê o bloco seguinte e avança. Nunca vê o que prevê.
4. **Métrica dos passos 1 a 4:** correlação de Spearman entre previsão e retorno, calculada mês a mês em 2024. Mede se a **ordem** das previsões está certa.
5. **t robusto:** média das 12 correlações mensais ÷ erro-padrão entre os meses. Acima de 2, o sinal se distingue do ruído. Não usamos 1/√n porque a volatilidade vem em ondas, e isso inflaria o t.
6. **Desempate:** entre opções a até 1 erro-padrão da melhor, fica a mais simples.

**Os períodos:**

| Período | Uso |
|---|---|
| até 2023 | análise de dados; em 2022–2023, também a grade de regras da etapa 5 |
| 2024 | escolhas dos passos 1 a 4 e seleção dos finalistas da etapa 5 |
| 01/2025–03/2026 | rodada com a regra da etapa 5 congelada |

---

## Slide 13 · Seis passos, um grupo de decisões por vez

**Está no slide:** os seis passos, todos concluídos: candles (1 min), features (21), janela e retreino (12 meses, semestral), hiperparâmetros (50 árvores, profundidade 2, folha 10%, raiz das features), regras (nenhuma aprovada) e validação (sinal mantido, sem lucro).

**Fala:**
> "Usamos um Random Forest, que é a média de muitas árvores de decisão. Em vez de testar todas as combinações de uma vez, o que facilitaria achar algo por acaso, dividimos as decisões em seis passos. Cada um decide um grupo de parâmetros e congela a escolha antes do próximo. Os passos 1 a 4 escolhem o modelo olhando a qualidade da previsão em 2024. O passo 5 transforma a previsão em operação, e o 6 confere fora do período de escolha."

**Por que essa ordem:**
- o candle muda o próprio alvo, por isso vem primeiro;
- as features são a decisão que mais pesa;
- a janela define o tamanho do treino, e a folha é um percentual dele;
- as regras vêm por último e nunca voltam a mudar o modelo.

**Se perguntarem por que Random Forest:** capta relações não lineares e condicionais, aceita features em escalas diferentes e permite controlar o quanto decora (profundidade e folha).

---

## Slide 14 · Candle: escolhido o de 1 minuto

**Está no slide:** correlação média mensal em 2024, com ±2 erros-padrão.

| Candle | Correlação | t | Meses positivos |
|---|---|---|---|
| **1 min** | **0,018** | **4,9** | **12 de 12** |
| 5 min | 0,002 | 0,3 | 7 |
| 15 min | −0,003 | −0,3 | 5 |
| 30 min | −0,004 | −0,2 | 5 |
| 60 min | −0,011 | −0,5 | 4 |

Quadro do modelo: ganho bruto +0,37 ± 0,11 · após custo −4,63 · acerto 50,9%.

**Fala:**
> "Há um dilema. Em candles grandes o preço anda mais, e o custo pesa menos: 1,6% do movimento no de 60 minutos, contra 12% no de 1 minuto. Mas só o de 1 minuto mostrou sinal: correlação de 0,018, t de 4,9, positiva nos 12 meses. Nos outros o t ficou entre menos 0,5 e 0,3. Custo baixo sem sinal não gera lucro."

**Como foi testado:**
- O mesmo RF nos cinco candles: 200 árvores, profundidade 5, folha de 1%, 5 features básicas, janela de 12 meses, retreino trimestral.
- Antes do modelo, até 2023: a correlação de um candle com o seguinte (ρ₁) ficou dentro do ruído em todos os tamanhos. Não há memória linear.
- Filtro: Theil U ≤ 1,001.

**Outros cenários bons:** o ranking muda com a janela.

| Candle | 6 meses (t) | 12 meses (t) | 24 meses (t) |
|---|---|---|---|
| 1 min | 3,6 | 4,9 | 1,3 |
| 5 min | 2,0 | 0,3 | 1,6 |

O **5 min com 6 meses** foi o vice mais consistente (t = 1,99) e ficou anotado como alternativa.

**Correlação necessária para pagar 5 pts:** 0,107 (1 min), 0,049 (5), 0,029 (15), 0,020 (30) e 0,014 (60). É uma aproximação: ρ = custo ÷ (desvio em pontos × 0,8).

**Atenção:** outra abordagem da frente (slide 26) escolheu 60 min, com 36 meses de treino. Aqui só testamos 60 min com 6, 12 e 24 meses, e não houve sinal.

---

## Slide 15 · Features: escolhidas todas as 21

**Está no slide:** básico com 5 features, 0,018 (t 4,9); todas as 21, 0,023 (t 5,3). Os sete grupos: retornos recentes, retornos acumulados, posição no dia, volatilidade, forma do candle, volume e gap, dólar.

Quadro: bruto +0,43 ± 0,11 · após custo −4,57 · acerto 51,3%.

**Fala:**
> "Testamos conjuntos de variáveis, todas conhecidas no fechamento do candle. Com as 5 básicas, a correlação foi 0,018. Com as 21, subiu para 0,023. Filtrar as variáveis pela análise de dados piorou: cada uma foi avaliada sozinha, e o Random Forest usa as combinações."

**Como foi testado:**
1. **Análise até 2023**, por feature: Spearman com t mensal, mesmo sinal em 2021, 2022 e 2023, informação mútua em decis e redundância (correlação acima de 0,8).
2. **Modelo em 2024**, com o RF do passo 1, comparando os conjuntos:

| Conjunto | Features | Correlação | t | Bruto por operação |
|---|---|---|---|---|
| Básico | 5 | 0,018 | 4,9 | +0,37 |
| **Todas** | **21** | **0,023** | **5,3** | **+0,43** |
| Pré-selecionadas pela análise | 13 | 0,012 | 4,2 | +0,11 |
| Confirmadas por permutação | 5 | 0,015 | 8,2 | +0,08 |
| Finais, após retirada | 4 | 0,017 | 7,3 | +0,33 |

**Outro cenário bom:** o conjunto de **4 features** (`ret_acum_15c`, `pos_range_dia`, `ret_acum_5c`, `ret_2`) tem correlação menor (0,017), mas t maior (7,3): é o mais estável entre os meses. Ficou fora da faixa de empate (0,019), por isso não foi escolhido.

### Importância das features

São **três medidas diferentes**. Não as misture na fala.

**1. Relação de cada feature, sozinha, com o próximo retorno (até 2023):**

| Feature | t mensal | Leitura |
|---|---|---|
| `ret_acum_15c` | −8,9 | subiu em 15 min, tende a devolver |
| `pos_range_dia` | −7,1 | perto da máxima do dia, tende a cair |
| `ret_dia` | −5,6 | redundante com a anterior (0,84) |
| `ret_acum_60c` | −5,3 | reversão de 60 min |
| `wdo_ret_acum_15c` | +4,5 | dólar dos últimos 15 min |
| `ret_acum_5c` | −4,5 | reversão de 5 min |

As seis têm o mesmo sinal nos três anos. As de volatilidade (`vol_20`, `vol_60`, `range_hl`) informam o **tamanho** do movimento, e não a direção.

**2. Permutação no modelo de 13 features (2024, modelo intermediário):** `pos_range_dia`, `ret_acum_15c`, `ret_acum_5c`, `ret_2` e `ret_1` foram confirmadas.

**3. Permutação no modelo final, de 21 features (2024, calculada em 23/11):** queda da correlação ao embaralhar cada feature.

| Feature | Folha 10% (modelo entregue) | Folha 5% (etapa 5) |
|---|---|---|
| **`pos_fech_candle`** (onde o candle fechou entre a mínima e a máxima) | **0,0078** | **0,0094** |
| `ret_acum_15c` | 0,0017 | 0,0022 |
| `ret_0` | 0,0011 | 0,0017 |
| `gap_abertura` | 0,0008 | 0,0004 |
| `wdo_ret_acum_15c` | 0,0006 | −0,0002 |
| `pos_range_dia` | −0,0002 | −0,0001 |
| correlação do modelo | 0,0239 | 0,0263 |

> **Fala sobre importância:** "Sozinhas, as variáveis mais fortes são as de reversão: o retorno dos últimos 15 minutos e a posição no dia. Dentro do modelo final, a que mais pesa é onde o candle fechou entre a mínima e a máxima, seguida da reversão de 15 minutos. O resto contribui pouco, uma por uma."

**Atenção (ponto sensível):**
- `pos_fech_candle` é a mais importante no modelo final, mas na análise até 2023 **não tinha relação estável** (Spearman +0,008 em 2021, −0,004 em 2022 e −0,028 em 2023). Parte do sinal de 2024 pode ser recente.
- Ela é compatível com o *bid-ask bounce* (o preço oscilar entre compra e venda). Na frente original, entrar na abertura seguinte manteve +0,50 de +0,59 bruto, o que sugere que o efeito não explica tudo. Não temos dados de book para confirmar.
- **Não diga** que o sinal do modelo final "vem da reversão, do range e do dólar": isso vale para a análise isolada. No modelo final, range e dólar quase não pesam.
- Permutação mede o peso de uma feature com as outras presentes. Features parecidas, como `ret_0` e `corpo` (0,99), dividem o peso.

---

## Slide 16 · Janela de treino: 12 meses

**Está no slide:**

| Janela | Correlação | t |
|---|---|---|
| 3 meses | 0,012 | 2,3 |
| 6 meses | 0,017 | 3,3 |
| **12 meses** | **0,024** | **6,0** |
| 24 meses | 0,010 | 2,3 |
| Todo o histórico | −0,006 | −1,2 |

Quadro: bruto +0,51 ± 0,11 · após custo −4,49 · acerto 51,4%.

**Fala:**
> "Quanto passado o modelo deve ver? Com 3 ou 6 meses, faltam dados. Com 24 meses ou todo o histórico, misturamos épocas com preço e volatilidade diferentes. Com 12 meses a correlação foi a maior, 0,024 com t de 6."

**Como foi testado:** retreino trimestral e RF mais leve (100 árvores, até 50 mil linhas por árvore), igual em todas. O número da etapa anterior (0,023) não é exatamente comparável.

**Outro cenário bom:** 6 meses com retreino trimestral, 0,017 (t 3,3).

**Atenção:** a frente original e a de 60 min chegaram a conclusões diferentes sobre janela. Lá a janela longa ajudou no candle de 60 min. Aqui é 1 min.

---

## Slide 17 · Retreino: semestral

**Está no slide:** mensal 0,024 (t 5,2), trimestral 0,024 (t 6,0), semestral 0,022 (t 4,8). Faixa de empate de ±1 erro-padrão do melhor.

Quadro: bruto +0,44 ± 0,11 · após custo −4,56 · acerto 51,0%.

**Fala:**
> "As três frequências empataram. Pela regra, fica a mais simples: semestral, com só dois treinos por ano. O trimestral foi o melhor no número e ficou anotado."

**Como foi testado:** 9 combinações de janela e retreino, com as duas melhores janelas (12 e 6 meses) também em mensal e semestral.

**Outros cenários bons:**

| Cenário | Correlação | t | Bruto por operação |
|---|---|---|---|
| 12 meses, mensal | 0,0241 | 5,2 | **+0,56** |
| 12 meses, trimestral | **0,0242** | **6,0** | +0,51 |
| 12 meses, semestral (escolhido) | 0,0225 | 4,8 | +0,44 |

**O modelo envelhece?** Sim, segundo a análise até 2023: a correlação cai cerca de 0,002 por mês depois do treino (t = −2,5) e some em torno de 9 meses. Um retreino a cada 6 meses fica dentro do prazo. Os pontos mensais variam muito, portanto é uma tendência, não uma regra.

---

## Slide 18 · Número de árvores: 50

**Está no slide:** 25 árvores 0,0188 · 50 árvores 0,0216 · 100 árvores 0,0225 · 200 árvores 0,0217 · 400 árvores 0,0220. Quadro: bruto +0,42 ± 0,11 · após custo −4,58 · t 5,1.

**Fala:**
> "Mais árvores só deixam a média mais estável. A partir de 50, a correlação fica a menos de 0,001 do valor com 400 árvores. Mais do que isso só custa tempo."

**Como foi testado:** o mesmo modelo cresceu de 25 até 400 árvores. Regra: o menor número a menos de 0,001 do valor com 400.

**Outro cenário bom:** 100 árvores (0,0225), a maior, dentro da tolerância.

---

## Slide 19 · Profundidade e folha: rasas e grandes

**Está no slide:** o mapa da correlação em 2024, com o sobreajuste (treino menos teste) embaixo.

| | Folha 0,5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| Prof. 2 | – | – | – | **0,026** | **0,024** |
| Prof. 3 | 0,014 | 0,019 | 0,021 | 0,025 | 0,024 |
| Prof. 5 | 0,020 | 0,022 | 0,019 | 0,024 | – |
| Prof. 8 | 0,019 | 0,020 | 0,020 | 0,024 | – |

**Fala:**
> "Testamos quão complexa deve ser cada árvore. Árvores fundas com folhas pequenas decoram o treino: na profundidade 8 com folha de 0,5%, a correlação é 0,082 no treino e só 0,019 no teste. Árvores rasas com folhas grandes foram melhores. O melhor caiu na borda da grade, então a estendemos."

**Como foi testado:** profundidades 3, 5 e 8 × folhas 0,5% a 5%; depois raiz e metade das features nas duas melhores; depois a extensão. São 21 configurações ao todo.

**Conta que apoiou a escolha:** para a média de uma folha ser distinguível de zero, seriam necessários cerca de 540 mil candles, mais que a janela de 12 meses (cerca de 131 mil). Nenhuma folha, sozinha, é significativa: o sinal vem da média de muitas.

**Atenção:** com folha de 10%, profundidades 2 e 3 dão o mesmo modelo. A árvore não consegue dividir mais.

---

## Slide 20 · Modelo final

**Está no slide:**

| Configuração | Correlação | t |
|---|---|---|
| prof. 3 · folha 5% · todas | 0,025 | 4,7 |
| prof. 3 · folha 5% · raiz | 0,026 | 4,6 |
| prof. 5 · folha 5% · metade | 0,026 | 4,6 |
| **prof. 2 · folha 5% · raiz** | **0,026** | **4,9** |
| **prof. 2 · folha 10% · raiz (escolhida)** | 0,024 | 4,1 |
| prof. 2 · folha 20% · raiz | previsão constante | – |

Quadro: bruto +0,26 ± 0,11 · após custo −4,74 · acerto 51,2%.

**Fala:**
> "Várias configurações empataram. A maior foi a profundidade 2 com folha de 5%, com 0,026. Pela regra, ficou a mais simples entre as empatadas: folha de 10%. Ela achata as previsões, e o ganho bruto caiu de 0,37 para 0,26. Por isso a folha de 5% seguiu como alternativa, e foi ela que ganhou na etapa 5."

**Modelo entregue:** 1 min · 21 features · 12 meses · semestral · 50 árvores · profundidade 2 · folha 10% · raiz das features.

**Outros cenários bons:** todos os da tabela acima, menos o de 20%. Maior ganho bruto da grade: prof. 5, folha 2% (+0,49), com correlação de 0,019.

**Se perguntarem por que não a maior correlação:** a diferença (0,026 contra 0,024) está dentro de 1 erro-padrão. Escolher o maior entre empatados costuma ser escolher o mais sortudo.

---

## Slide 21 · Ganho acumulado em 2024

**Está no slide:** +36,8 mil pts sem custo e −670 mil com custo. 1º semestre, +0,01 pt por operação; 2º, +0,50. Treino × teste: bruto +0,99 e +0,68 contra +0,26; correlação 0,034 contra 0,024.

**Fala:**
> "Este é o modelo final operando os 141 mil candles de 2024. Sem custo, soma 36,8 mil pontos, quase tudo no segundo semestre. Com 5 pontos por operação, termina em menos 670 mil: são 707 mil de custo contra 37 mil de ganho. O sinal existe, mas operar todo candle não paga. Daí a etapa 5: operar só os candles mais fortes."

**Se perguntarem do 1º semestre fraco:** pode ser azar ou mudança de regime. A correlação foi positiva em 11 dos 12 meses; o ganho em pontos depende de poucos movimentos grandes.

---

## Slide 22 · Etapa 5: só operamos as caudas

**Está no slide:** a régua vende / não opera / compra, com limites de 2,5% do treino em cada ponta. Entrada na abertura seguinte, uma posição por vez. Saída por stop de 1,5× a volatilidade ou 120 minutos. Custo de 5 pts.

**Fala:**
> "Pegamos as previsões do treino e marcamos os limites das caudas: as 2,5% mais altas viram compra, as 2,5% mais baixas viram venda, e o resto não opera. Entramos na abertura do candle seguinte, com uma posição por vez. A saída é um stop proporcional à volatilidade ou 120 minutos."

**Como foi testado / detalhes:**
- **Distância do stop:** 1,5 × volatilidade dos últimos 20 minutos × preço × √120, cerca de 1.025 pts em média em 2022–23.
- Se o preço abre além do stop, sai na abertura. TP e stop no mesmo candle contam como stop.
- Sem entradas depois de 17:50; tudo zerado até 18:20.
- **Por que duas caudas, e não |previsão|:** a previsão tem média positiva, e o valor absoluto daria compras demais.
- Os limites são do treino. Fora dele, a fração operada muda um pouco: 3,0% compra e 1,7% vende (2022–2024).

---

## Slide 23 · 72 regras: o alvo não ajudou

**Está no slide:** mapa do líquido por pregão em 2022–23.

| | sem stop | stop 1× | stop 1,5× |
|---|---|---|---|
| **sem alvo** | +130 | +123 | +124 |
| **alvo 1×** | +73 | +75 | +66 |
| **alvo 1,5×** | +112 | +83 | +102 |

À direita, o que variou: entrada, lado, tempo, alvo e stop, filtro, modelo.

**Fala:**
> "Testamos 72 regras em 2022 e 2023, em etapas: primeiro entrada, lado e tempo; depois alvo e stop; por fim, filtros. O take profit reduziu o ganho, porque corta os movimentos grandes. O stop largo quase não mudou a média, mas reduz a oscilação. A regra de escolha é a média diária menos um erro-padrão, lucro nos dois anos e pelo menos 100 operações por ano. 45 das 72 passaram, e as três melhores foram a 2024."

**Como foi testado:**
- **Etapa A (54):** 2 modelos × 3 cortes × 3 lados × 3 tempos.
- **Etapa B (16 novas):** alvo e stop nas 2 melhores entradas.
- **Etapa C (2 novas):** manhã e volatilidade alta. Nenhum filtro melhorou (+80 e +59 por pregão contra +130 sem filtro).
- Custo de 5 pts em todas, com previsões walk-forward.

**Outros cenários bons em 2022–23 (líquido por pregão):**

| Regra | Por pregão | Por operação |
|---|---|---|
| caudas de 5%, 120 min, sem saída | +130 | +40,1 |
| caudas de 5%, 120 min, stop 1,5× (**melhor em 2024**) | +124 | +36,9 |
| caudas de 5%, 120 min, TP 1,5× | +112 | +33,1 |
| caudas de 10%, 60 min, TP 1,5× | +105 | +14,4 |

**Atenção:** o erro-padrão de cada célula é de dezenas de pontos. As diferenças entre saídas não são significativas.

---

## Slide 24 · O ganho não se repetiu (slide principal)

**Está no slide:** +124 (2022–23), +7 (2024) e −114 (01/2025–03/2026) por pregão. Por operação: +36,9, +1,8 e −28,3. Meses positivos: 7 de 12 e 4 de 15. Percentil contra sorteios: 57 e 11. Destaque: das 45 regras lucrativas em 2022 e em 2023, 12 foram positivas em 2024, 12 em 2025–26 e 2 nos dois.

**Fala:**
> "Aqui está o resultado central. Em 2022 e 2023 a regra parecia ótima: 124 pontos por pregão. Em 2024 ficou perto de zero, e no período seguinte perdeu 114 por pregão. Nenhum dos três finalistas passou no nosso critério, que exigia ficar acima de 95% dos sorteios de direção nas mesmas entradas. E não foi azar de escolha: das 45 regras lucrativas em 2022 e 2023, só 12 foram positivas em 2024, 12 em 2025–26, e só duas nos dois períodos."

**Como foi testado:**
- Regra congelada, retreinada por semestre, 1.254 operações e −35.530 pts no total (cerca de −R$ 7.106 por contrato).
- **2.000 sorteios:** mesmas entradas e saídas, direção ao acaso.
- **Diagnóstico das 45:** feito depois da validação, para ver o conjunto, e não para escolher.

**Outros cenários bons (só citar se perguntarem):** duas regras ficaram positivas nos três períodos: folha de 10%, ambos os lados, 120 min, sem saída, caudas de 30% e de 5%. Em 2025–26, +0,2 e +2,8 pts por operação, percentis 62 e 65. São compatíveis com sorte.

**Atenção (honestidade):**
- A regra rodada em 2025–26 **não estava aprovada em 2024**. Rodamos para registro.
- 2025–26 já tinha sido consultado por outras abordagens da frente. **Não diga "validação única".**

---

## Slide 25 · Prevê, mas não paga o custo

**Está no slide:** correlação de +0,029 (t 5,8, Theil U 0,99995) fora do período de escolha; correlação necessária de ~0,11; e a inversão do lado.

| Bruto por operação, mesmas entradas | 2024 | 2025–26 |
|---|---|---|
| Só vender | +15,0 | −34,5 |
| Só comprar | −6,3 | +34,1 |
| Regra | +6,8 | −23,3 |

**Fala:**
> "O modelo prevê: fora do período de escolha a correlação foi 0,029, com t de 5,8. Mas o sinal é pequeno. Para pagar 5 pontos operando cada candle de 1 minuto, seria preciso perto de 0,11, quase quatro vezes mais. E o lado que ganha muda: em 2024, só vender teria ganhado; em 2025 e 2026, só comprar. Por isso alvo e stop não resolvem: eles controlam quanto se ganha e perde, não a direção."

**Detalhe:** a mesma folha de 5% teve 0,026 (t 4,9) em 2024. Esse número de 2024 foi usado na escolha; o de 2025–26, não.

**Outro pipeline confirma a previsão:** a frente original, com outras 16 features, teve Clark-West t = 3,72 e +0,59 pt bruto no mesmo período.

**Conclusão a dizer:** "Pela regra que definimos, a decisão é não operar."

---

## Slide 26 · Outras abordagens

**Está no slide:** seis abordagens, com característica, motivo e resultado, divididas em antes e depois da etapa 5.

**Fala:**
> "Antes da etapa 5, tentamos pagar menos custo de três formas: segurar a posição, prever só os movimentos maiores e usar candles de 60 minutos. Depois, ajustamos o alvo, as caudas e o stop. Cada teste tinha um motivo. Nenhum deu lucro significativo."

| Abordagem | Como foi testada | Resultado, líquido por operação |
|---|---|---|
| Segurar a posição | RF 1 min; limiar de 90% fixado em 2024; troca quando o sinal vira | −2,5 (2024), −0,7, +1,9 (t 0,64) |
| Classificador | 3 classes, 30 variáveis; nenhuma candidata com t ≥ 2; opera sempre | −4,5, −4,8, −4,1 |
| Candle de 60 min | 36 meses, trimestral, 15 variáveis; opera se prevê > 5 pts | +14,1 (t 1,76), +3,0 (t 0,29), −4,8 |
| Alvo de 30–120 min | 48 regras em 2022–23, 3 finalistas | +0,7, +0,4 e +1,0 em 2024 |
| Caudas extremas | 36 regras, 2% a 0,5% | −23,9, −7,4 e −4,9 em 2024 |
| Stop mais rigoroso | 18 regras; um teste extra declarado | +1,8 (repete a etapa 5) e −17,5 |

Nas três primeiras, a sequência é 2024, 01/2025–03/2026 e 04–09/2026.

**Sobre o critério, se perguntarem:**
- Classificador e RF original exigiam lucro com t ≥ 2. Nada passou.
- **No candle de 60 min, o critério escrito escolhia 1 min** (t 2,3) e foi trocado pelo lucro após o custo. É por isso que o slide diz "escolhida pelo lucro de 2024".
- Na frente original, a ordem dos 12 cenários por Theil U se repetiu no teste (0,99); a ordem por lucro, não (−0,24).

**Atenção:** a de 60 min soma +R$ 441 em 21 meses (t = 0,08). Só comprar nas mesmas entradas renderia +14,9 por operação em 01/2025–03/2026. É empate, e não lucro.

---

## Fechamento da frente (30 s)

> "O Random Forest acha um sinal real, pequeno e que se repete fora do período de escolha. Ele não paga 5 pontos por operação em nenhuma das formas que testamos. O protocolo evitou que apresentássemos um lucro que não existe. O próximo passo é esperar dados novos, depois de setembro de 2026."

## Perguntas prováveis

- **"Quantas regras vocês testaram?"** "72 na etapa 5 e, somando os testes posteriores, 170 distintas nesse pipeline, mais 24 exploratórias. As outras abordagens têm contagem própria."
- **"Por que não usar as que deram lucro?"** "Escolhê-las depois de ver o resultado é o erro que o protocolo evita. Elas não se distinguem de sorteio."
- **"51% de acerto não é moeda?"** "Para cada candle, quase. O ponto é a consistência entre os meses. O problema é o tamanho."
- **"Por que não trocar o custo?"** "É o custo do trabalho inteiro, e vale para as três frentes."
- **"O modelo usa o quê?"** "Principalmente onde o candle fechou entre a mínima e a máxima e a reversão de 15 minutos. Ainda não separamos isso do bid-ask bounce por completo."
- **"O 2025–26 foi usado uma vez?"** "A regra da etapa 5 rodou uma vez. O período já tinha sido consultado por outras abordagens da frente."

## Fontes dos números

`resultados/passo1_*` a `passo5_*`, `passo4_importancia_final.csv`, `passo5_diagnostico_todas.csv`, `passo7_*`, `redesenho_*`, `caudas_*`, `stop_*`; `resultados_ml/`; `rf_do_zero/saidas/`.
