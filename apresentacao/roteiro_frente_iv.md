# Roteiro de apresentação — Frente IV · Random Forest

Slides 13 a 22 e 29 de `slides.html` · duração sugerida: 15 a 18 minutos

Este roteiro tem quatro partes:

1. **Conceitos de base.** O que é o modelo, como ele é testado e o que significa cada métrica. Vale ler antes de ensaiar; a fala dos slides usa esses conceitos.
2. **Fala slide a slide.** O que dizer em cada slide, com os números e o motivo de cada escolha.
3. **Por que a escolha final segue a correlação, e não o ganho bruto.** O argumento central, que a banca provavelmente vai cobrar.
4. **Perguntas prováveis e respostas curtas.**

Todos os números vêm do **teste de 2024**. A validação de 2025–26 ainda não foi usada.

---

## Parte 1 — Conceitos de base

### 1.1 O problema

- **Alvo:** o retorno do próximo candle do WIN, em %: `r(t+1) = 100 · ln(C(t+1) / C(t))`, onde C é o preço de fechamento. Só entram pares de candles do mesmo pregão. Não se prevê a passagem da noite, porque o gap de abertura é outro fenômeno.
- **Custo:** cada operação completa (entrar e sair) custa **5 pontos**, que são as taxas da B3 mais meio spread. Para valer dinheiro, a previsão precisa ganhar em média mais que isso por operação.
- **1 ponto = R$ 0,20** por minicontrato.

### 1.2 Como o tempo foi dividido

| Período | Uso | Por quê |
|---|---|---|
| 15/10/2020 → 31/12/2023 | **Exploração:** análise de dados, sem modelo | Entender os dados sem "espiar" o período em que as escolhas são medidas |
| 2024 | **Teste:** todas as escolhas são medidas aqui | Cada período de 2024 é previsto por um modelo que nunca o viu |
| 01/2025 → 03/2026 | **Validação:** rodada uma única vez, com tudo congelado | Medida honesta do desempenho, sem nenhuma escolha feita olhando para ela |

**Walk-forward**, como funciona o teste:
1. Treina-se o modelo com os 12 meses anteriores a um período de 2024 (por exemplo, o 1º semestre).
2. Ele prevê esse período.
3. Avança-se para o período seguinte e repete-se.

As previsões dos vários períodos formam o ano de 2024. É o que aconteceria na prática: o modelo só conhece o passado.

**Por que isso importa:** em séries financeiras, testar com dados embaralhados (validação cruzada comum) deixa o modelo "ver o futuro". Os números saem bons demais e não se repetem.

### 1.3 O que é um Random Forest

- **Árvore de decisão:** uma sequência de perguntas sobre as features ("o retorno dos últimos 15 minutos foi maior que +0,05%?"). Cada pergunta divide os candles em dois grupos. No fim, cada grupo (uma **folha**) prevê a média do retorno seguinte dos candles de treino que caíram nele.
- **Floresta (Random Forest):** muitas árvores, cada uma treinada com:
  - uma amostra aleatória dos candles de treino;
  - em cada divisão, um subconjunto aleatório das features.

  A previsão final é a **média das árvores**. Árvores diferentes erram de jeitos diferentes, e a média cancela parte do erro.
- **Por que RF neste problema:**
  - **Capta relações não lineares e condicionais.** Por exemplo: "reversão só quando o dia já andou muito". O ARIMA só enxerga relações lineares com o último retorno.
  - **Aceita muitas features** de escalas diferentes, sem padronizar.
  - **É controlável:** a profundidade e o tamanho da folha regulam o quanto ele pode "decorar" o treino.

**Hiperparâmetros usados no passo 4:**

| Hiperparâmetro | O que é | Efeito |
|---|---|---|
| Nº de árvores | quantas árvores entram na média | mais árvores = média mais estável; **não** aumenta o sobreajuste |
| Profundidade máxima | quantas perguntas em sequência cada árvore faz | mais funda = aprende interações mais complexas, mas decora mais ruído |
| % do treino por folha | mínimo de candles para formar uma folha | folha grande = média mais confiável, previsão mais suave |
| `max_features` | quantas features cada divisão pode considerar | menos = árvores mais diferentes entre si, e a média funciona melhor |

**O que o modelo devolve (a saída):**
- O modelo é um **regressor**. A saída é o **retorno previsto do próximo candle, em %, com sinal**, ou seja, "quanto vai subir" ou "quanto vai descer".
- **Positivo** = prevê alta; **negativo** = prevê queda. O valor é a magnitude.
- **Em pontos:** pontos previstos ≈ preço atual × previsão ÷ 100. Com o WIN em 130.000 pontos, uma previsão de +0,002% equivale a +2,6 pontos.
- **Ressalva:** a previsão é a **média** dos retornos seguintes dos candles de treino parecidos com o atual (a mesma folha). Com sinal fraco, subidas e descidas dentro da folha quase se anulam, e a média fica perto de zero. O desvio das previsões do modelo final é cerca de 1% do desvio dos movimentos reais: o modelo diz "+0,5 ponto" quando o candle de 1 minuto costuma andar ~40 pontos.
- **Consequência:** a **ordem** das previsões é informativa (os candles com previsão mais alta sobem mais, em média), mas a **escala** não é realista. Por isso o modelo é avaliado pela correlação, que mede a ordem.

**Como falar, se perguntarem:**
> "A saída é o retorno previsto do próximo candle, em %, com sinal: positivo para alta, negativo para queda, e o valor é a magnitude. Por ser a média condicional de um sinal fraco, a magnitude prevista é muito menor que o movimento real. Por isso usamos a previsão principalmente para ordenar os candles, e não como estimativa literal de quantos pontos o preço vai andar."

**Se pedirem uma magnitude mais realista**, há dois caminhos para os próximos passos:
1. **Calibrar a escala** com uma regressão do retorno real sobre o previsto.
2. **Separar em dois modelos:** um para a direção e outro para o tamanho do movimento. O tamanho é bem previsível pelas features de volatilidade.

### 1.4 As métricas, uma por uma

#### Correlação de Spearman (a métrica principal)

**O que é:** mede se a **ordem** das previsões bate com a **ordem** dos retornos que aconteceram. Vale +1 se a ordem é idêntica, 0 se não há relação e −1 se é invertida.

**Como é calculada:**
1. Ordenam-se as previsões e dá-se a cada uma o seu **posto** (1ª, 2ª, 3ª...).
2. Faz-se o mesmo com os retornos reais.
3. Calcula-se a correlação comum (de Pearson) entre os dois postos.

Exemplo com 5 candles:

| Candle | Previsto | Posto | Real | Posto |
|---|---|---|---|---|
| A | +0,010 | 5 | +0,08 | 4 |
| B | −0,004 | 1 | −0,05 | 1 |
| C | +0,002 | 3 | +0,01 | 3 |
| D | +0,006 | 4 | +0,12 | 5 |
| E | −0,001 | 2 | −0,02 | 2 |

Os postos quase coincidem, e a correlação é de 0,9. Note que **o tamanho** da previsão não importa, só a ordem.

**Por que Spearman, e não Pearson ou o erro quadrático:**
1. **O RF encolhe as previsões.** Com sinal fraco, ele prevê valores muito perto de zero: o desvio das previsões é 1% a 2% do desvio dos retornos reais. Métricas de escala (erro quadrático, Theil U) quase não distinguem isso de "prever zero". A ordem, por outro lado, continua informativa.
2. **Caudas pesadas.** Alguns candles de notícia andam 20× o normal. Na correlação de Pearson, eles dominariam o resultado. Os postos limitam a influência de cada candle.
3. **É o que a operação usa.** A regra de compra e venda (passo 5) vai operar os candles com as previsões mais fortes, os decis extremos. Se o modelo **ordena bem**, os extremos dele concentram os melhores movimentos. A correlação de postos mede isso diretamente.

**Ordem de grandeza:** valores como 0,02 parecem minúsculos, mas em retornos de 1 minuto são relevantes. O retorno de um candle praticamente não tem memória do anterior (ρ₁ ≈ 0). Qualquer correlação consistente acima de zero já é informação que o mercado não precificou por completo.

#### Correlação média mensal e t robusto (a significância)

**Como é calculado:**
1. Calcula-se o Spearman **separadamente em cada um dos 12 meses** de 2024.
2. Tira-se a média dessas 12 correlações.
3. Erro-padrão (EP) = desvio-padrão entre os meses ÷ √12.
4. **t robusto** = média ÷ EP.

**Por que não usar o erro-padrão "de livro" (1/√n):** com 141 mil candles de 1 minuto em 2024, 1/√n ≈ 0,003. Qualquer correlação de 0,006 pareceria significativa. Só que os candles não são independentes: a volatilidade vem em ondas, e um dia agitado produz centenas de candles parecidos. Agrupar por mês trata cada mês como uma observação e captura essa dependência. Na prática, o t robusto responde: **"o sinal aparece de forma consistente, mês após mês?"**

**Leitura:**
- **|t| > 2:** sinal distinguível do ruído, com ~95% de confiança.
- **"Meses com ρ > 0"** mostra a mesma coisa de forma intuitiva. O modelo final teve correlação positiva em 11 dos 12 meses.

#### Theil U (filtro, não critério)

- **Fórmula:** `U = RMSE(modelo) ÷ RMSE(prever retorno zero)`, onde RMSE é a raiz do erro quadrático médio.
- **Leitura:** U < 1 significa que o modelo erra menos que o palpite mais simples, "o preço não muda".
- **Por que só como filtro:** como o RF encolhe as previsões, U fica entre 0,9999 e 1,0005 em todos os cenários. A diferença está na 4ª ou 5ª casa decimal e não ordena nada. Ele serve para garantir que o modelo **não é pior** que prever zero. Usamos uma tolerância de U ≤ 1,001: até 0,1% pior conta como empate.

#### Acerto de direção

- **O que é:** a % de candles em que o sinal da previsão (sobe/desce) acertou.
- **Valores:** ficou entre 50,9% e 51,4%.
- **Por que não é critério:** ignora o tamanho do movimento. Acertar 60% de movimentos de 1 ponto e errar 40% de movimentos de 20 pontos perde dinheiro. Entra só como informação complementar.

#### Ganho bruto por operação

- **O que é:** a média, em pontos, de operar o sinal da previsão **em todos os candles**: comprar quando a previsão é positiva, vender quando é negativa, sair no candle seguinte. É calculado **antes do custo**.
- **Fórmula:** `ganho = média( sinal(previsão) × (C(t+1) − C(t)) )`, com erro-padrão = desvio ÷ √n.
- **Para que serve:** traduz a previsão para pontos e mostra a distância até o custo de 5 pts.
- **Por que NÃO é o critério de escolha:** ver a Parte 3.

#### Correlação necessária para pagar o custo

Se a previsão e o movimento seguem uma relação aproximadamente normal, operar o sinal rende em média cerca de:

```
ganho esperado ≈ ρ × σ_pts × 0,8
```

onde σ_pts é o desvio do movimento do candle, em pontos. Daí sai:

```
ρ necessário = custo ÷ (σ_pts × 0,8)
```

- No candle de 1 minuto, σ_pts ≈ 58 pts, e o ρ necessário para pagar 5 pts é ≈ **0,11**.
- Em candles maiores, o preço anda mais e o ρ necessário cai (0,014 no candle de 60 min).

É uma aproximação: os retornos têm caudas pesadas. Serve como **ordem de grandeza** da distância até o lucro.

### 1.5 A regra de desempate (usada em todos os passos)

> Entre as opções a até **1 erro-padrão** da melhor, fica a **mais simples**: menos features, retreino menos frequente, árvore mais rasa, folha maior.

**Por quê:** diferenças menores que o erro-padrão são indistinguíveis de sorte. Escolher "o maior número" entre opções empatadas é escolher o ruído mais favorável, e ele não se repete fora da amostra. Foi exatamente o que aconteceu com o ARIMAX:
- foi escolhido pela maior esperança na seleção, +10 ± 19 pts;
- na validação, deu −41 ± 42.

A opção mais simples tem menos chance de ter "decorado" o período de teste.

**Os critérios foram escritos antes de rodar cada passo.** Isso impede ajustar a regra depois de ver o resultado.

---

## Parte 2 — Fala slide a slide

### Slide 13 — Roteiro: seis passos, um grupo de decisões por vez (≈1 min)

**Fala:**
> "Na frente de machine learning usamos um Random Forest, que é uma média de muitas árvores de decisão. Em vez de testar todas as combinações de uma vez, o que geraria milhares de modelos e muito risco de achar algo por acaso, dividimos as decisões em seis passos. Cada passo decide um grupo de parâmetros e congela a escolha antes do próximo. A ordem vai do que define o problema (o tamanho do candle) ao que só ajusta o resultado (os hiperparâmetros). Concluímos os passos 1 a 4. O passo 5, as regras de compra e venda, está em desenvolvimento."

**Ponto-chave:** cada escolha foi medida no teste de 2024 e seguiu um critério escrito antes de rodar.

**Se perguntarem por que essa ordem:**
- o candle muda o próprio alvo e a escala de tudo, por isso vem primeiro;
- as features são a decisão de maior impacto;
- a janela muda o tamanho do treino, e a folha é definida em % do treino;
- o RF é pouco sensível aos hiperparâmetros, por isso eles vêm por último;
- as regras nunca voltam a mudar os passos anteriores.

### Slide 14 — IV.1 · Candles: escolhido o de 1 minuto (≈2 min)

**O que se decidiu:** o horizonte da previsão, ou seja, o tamanho do candle: 1, 5, 15, 30 ou 60 minutos.

**Fala:**
> "Primeiro, o tamanho do candle. Há um dilema: em candles grandes, o preço anda mais, e o custo de 5 pontos pesa menos. No de 60 minutos, ele é 1,6% do movimento médio; no de 1 minuto, 12%. Por outro lado, candles pequenos dão muito mais amostras e o sinal pode ser mais curto que uma hora. Treinamos o mesmo RF, com as mesmas 5 features básicas, nos cinco tamanhos. Só o de 1 minuto teve sinal fora do ruído: correlação de 0,018, t de 4,9, positiva nos 12 meses. Os outros ficaram com t entre −0,5 e +0,3, ou seja, nenhum sinal."

**Análise de dados que apoiou o passo (até 2023):**
- **ρ₁**, a correlação do retorno de um candle com o do seguinte, ficou dentro da faixa de ruído em **todos** os tamanhos. Não há memória linear simples. É o mesmo achado da frente ARIMA, e qualquer ganho do RF tem de vir de relações não lineares ou de outras features.
- **Correlação necessária para pagar 5 pts:** 0,107 (1 min), 0,049 (5 min), 0,029 (15 min), 0,020 (30 min), 0,014 (60 min). Os candles grandes exigem menos, mas o RF não achou sinal neles.

**Por que 1 min, mesmo com o custo pesando mais:** é o único com sinal estatístico. Sem sinal, o custo baixo não ajuda. A opção de 5 min fica anotada como alternativa para a comparação final, porque o ranking mudou quando testamos outras janelas.

### Slide 15 — IV.2 · Features: escolhidas as 21 (≈3 min)

**O que se decidiu:** quais informações o modelo enxerga.

**As 21 features**, todas calculadas só com informação disponível no fechamento do candle t:

| Grupo | Feature | O que mede |
|---|---|---|
| Retornos recentes | `ret_0`, `ret_1`, `ret_2` | retorno do candle atual e dos 2 anteriores |
| Retornos acumulados | `ret_acum_5c`, `ret_acum_15c`, `ret_acum_60c` | soma dos retornos dos últimos 5, 15 e 60 candles, dentro do dia |
| Posição no dia | `ret_dia` | quanto o preço andou desde a abertura do dia |
| | `pos_range_dia` | onde o preço está entre a mínima (0) e a máxima (1) do dia até agora |
| | `hora` | horário do candle |
| Volatilidade | `vol_20`, `vol_60` | desvio dos últimos 20 e 60 retornos |
| | `vol_razao` | `vol_20 ÷ vol_60`: a agitação está subindo ou caindo? |
| | `rv_dia_ant`, `rv_media_5d` | volatilidade realizada do dia anterior e média dos últimos 5 dias (termos do modelo HAR) |
| Forma do candle | `range_hl` | tamanho do candle (máxima ÷ mínima) |
| | `corpo` | fechamento ÷ abertura do candle |
| | `pos_fech_candle` | onde o candle fechou dentro da própria faixa |
| Volume | `vol_rel` | volume ÷ média do mesmo horário nos 20 pregões anteriores |
| Abertura | `gap_abertura` | gap do dia sobre o fechamento anterior |
| Dólar (WDO) | `wdo_ret_0`, `wdo_ret_acum_15c` | retorno do dólar no candle atual e nos últimos 15 |

**Análise de dados (até 2023), para cada feature:**
1. **Spearman com o alvo e t mensal:** existe relação com a direção do próximo candle, consistente mês a mês?
2. **Estabilidade:** a relação tem o mesmo sinal em 2021, 2022 e 2023?
3. **Informação mútua em decis:** mede qualquer relação, inclusive não linear. O cálculo:
   - divide-se a feature e o alvo em 10 faixas;
   - mede-se quanto saber a faixa da feature reduz a incerteza sobre a faixa do alvo;
   - **desconta-se** o valor que aparece com o alvo embaralhado, ou seja, por acaso.
4. **Redundância:** pares com correlação acima de 0,8 carregam a mesma informação. Foram encontrados três:
   - `ret_0` × `corpo`: 0,99;
   - `ret_dia` × `pos_range_dia`: 0,84;
   - `vol_20` × `vol_60`: 0,86.

**O que os dados mostraram** (o sinal tem uma história econômica):
- **Reversão de curto prazo:** depois de subir nos últimos 15 minutos, o próximo minuto tende a cair (`ret_acum_15c`, t = −8,9). Movimentos rápidos tendem a ser parcialmente devolvidos.
- **Posição no range do dia:** perto da máxima do dia, a tendência é de queda; perto da mínima, de alta (`pos_range_dia`, t = −7,1).
- **Dólar:** quando o dólar subiu nos últimos 15 minutos, o índice tende a subir no minuto seguinte (t = +4,5). A relação é contemporaneamente negativa, mas tem um efeito defasado positivo.
- As três têm **o mesmo sinal em 2021, 2022 e 2023**.
- As features de **volatilidade** dizem quanto o preço vai andar, não para que lado. Elas servirão para o alvo e o stop do passo 5.

**Fala:**
> "Testamos 3 conjuntos: as 5 features básicas do passo 1, as 21 candidatas e as 13 pré-selecionadas pela análise de dados. O conjunto completo foi o melhor: correlação de 0,023 contra 0,018 do básico e 0,012 do pré-selecionado. Os conjuntos menores ficaram abaixo da faixa de empate."

**Por que filtrar piorou** (pergunta provável): a análise de dados avalia **cada feature sozinha**. Uma feature de volatilidade, sozinha, não diz a direção. Combinada com a de reversão, ela diz **quando** a reversão é mais forte. O RF usa essas combinações, e a pré-seleção as descartou. A análise de dados continua valendo como **explicação** do sinal.

### Slide 16 — IV.3 · Janela de treino: 12 meses (≈1,5 min)

**O que se decidiu:** quanto passado o modelo vê: 3, 6, 12, 24 meses ou todo o histórico.

**O dilema:**
- **Janela curta:** se adapta ao mercado atual, mas tem poucos dados e as estimativas ficam ruidosas.
- **Janela longa:** tem muitos dados, mas mistura épocas diferentes. O WIN foi de ~100 mil para ~130 mil pontos em 2024, e a volatilidade muda de regime.

**Fala:**
> "Com 12 meses, a correlação foi de 0,024 com t de 6, a melhor com folga. Com 3 ou 6 meses, faltam dados; com 24 meses ou todo o histórico, o modelo mistura regimes. Com todo o histórico, o sinal chega a ficar negativo."

**Análise de dados que apoiou:** a correlação mês a mês das features principais com o alvo, até 2023. A reversão de 15 minutos fica do mesmo lado do zero em 94% dos meses. O sinal é estrutural, mas os níveis de preço e de volatilidade mudam, e isso pune as janelas muito longas.

### Slide 17 — IV.3 · Retreino: semestral (≈1,5 min)

**O que se decidiu:** de quanto em quanto tempo refazer o modelo: mensal, trimestral ou semestral.

**Fala:**
> "As três frequências empataram dentro de 1 erro-padrão: 0,024 a mensal, 0,024 a trimestral e 0,022 a semestral. Pela regra, fica a mais simples de operar, a semestral, com 2 treinos por ano."

**Se perguntarem se o modelo envelhece:** antes, só com dados até 2023, treinamos o modelo uma vez e medimos a correlação em cada mês depois do treino. Ela cai cerca de 0,002 por mês e desaparece depois de uns 9 meses. Um retreino a cada 6 meses fica dentro desse prazo de validade.

**Ressalva honesta:** a curva de envelhecimento é um argumento a favor de retreinar mais vezes. O trimestral foi o melhor no número e fica anotado como alternativa para a comparação final.

### Slide 18 — IV.4 · Nº de árvores: 50 (≈1 min)

**Fala:**
> "Mais árvores só tornam a média mais estável; não acrescentam sinal e não causam sobreajuste. Fizemos o mesmo modelo crescer de 25 até 400 árvores. A partir de 50, a correlação fica a menos de 0,001 do valor com 400: 0,0216 contra 0,0220. Mais árvores só custariam tempo."

**Regra:** o menor número de árvores a menos de 0,001 da correlação com 400.

### Slide 19 — IV.4 · Profundidade × folha (≈2 min)

**O que se decidiu:** a complexidade de cada árvore.

**Análise de dados antes do modelo** (vale explicar, porque é didática): para a média de uma folha ser distinguível de zero, ela precisa de:

```
n ≥ 4 σ² / δ²
```

com σ = desvio do retorno e δ = o efeito que se quer detectar.
- Com o efeito das features mais fortes, isso dá **~540 mil candles**, mais que a janela inteira de 12 meses (~131 mil).
- Conclusão: **nenhuma folha sozinha tem média significativa.** O sinal só aparece na média de muitas folhas e muitas árvores.
- Os dados apontavam para **folhas grandes e árvores rasas**.

**Fala:**
> "Testamos 3 profundidades e 4 tamanhos de folha. O mapa mostra a correlação no teste e, embaixo, a diferença entre o treino e o teste, que mede o sobreajuste. Árvores fundas com folhas pequenas decoram o treino: com profundidade 8 e folha de 0,5%, a correlação é 0,082 no treino e só 0,019 no teste. As folhas de 5% foram as melhores em todas as profundidades. Mas o melhor ficou na borda da grade, com a maior folha e a menor profundidade. Quando isso acontece, o ótimo pode estar além, então estendemos a grade."

### Slide 20 — IV.4 · Modelo final (≈2 min)

**O que se decidiu:** quantas features cada divisão considera e o tamanho final da folha.

**Fala:**
> "Primeiro testamos quantas features cada divisão pode considerar: todas, metade ou a raiz do número de features. Considerar menos deixa as árvores mais diferentes entre si, e a média funciona melhor. Depois estendemos a grade: profundidade 2 e folhas de 10% e 20%. Várias configurações empataram dentro de 1 erro-padrão. A maior correlação foi da profundidade 2 com folha de 5%, 0,026. Pela regra, fica a mais simples entre as empatadas: profundidade 2, folha de 10%, raiz das features, com correlação de 0,024 e t de 4,1. Com folha de 20%, a árvore praticamente não divide, e a previsão fica constante."

**Modelo final:** candle de 1 min · 21 features · janela de 12 meses · retreino semestral · 50 árvores · profundidade 2 · folha de 10% · raiz das features.

**Diga com transparência:** esse modelo achata as previsões (desvio de 1,1% do retorno real), e o ganho bruto caiu para +0,26. A configuração com folha de 5% teve ganho bruto de +0,37 e fica anotada como alternativa para o passo 5. Ponte para o próximo slide.

### Slide 21 — IV.4 · Ganho acumulado do modelo final (≈1,5 min)

**O que o gráfico mostra:** os pontos acumulados ao longo de 2024 se o modelo final operasse **todos** os candles: compra quando a previsão é positiva, vende quando é negativa. Em cima, sem custo; embaixo, descontando 5 pts por operação. As escalas são diferentes: o painel de baixo vai até −800 mil. Cada semestre é previsto por um modelo treinado nos 12 meses anteriores, que nunca viu aquele período.

**Fala:**
> "Este é o modelo final operando o ano de 2024 inteiro, candle a candle. Ele acumula 36,8 mil pontos brutos em 141 mil operações, os +0,26 ponto por operação do slide anterior. Repare que quase tudo vem do segundo semestre: no primeiro, o ganho foi praticamente zero, +0,01 por operação; no segundo, depois do retreino de julho, +0,50. Isso mostra o quanto o resultado varia de um período para outro. A linha de cima sobe, então o sinal existe. Embaixo está o mesmo modelo pagando 5 pontos por operação: são cerca de 707 mil pontos de custo contra 37 mil de ganho, e o ano termina em −670 mil pontos. Cada operação perde perto de 4,7 pontos. Por isso o próximo passo é operar só as previsões mais fortes, para que o ganho por operação supere o custo."

**Treino × teste** (no quadro à direita): no próprio período de treino, os dois modelos ganhavam +0,99 e +0,68 pt/op, com correlação de 0,034. No teste, +0,26 e 0,024. É normal o modelo ir melhor nos dados que já viu. A queda de correlação é moderada, bem menor que a da árvore funda (0,082 no treino contra 0,019 no teste).

**Se perguntarem do 1º semestre fraco:** com um único ano de teste, um semestre ruim pode ser azar ou mudança de regime. A correlação foi positiva em 11 dos 12 meses, então o sinal de ordenação se manteve; o que variou foi o ganho em pontos, que depende de poucos movimentos grandes. É mais um motivo para escolher pela correlação (Parte 3).

### Fechamento da frente (≈1 min)

> "Resumindo: o Random Forest achou um sinal real, consistente mês a mês, que o ARIMA não tinha encontrado. Ele vem de reversão de curto prazo, da posição no range do dia e do dólar. Mas operar todo candle ainda rende mais de 10 vezes menos que o custo. O próximo passo é a regra de decisão: operar só quando a previsão é forte. Ela vai ser rodada para cada modelo candidato, e todos serão comparados com a mesma régua antes da validação única em 2025–26."


### Slide 29 — VI · Próximos passos (≈1,5 min)

**O que o slide mostra:** à esquerda, as regras de compra e venda que serão testadas; à direita, a ordem do trabalho que falta.

**Fala:**
> "O próximo passo é consolidar as regras de compra e venda. Primeiro, a entrada: em vez de operar todo candle, só as previsões mais fortes, por exemplo os 10% de maior previsão em valor absoluto. Depois, o lado: compra e venda, ou só um deles. E a saída: um alvo de lucro, o take profit, e um stop loss, os dois medidos em múltiplos da volatilidade recente, 1 ou 1,5 vez. Também vamos testar sair simplesmente no fechamento do próximo candle e filtros de horário e de volatilidade. Essas regras vão ser rodadas para cada modelo candidato, o RF final, as alternativas anotadas, a melhor rede neural e o ARIMAX, e todos serão comparados com a mesma régua: pontos por operação depois do custo, custo de equilíbrio, número de operações e comparação com sorteios aleatórios de direção. O vencedor é congelado e roda uma única vez na validação de 2025–26."

**Como o alvo e o stop serão escolhidos:** depois de cada sinal, mede-se quanto o preço chegou a andar a favor (MFE) e contra (MAE) nos candles seguintes. Se o preço costuma andar 1,5× a volatilidade a favor antes de andar 1× contra, essa combinação de alvo e stop tem esperança positiva. Medir isso em múltiplos da volatilidade faz o alvo e o stop se ajustarem sozinhos a dias calmos e agitados.

**Se perguntarem do sobreajuste:** este é o passo mais fácil de sobreajustar, porque cada combinação de regras é mais uma tentativa no mesmo ano de 2024. Por isso as opções são poucas, o número de combinações testadas fica registrado e cada regra precisa de pelo menos ~100 operações em 2024.

---

## Parte 3 — Por que a escolha final segue a correlação, e não o ganho bruto

Este é o ponto que a banca provavelmente vai questionar: "se o objetivo é dinheiro, por que não escolher o modelo de maior ganho?". São cinco argumentos, em ordem de importância.

### 1. O ganho bruto mede uma regra que não vai ser usada

O ganho bruto opera **todos** os 141 mil candles de 2024. Ninguém vai operar assim: com 5 pts de custo, operar tudo perde ~4,6 pts por operação em qualquer modelo.

A regra real, do passo 5, vai operar **só os candles com as previsões mais fortes**. O que importa para ela é **o modelo ordenar bem**, para que os extremos dele concentrem os melhores movimentos. É exatamente o que a correlação de postos mede. Escolher o modelo pelo ganho de uma regra descartável é otimizar a coisa errada.

### 2. O ganho bruto é muito mais ruidoso que a correlação

- **Precisão:** o erro-padrão do ganho bruto é ±0,11 pt, num ganho de 0,26 a 0,51. A razão sinal/ruído fica em torno de 2,5 a 4,5. A correlação tem t de 4 a 6, e com o EP **entre meses**, que já considera a volatilidade em ondas.
- **Caudas pesadas:** o ganho em pontos é dominado por poucos candles enormes (notícias, aberturas). Um modelo pode "ganhar" num minuto de notícia por acaso e parecer melhor. A correlação de postos dá o mesmo peso a cada candle e não depende desses poucos eventos.
- **Diferenças dentro do ruído:** entre as opções de um mesmo passo, os ganhos brutos diferem em 0,1 a 0,2 pt, da ordem de 1 a 2 erros-padrão. Ordenar opções por esse número seria, em boa parte, ordenar ruído.

### 3. Escolher pelo lucro na amostra de teste já falhou: o caso ARIMAX

Na frente estatística, o ARIMAX foi escolhido entre 180 combinações **pela maior esperança de ganho** na seleção: +10 ± 19 pts. A vantagem era menor que o próprio erro-padrão. Na validação, deu **−41 ± 42 pts**, igual a um sinal aleatório (percentil 19 entre 2.000 sorteios).

É a **maldição do vencedor**: entre muitas opções ruidosas, a de maior número quase sempre é a que teve mais sorte. Usar uma métrica mais estável (correlação com t robusto) e desempatar pela simplicidade é a defesa contra isso.

### 4. Ganho e correlação andam juntos, mas a correlação é a parte estável

Pela aproximação `ganho ≈ ρ × σ × 0,8`, com o candle fixo, o ganho esperado é **proporcional à correlação**. Maximizar a correlação é maximizar o ganho esperado, só que estimando-o pela parte que tem menos ruído. A volatilidade (σ) muda de um ano para o outro e não depende do modelo. A capacidade de ordenar (ρ) é o que o modelo controla.

### 5. Por que o ganho caiu no modelo final, e o que isso significa

- O modelo final usa **folhas maiores**, que **achatam** as previsões. A ordem se mantém (a correlação de 0,024 está empatada com a melhor), mas a previsão fica mais conservadora nos candles extremos, e é neles que o ganho bruto em pontos se decide.
- A diferença de ganho entre a folha de 10% (+0,26) e a de 5% (+0,37) é de cerca de 1 erro-padrão. Não é uma diferença confiável.
- **Não descartamos essa informação:** a folha de 5% está registrada como alternativa. No passo 5, as duas configurações passam pela mesma regra de decisão, e aí sim o critério passa a ser dinheiro: esperança por operação após o custo, com erro-padrão, número de operações e comparação com o sorteio aleatório nos mesmos candles.

**Frase-resumo para a banca:**
> "Nos passos de modelagem, escolhemos pela métrica que mede a capacidade de ordenar e que é estável mês a mês, a correlação. O dinheiro é o critério da etapa de regras, porque só ali sabemos quais candles serão operados. Escolher pelo lucro antes disso foi o que fez o ARIMAX parecer bom na seleção e perder 41 pontos por operação na validação."

---

## Parte 4 — Perguntas prováveis

**"51% de acerto não é praticamente uma moeda?"**
Para cada candle, sim. O ponto é a consistência: com 141 mil candles e correlação positiva em 11 ou 12 dos 12 meses, essa vantagem pequena é real. O desafio é transformá-la em dinheiro, operando só onde ela é maior. Isso é o passo 5.

**"Por que 1 minuto, se o custo pesa 12% do movimento?"**
Porque foi o único tamanho com sinal. Nos candles maiores, o custo pesa menos, mas a correlação ficou em zero. Custo baixo sem sinal não gera lucro.

**"Por que usar 21 features se a análise de dados indicava menos?"**
A análise olha cada feature sozinha. O RF usa combinações: a volatilidade, por exemplo, não indica direção, mas indica quando a reversão é mais forte. No teste, o conjunto pré-selecionado ficou abaixo do completo, e a regra, definida antes de rodar, escolheu as 21.

**"O RF não está sobreajustando com 21 features?"**
O modelo final é extremamente regularizado: profundidade 2 e folhas com 10% do treino (~13 mil candles cada). A diferença entre a correlação no treino (0,034) e no teste (0,024) é pequena. O sobreajuste aparece nas árvores fundas, com 0,082 no treino contra 0,019 no teste, e por isso elas foram descartadas.

**"Por que Spearman e não R² ou erro quadrático?"**
O RF encolhe as previsões para perto de zero, e métricas de escala não distinguem isso de "prever zero" (Theil U ≈ 1 em todos os casos). Além disso, os retornos têm caudas pesadas, e a regra de operação usa a ordem das previsões. A correlação de postos mede exatamente a ordem.

**"Por que o t mensal e não o t comum?"**
Os candles não são independentes: a volatilidade vem em ondas. Com n = 141 mil, o erro-padrão comum daria significância a quase tudo. Tratar cada mês como uma observação é conservador e responde se o sinal se repete ao longo do ano.

**"Os números de passos diferentes são comparáveis?"**
Dentro de cada passo, sim: só muda a decisão em teste. Entre passos, aproximadamente: nos passos 3 e 4, o RF ficou mais leve (100 ou 50 árvores, até 50 mil candles por árvore) para caber no tempo de processamento. O slide de evolução avisa isso.

**"A validação de 2025–26 foi usada em algum momento?"**
Não. Todas as escolhas usam dados até 2024. A validação será rodada uma única vez, com o modelo e a regra congelados por escrito.

**"Então o modelo falhou?"**
Não ainda. Ele encontrou um sinal que o ARIMA não encontrou, e isso é um resultado. Se esse sinal paga o custo, só a regra de decisão (operar só os extremos) e a validação vão dizer. Se não pagar, a conclusão honesta é que o sinal existe, mas é pequeno demais para o custo de operar no varejo.
