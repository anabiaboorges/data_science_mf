# Plano de ação — Random Forest T+1 (WIN)

Atualizado em 23/11/2026 · Entrega: 18/11/2026

O objetivo é encontrar o melhor cenário de Random Forest para prever o retorno do próximo candle do WIN. Os parâmetros são decididos um grupo por vez, em passos, e o modelo final é validado uma única vez em 01/2025 → 03/2026.

**Entrega de 18/11:** os modelos até o passo 4, ou seja, candle, features, janela de treino e retreino, e hiperparâmetros. A regra de decisão (passo 5) foi concluída em 23/11.

Notebook: `notebooks/rf_t1_win.ipynb`, com uma seção por passo. Resultados salvos em `resultados/`.

## Status

| Passo | O que decide | Status |
|---|---|---|
| 1. Candles | horizonte da previsão (1, 5, 15, 30, 60 min) | Concluído |
| 2. Features | o que o modelo enxerga (21 candidatas) | Concluído |
| 3. Janela + retreino | quanto passado e com que frequência | Concluído |
| 4. Hiperparâmetros | complexidade do RF | Concluído |
| 5. Regras de decisão | quando comprar e vender, TP e stop | Concluído: nenhuma regra aprovada |
| 6. Comparação entre modelos | rodar o passo 5 para cada modelo e compará-los | Próximo passo |
| 7. Validação final | modelo congelado em 01/2025 → 03/2026 | RF concluído (rodada única, 23/11): sinal confirmado, operação com prejuízo |

## Protocolo fixo

Estas regras não mudam entre os passos; sem elas, os resultados deixam de ser comparáveis.

| Item | Definição |
|---|---|
| Alvo | 100 · ln(C_{t+1} / C_t), só entre candles do mesmo pregão |
| Dados | WIN$N 1 min, pregão regular 9:00–18:24, 15/10/2020 → 03/2026 (buraco jul–set/2023); WDO$N 1 min para as features do dólar |
| Análise exploratória | até 31/12/2023 (só dados de treino) |
| Seleção | 2024, walk-forward |
| Validação final | 01/2025 → 03/2026, rodada uma vez com o modelo congelado |
| Custo | 5 pts por operação completa, em todas as etapas |
| Métrica principal | correlação de Spearman entre previsão e retorno real (média mensal de 2024) |
| Significância | t robusto = média das correlações mensais ÷ erro-padrão entre os 12 meses |
| Filtro | Theil U ≤ 1,001 (até 0,1% pior que prever zero conta como empate) |
| Desempate | entre opções a até 1 erro-padrão da melhor, fica a mais simples |

## Modelo entregue

| Escolha | Valor | Passo |
|---|---|---|
| Candle | 1 min | 1 |
| Features | 21 (retornos recentes e acumulados, posição no dia, volatilidade, forma do candle, volume relativo, gap, dólar) | 2 |
| Janela de treino | 12 meses | 3 |
| Retreino | semestral | 3 |
| Hiperparâmetros | 50 árvores, profundidade 2, folha de 10% do treino, `max_features = sqrt` | 4 |
| Desempenho em 2024 | Spearman +0,024 ± 0,006 · t robusto +4,1 · positivo em 11 de 12 meses · Theil U 0,99998 | 4 |

O sinal é estatisticamente real, mas ainda não é econômico. Operando todo candle, o ganho bruto é de cerca de 0,4 pt por operação, contra 5 pts de custo. A correlação necessária para pagar o custo no candle de 1 min é ~0,11. Cabe ao passo 5 operar só nos candles de previsão mais forte.

## Passo 1 — Candles (concluído)

- [x] Estatísticas por candle: desvio em % e em pontos, custo ÷ movimento médio
- [x] ρ₁ de cada candle com faixa de ruído robusta
- [x] Correlação necessária para pagar o custo: ρ = custo ÷ (σ em pontos × 0,80)
- [x] RF fixo em 2024 (janela de 12 meses, retreino trimestral) nos 5 candles
- [x] Robustez com janelas de 6 e 24 meses

**Resultado:** **1 min** é o único candle com sinal significativo (Spearman +0,016, t robusto +4,9, positivo nos 12 meses). Nos outros (5, 15, 30, 60 min), o t robusto ficou entre −0,5 e +0,3. Nenhum candle tem memória linear: o ρ₁ fica dentro do ruído em todos. A posição do 1 min no ranking muda conforme a janela; a confirmação fica para a comparação final.

## Passo 2 — Features (concluído)

- [x] 21 candidatas em 7 grupos, só com informação até t
- [x] Até 2023: Spearman com t mensal, estabilidade nos anos 2021–2023, informação mútua em decis, relação com o tamanho do movimento
- [x] Remoção das redundantes (|ρ| > 0,8)
- [x] Confirmação em 2024: comparação de conjuntos, importância por permutação, retirada uma a uma

**Resultado:** ficaram as **21 candidatas** (Spearman +0,022, t +5,3), contra +0,016 do conjunto básico. A pré-seleção feita só pela análise de dados piorou o modelo (+0,009), porque avalia cada feature sozinha e o RF usa as combinações.

A análise mostrou de onde vem o sinal:
- **reversão em 15–60 min:** `ret_acum_15c` com t = −8,9;
- **posição no range do dia:** t = −7,1;
- **dólar dos últimos 15 min:** t = +4,5.

As três têm o mesmo sinal em 2021, 2022 e 2023. As features de volatilidade informam o tamanho do próximo movimento, não a direção.

## Passo 3 — Janela de treino e retreino (concluído)

- [x] Correlação mês a mês das principais features e regimes de preço e volatilidade
- [x] Curva de envelhecimento do modelo (5 origens em 2022–2023)
- [x] Walk-forward em 2024: 5 janelas × retreino trimestral, mais mensal e semestral nas 2 melhores

**Resultado:** **12 meses com retreino semestral** (Spearman +0,022, t +4,8), empatada dentro de 1 erro-padrão com a trimestral (+0,024, a melhor no número). Janelas de 24 meses e expansiva pioram, porque misturam regimes. A curva de envelhecimento cai cerca de 0,002 por mês (t = −2,5), e o sinal some depois de ~9 meses. O retreino trimestral fica anotado como alternativa.

## Passo 4 — Hiperparâmetros (concluído)

- [x] Análise de dados: tamanho mínimo de folha para uma média significativa (n ≥ 4σ²/δ²)
- [x] Curva do nº de árvores (25 → 400)
- [x] Grade profundidade (3, 5, 8) × % por folha (0,5%–5%), depois `max_features` nas 2 melhores
- [x] Extensão da grade quando o ótimo cai na borda (folhas de 10% e 20%, profundidade 2)

**Resultado:** **50 árvores, profundidade 2, folha de 10%, `max_features = sqrt`** (Spearman +0,024, t +4,1).

- **Folha mínima:** pela análise de dados, uma folha precisaria de ~540 mil candles para ter média significativa, contra ~131 mil numa janela de 12 meses. Nenhuma folha isolada é significativa, e o sinal vem da média de muitas folhas e árvores. Os dados já apontavam folhas grandes e árvores rasas, e a grade confirmou.
- **Sobreajuste:** árvores fundas com folhas pequenas decoram o treino (profundidade 8, folha de 0,5%: Spearman 0,082 no treino e 0,019 na validação).
- **Escolha:** a melhor configuração no número foi profundidade 2 com folha de 5% (+0,026). A regra ficou com a mais simples entre as empatadas (folha de 10%). Folhas de 20% deixam a previsão constante.
- **Alternativa anotada:** a escolhida comprime muito as previsões (desvio de 1,1% do retorno real) e tem ganho bruto de 0,26 pt/op, contra ~0,4 das configurações com folha de 5%. No passo 5, que opera os candles de previsão mais forte, vale testar também a folha de 5%.

## Passo 5 — Regras de decisão (concluído)

O desenho mudou em relação ao previsto aqui: desenvolvimento em 2022–2023 (walk-forward), seleção em 2024 com só 3 finalistas, e saída por tempo de 30/60/120 min, porque sair no próximo candle rende menos que o custo. 2025 em diante não foi usado.

- [x] Grade de 72 combinações em 2022–2023: entrada × lado × tempo (54), TP/stop (16 novas), filtro (2 novas)
- [x] 3 finalistas em 2024, com custo de 5 pts
- [x] Comparação com 2.000 sorteios de direção, comprar/vender sempre e mesmo horário em outro pregão
- [x] Contagem de testes: 72 na grade, 3 finalistas e 24 exploratórias vistas antes em 2024

**Resultado: não operar.** As regras foram bem em 2022–2023 e não se repetiram em 2024. As três usam a folha de 5% e os dois lados.

| Finalista | Líquido/pregão 2022–23 | Líquido/pregão 2024 | Líquido/op 2024 | Percentil contra sorteios | Meses positivos |
|---|---|---|---|---|---|
| corte 5%, 120 min, TP 1,5× | +112 ± 46 | −33 ± 39 | −8,6 | 45 | 5 |
| corte 5%, 120 min, stop 1,5× (melhor) | +124 ± 60 | +7 ± 45 | +1,8 | 57 | 7 |
| corte 10%, 60 min, TP 1,5× | +105 ± 41 | −51 ± 67 | −6,4 | 25 | 5 |

- A melhor regra rende +6,8 pts brutos e +1,8 líquido por operação com o custo de 5 pts, dentro do ruído e abaixo de "vender sempre" nas mesmas entradas (+15,0 bruto, +10,0 líquido).
- O stop de 1,5× melhorou pouco em 2024 (+7,2 contra +3,8 pts por pregão sem stop). O TP piorou as duas regras em que foi usado.
- A regra fica em `passo5_regra.json` só como registro. Detalhes: `passo5_grade.csv`, `passo5_finalistas_2024.csv`.

## Passo 7 — Validação do RF (rodada única, 01/2025 → 03/2026)

Regra congelada no passo 5 (folha de 5%, corte de 5%, 120 min, stop 1,5×, sem TP), custo de 5 pts. Rodada antes do passo 6, por decisão da equipe. A regra já não tinha sido aprovada em 2024.

- **Previsão:** Spearman mensal +0,029 (t robusto +5,8), Theil U 0,99995. O sinal se repetiu fora da amostra.
- **Operação:** 1.254 operações, −23,3 pts brutos e **−28,3 líquidos por operação**; −114 ± 75 pts por pregão; 4 de 15 meses positivos; percentil 11 contra sorteios. Total de −35.530 pts.
- Este período está gasto para o RF: novas regras só podem ser testadas em dados que não foram usados (reserva a partir de 04/2026).

## Redesenho do RF — alvo no horizonte da operação (23/11/2026)

Feito depois do prejuízo na validação, com escolha só em 2022–2024. 12 modelos e 48 regras (notebook `redesenho_rf.ipynb`), custo de 5 pts e critérios do passo 5.

**Resultado: nenhuma regra aprovada.** Só 4 das 48 foram elegíveis em 2022–2023. Em 2024, os 3 finalistas renderam +0,7, +0,4 e +1,0 pt por operação (percentis 68, 68 e 75; 5, 4 e 5 meses positivos). O terceiro operou só na venda. A reserva (04/2026 em diante) não foi aberta.

Total de regras simuladas pelo RF até aqui: 72 + 48, além de 24 combinações exploratórias.

## Caudas extremas da previsão (23/11/2026)

Sugestão do professor. RF T+1, caudas de 2%, 1% e 0,5% × 30/60/120 min × sem stop/stop 1,5× (36 regras, notebook `caudas_rf.ipynb`), mesma escolha em 2022–2023 e 3 finalistas em 2024. **Nenhuma aprovada:** −23,9, −7,4 e −4,9 pts líquidos por operação em 2024 (percentis 17, 48 e 58). Reserva não aberta. Total de regras simuladas: 72 + 48 + 36, além de 24 exploratórias.

## Stop mais rigoroso (23/11/2026)

Entrada congelada, 6 stops × 3 limites diários (18 regras, notebook `stop_rf.ipynb`), escolha em 2022–2023. Venceu de novo o stop de 1,5× sem limite diário, que é a regra do passo 5 (2024: +1,8 pt por operação, percentil 57). Stops de 0,25× e 0,5× pioraram. Reserva não aberta. Das 18 regras, 3 repetem configurações já testadas sem limite diário (sem stop, 1× e 1,5×), e "parar após stop" sem stop equivale a "livre".

**Teste extra (uma regra em 2024):** stop 1× e parar o dia após um stop. 565 operações, −17,5 pts líquidos por operação, 3 meses positivos, percentil 48. **Não aprovado.** Busca por regras encerrada.

## Diagnóstico das 72 regras (23/11/2026, por curiosidade)

Feito depois da validação, sem escolher regra (`diagnostico_regras.ipynb`, `passo5_diagnostico_todas.csv`). Das 45 que passaram em 2022–2023, 12 foram positivas em 2024, 12 em 2025–26 e 2 nos dois períodos. Correlação de ordem (líquido por pregão) de 2022–23 com 2024: +0,22; com 2025–26: −0,30. Percentil ≥ 95 contra sorteios: 1 em 2024, 1 em 2025–26, nenhuma nos dois.

## Próximos passos

1. **Rodar a etapa de regras de decisão para cada modelo.** O mesmo passo 5, com as mesmas opções e o mesmo critério, aplicado a cada modelo candidato. Isso inclui o RF entregue, as alternativas anotadas (retreino trimestral, 5 min) a folha de 5% do passo 4 e os modelos da frente estatística.
2. **Comparar os modelos com uma régua só:**
   - esperança por operação ± erro-padrão;
   - custo de equilíbrio;
   - nº de operações;
   - percentil contra sinais aleatórios nos mesmos candles;
   - Spearman e Theil U.
3. **Confirmação única:** refazer candle × janela com as features e os hiperparâmetros finais, sem ciclo.
4. **Congelar e validar uma vez em 01/2025 → 03/2026:** reportar Theil U, Spearman, acerto de direção, nº de operações, esperança ± erro-padrão, resultado acumulado, custo de equilíbrio e comparação com o naive e com o ARIMAX.

**Fora do escopo por agora:**
- boosting (XGBoost, LightGBM);
- alvo normalizado pela volatilidade, ou classificação "move mais que o custo";
- RF prevendo a volatilidade, contra o HAR;
- tamanho de posição pela volatilidade prevista;
- segundo teste com o WIN de 04/2026 → 09/2026.
