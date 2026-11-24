# Data Science: previsão do mini-índice (WIN), T+1

Projeto UFG. Cópia completa da pasta `data_science_v2` em 23/11/2026. Custo usado em todo o projeto: **5 pontos por operação completa**.

## Conclusões da frente de ML
1. **Há um sinal estatístico pequeno.** RF de 1 min, 2025–26 (com folha de 5%): correlação +0,029 (t 5,8), Theil U 0,99995. Seria preciso cerca de 0,11 para cobrir o custo (estimativa).
2. **A regra de compra/venda não deu lucro** (folha 5%, corte 5%, ambos os lados, 120 min, stop 1,5×, sem TP):

| Período | Pontos por pregão | Leitura |
|---|---|---|
| 2022–23 (grade) | +124,3 ± 59,8 | período em que foi escolhida |
| 2024 (seleção) | +7,2 ± 44,5 | percentil 57 contra sorteios, 7/12 meses: reprovada |
| 01/2025–03/2026 (validação) | −114,2 ± 74,5 | −28,3 por operação, percentil 11, 4/15 meses |

3. **Outras tentativas não mudaram isso:** alvo em 30–120 min (48 regras), caudas (36) e stop (18). Total de 170 regras distintas, mais 24 combinações exploratórias; nenhuma aprovada.
4. **Diagnóstico posterior:** das 45 regras lucrativas em 2022 e em 2023, 12 foram positivas em 2024, 12 em 2025–26 e só 2 nos dois.
5. **RF de 60 min (v1):** +14,1 → +3,0 → −4,8 pontos por operação (2024, 01/2025–03/2026, 04–09/2026). Escolher por lucro não se repetiu; por Theil U, sim (concordância de ranking −0,24 contra 0,99).
6. **Decisão: não operar.**

**Ressalvas:** 04–09/2026 já foi consultado por outras abordagens; não é reserva intacta. A v2 não foi reproduzida. No modelo final pesa mais `pos_fech_candle`, instável até 2023 e compatível com *bid-ask bounce*.

## Onde está
| Item | Caminho |
|---|---|
| Slides | `apresentacao/slides.html` (ML: 13–26) |
| **Roteiro atual de ML** | `apresentacao/roteiro_completo_ml.md` |
| Linha do tempo visual | `apresentacao/linha_do_tempo_ml.html` |
| Histórico e decisões | `plano_de_acao.md` |
| Notebooks | `notebooks/` (principal: `rf_t1_win.ipynb`) |
| Resultados | `resultados/`, `resultados_ml/`, `rf_do_zero/` |
| Outra tentativa (v1) | `ml/`, `tt1_comum.py` |
| Dados brutos | `WIN$N_M1.csv`, `WDO$N_M1.csv` |

Use só `roteiro_completo_ml.md`; os roteiros antigos estão desatualizados.

## Pendências
- Corrigir slides com a versão antiga: "Divisão dos dados", "Onde estamos", "Próximos passos", "Objetivo" e "dados nunca vistos" (redes neurais).
- Corrigir `artigo_t1.tex` e `plano_de_acao.md` ("rodada única", "reserva não aberta").

## Abrir em outro computador
```
git lfs install
git clone https://github.com/anabiaboorges/data_science_mf.git
```
Use `git clone`, não "Download ZIP": `resultados/passo5_previsoes.csv` (160 MB) está no Git LFS.
