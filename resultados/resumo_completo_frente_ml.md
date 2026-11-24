# Resumo completo da frente de ML (Random Forest no WIN)

> Responsável: Ana Beatriz · Atualizado em 23/11/2026 · Detalhes da primeira rodada: [`relatorio.md`](relatorio.md)
>
> Salvo indicação, o custo é de **5 pts por operação** (ida e volta) e os valores em R$ são para **1 contrato de WIN** (R$ 0,20 por ponto).

## 1. Conclusão

**A Random Forest encontra um sinal pequeno, mas nenhuma configuração testada deu lucro de forma consistente depois do custo.**

- **No candle de 1 min o sinal existe, mas é pequeno demais.** O modelo original bate o naive no teste (Clark-West t = 3,72) e ganha +0,59 pt por operação antes do custo de 5 pts.
- **No candle de 60 min o lucro de 2024 não se repetiu.** A configuração congelada fez +14,1 pts/op em 2024, +3,0 no teste e −4,8 no período extra. Em 2025-2026 dá **+0,7 pt/op (R$ 441, t = 0,08)**, um empate com o custo.
- **O campeão de 2024 foi o que mais perdeu em 2025-2026** entre os seis cenários comparados: de +R$ 4.782 para −R$ 5.598.
- **Nenhum cenário com lucro na validação chegou a t ≥ 2.**

## 2. Problema e protocolo

O objetivo é prever o retorno do próximo candle, T(t+1), do mini-índice (WIN) e verificar se a previsão paga o custo de operar.

| Período | Datas | Uso |
|---|---|---|
| Treino | Janela deslizante anterior a cada re-treino | Ajustar o modelo |
| Validação | 2024 | Todas as escolhas |
| Teste | 01/2025 a 03/2026 | Prova fora da amostra |
| Extra | 04/2026 a 09/2026 (só WIN) | Segunda prova fora da amostra |

- **Walk-forward:** o modelo é re-treinado periodicamente só com o passado e prevê o bloco seguinte.
- **Sem posição de um dia para o outro:** o último candle do pregão não tem alvo.
- **Medidas de previsão:** R² fora da amostra, Theil U (menor que 1 é melhor que prever zero) e Clark-West.
- **Medidas de dinheiro:** ganho médio por operação, t, trimestres positivos, drawdown e comparação com direção sorteada.

## 3. Linha do tempo

| Etapa | Modelo | Critério de escolha | Resultado |
|---|---|---|---|
| 1. Frente original (02–03/11) | Regressor | Significância (t ≥ 2); senão, Theil U | 1 min × 12 m; teste −4,41 pts/op |
| 2. Otimização (11/11) | Classificador de 3 classes | Candidata com t ≥ 2; senão, t do ganho bruto | 1 min × 12 m; teste −4,82 pts/op |
| 3. RF do zero, v1 (12/11) | Regressor, 60 min | Lucro na validação | Validação +14,1; teste +3,0; extra −4,8 |
| 4. RF do zero, v2 (12/11) | Regressor | Fase A só R²; Fase B custo só na regra | Incompleta (passos 5 a 7 pendentes) |
| 5. Comparação final (23/11) | Melhores de 2024 | Nenhuma escolha nova | Nenhum se sustentou em 2025-2026 |

## 4. Etapa 1: frente original

**O que foi feito:** grade de 3 candles (1, 15 e 60 min) × 4 janelas (1, 3, 6 e 12 meses), 36 regras de operação e 8 variantes do modelo, com 16 variáveis (10 lags, amplitude, corpo, volume, volatilidade, hora e posição no pregão).

**Ajustes em relação ao plano comum:**

| Ajuste | Motivo |
|---|---|
| Folha mínima proporcional ao treino | Com folha fixa de 100, 60 min × 1 mês gerava só 4 previsões diferentes |
| Escolha exige t ≥ 2 | A vencedora por esperança (+17,8 pts/op) deu −31,3 no teste |
| Teste de Clark-West | Dizer se um Theil U de 0,9999 é sinal ou acaso |

**Escolhas:** 1 min × 12 meses, 100 árvores, profundidade 4, folha 2%, opera sempre, sem take profit nem stop. Nenhuma combinação ou regra teve lucro com t ≥ 2. Árvores mais simples foram melhores.

**Resultado no teste (174.579 operações):**

| Medida | Valor |
|---|---|
| Theil U / R² | 0,9999 / +0,0002 |
| Acerto de direção | 51,4% |
| Clark-West | t = 3,72 |
| Sem custo | **+0,59 pt/op** (t 3,85), +R$ 20.587 |
| Custo de 5 pts | **−4,41 pts/op**, −R$ 153.992 |
| Direção aleatória, custo de 5 pts | −4,83 pts/op |

**Investigações complementares (custo de 5 pts):**

| Ideia | Resultado |
|---|---|
| Só previsões fortes (limiar de 90%), 1 min | Bruto sobe para +1,99, mas fica em −3,01 no teste |
| Candles maiores com limiar de 90%, abr–set/2026 | 15 min: −22,2; 60 min: −109,5 |
| Segurar a posição até o sinal virar (90%) | Validação −2,50; teste −0,73; extra +1,93 (t 0,6). Melhor uso do sinal, sem evidência de lucro |
| Prever 10 ou 20 minutos à frente | Todas as 24 combinações negativas |

**Hipótese não testada:** o sinal de 1 min pode vir do *bid-ask bounce*, pois se concentra no último candle e é menor que o custo. Confirmar exigiria dados de book.

## 5. Etapa 2: otimização com classificador

**O que foi feito:** cada frente passou a buscar o maior retorno do próprio modelo. Três abordagens foram comparadas em 1 min × 12 meses, na validação:

| Abordagem | Ganho bruto (pts/op) | t |
|---|---|---|
| A. Regressor | +0,29 | 2,6 |
| **B. Classificador (sobe / neutro / desce), sinal = P(sobe) − P(desce)** | **+0,40** | **3,6** |
| C. Classificador com valor esperado | +0,25 | 2,2 |

A abordagem B seguiu por candle × janela, regra de saída, variáveis e hiperparâmetros. **Nenhuma configuração foi candidata** (esperança > 0, t ≥ 2 e 3 de 4 trimestres), então a decisão ficou pelo t do ganho bruto.

**Modelo final:** 1 min × 12 meses, 30 variáveis, 100 árvores, profundidade 4, folha 2%, corte de classe de 5 pts. Na validação, ganho bruto de +0,47 (t 4,2).

| Período | Opera sempre: bruto | Opera sempre: líquido | 2% mais fortes: líquido |
|---|---|---|---|
| Validação 2024 | +0,47 | −4,53 | −2,67 |
| Teste | +0,18 (t 1,2) | **−4,82** | −3,17 |
| Extra | +0,94 (t 3,2) | −4,06 | −1,00 |

No teste, o ganho bruto ficou abaixo do regressor original (+0,18 contra +0,59).

## 6. Etapa 3: RF do zero, versão 1 (critério lucro)

**O que foi feito:** pipeline refeito passo a passo, com 5 candles, janelas de 3 meses a todo o histórico e três frequências de re-treino.

| Passo | Principal conclusão |
|---|---|
| Alvo | Retorno em %: mais estável no tempo que pontos |
| Variáveis | 5 lags + forma, volatilidade, volume e contexto do dia (**15 variáveis**) |
| Candle, janela, re-treino | 60 min × 36 meses × trimestral. Janela longa ajuda; frequência quase não importa |
| Regra | Opera se o movimento previsto > 5 pts |
| Hiperparâmetros | 75 combinações, todas positivas em 2024 (+3,8 a +16,1 pts/op); escolha pela média da vizinhança |

Melhor ganho operando sempre, por candle (validação):

| Candle | Janela / re-treino | Bruto (pts/op) | Líquido (pts/op) |
|---|---|---|---|
| 1 min | 12 m / trimestral | +0,27 | −4,73 |
| 5 min | 24 m / semestral | +1,16 | −3,84 |
| 15 min | 6 m / mensal | +2,81 | −2,19 |
| 30 min | todo o histórico / trimestral | +6,31 | +1,31 |
| **60 min** | **36 m / trimestral** | **+12,04** | **+7,04** |

**Modelo congelado:** 300 árvores, profundidade 4, folha 1%, max_features 0,5.

| | Validação 2024 | Teste | Extra |
|---|---|---|---|
| Operações | 1.532 | 2.114 | 851 |
| Líquido (pts/op) | **+14,1** | **+3,0** | **−4,8** |
| t | 1,76 | 0,29 | −0,25 |
| Trimestres positivos | 4/4 | 2/5 | 1/2 |
| Lucro | R$ 4.307 | R$ 1.260 | −R$ 819 |
| Drawdown | R$ 1.190 | R$ 4.262 | R$ 2.905 |

## 7. Etapa 4: RF do zero, versão 2 (critério R²)

**Motivo:** na v1 o custo e o lucro entraram na escolha do modelo. Na v2, a **Fase A** escolhe só por R² e a **Fase B** usa o custo apenas para ajustar a regra.

| Passo | Estado | Resultado |
|---|---|---|
| 3. Variáveis | Feito | 10 lags + 4 grupos = 20 variáveis. R² médio sobe de −0,095% para +0,051%. No 60 min, o volume relativo é o mais importante |
| 4. Candle × janela × re-treino | 76 de 78 combinações | Provisório: 60 min × todo o histórico × semestral, R² +0,49%, 3/4 trimestres |
| 5. Hiperparâmetros | Pronto, **não rodado** | – |
| 6. Regra com custo | Pronto, **não rodado** | – |
| 7. Comparação de candles | Pronto, **não rodado** | – |

Faltam no passo 4: 1 min × 36 meses e 1 min × todo o histórico. O maior R² de cada candle foi 60 min +0,49%, 15 min +0,05%, 5 min +0,01%, 1 min −0,01% e 30 min −0,09%. **Só o 60 min tem R² claramente positivo.**

## 8. Melhores cenários de 2024 aplicados em 2025-2026

Os cenários foram aplicados **sem reajuste**, com limiares congelados em 2024. Cada célula traz pts/op líquido e lucro.

| Cenário | Validação 2024 | Teste | Extra | **Total 2025-2026** | t | p vs aleatório |
|---|---|---|---|---|---|---|
| RF 60 min × 36 m, maior lucro (prof. 6, folha 1%, mf 0,5) | +14,7 / R$ 4.782 | −3,5 / −R$ 1.491 | −23,7 / −R$ 4.107 | **−9,2 / −R$ 5.598** | −1,02 | 0,68 |
| RF 60 min × 36 m, maior pts/op das 75 (prof. 3, folha 0,5%, mf 0,33) | +16,1 / R$ 4.316 | +5,1 / R$ 2.092 | −22,7 / −R$ 3.781 | **−2,9 / −R$ 1.689** | −0,31 | 0,41 |
| **RF 60 min × 36 m, congelada (prof. 4, folha 1%, mf 0,5)** | +14,1 / R$ 4.307 | +3,0 / R$ 1.260 | −4,8 / −R$ 819 | **+0,7 / R$ 441** | 0,08 | 0,28 |
| Classificador 15 min × 1 m, segurar, 10% mais fortes | +75,8 / R$ 2.789 | −29,3 / −R$ 1.784 | +32,1 / R$ 687 | **−13,3 / −R$ 1.097** | −0,30 | 0,58 |
| Classificador 60 min × 3 m, segurar, 2% mais fortes | +157,6 / R$ 1.292 | −4,8 / −R$ 31 | +238,3 / R$ 715 | **+72,8 / R$ 684** | 0,86 | 0,21 |
| Classificador 1 min × 12 m, corte 10, segurar, 2% mais fortes | +5,1 / R$ 145 | −70,4 / −R$ 3.620 | +10,1 / R$ 2.621 | **−3,2 / −R$ 999** | −0,26 | 0,46 |

*p vs aleatório: fração de 1.000 sorteios de direção, nas mesmas operações, com resultado igual ou melhor que o do modelo.*

- **O classificador 60 min × 3 m fez só 47 operações em 21 meses.** Quase todo o ganho vem de 15 delas, no extra.
- **A configuração recomendada para apresentar é a congelada**, por ter sido definida antes do teste, e não pelo resultado desta tabela.

## 9. O que aprendemos

1. **Prever melhor que o naive não basta.** Em 1 min o sinal é significativo, mas muito menor que o custo.
2. **O lucro da validação engana.** Na grade original, a ordem por Theil U se repetiu no teste (Spearman 0,99) e a ordem por lucro, não (−0,24).
3. **Modelos simples vão melhor** quando o sinal é fraco: árvores rasas e folhas grandes.
4. **Mais histórico ajuda** na qualidade da previsão; a frequência de re-treino pesa pouco.
5. **Poucas operações geram números enganosos.** Os maiores pts/op vêm de amostras minúsculas.

## 10. Limitações

- **Teste e extra já foram vistos.** A seção 8 compara cenários, mas não pode servir para escolher um novo vencedor.
- **Muitas combinações no mesmo ano** (2024) aumentam a chance de achar algo bom por sorte.
- **Custo de 5 pts e margem de R$ 150 por contrato** são estimativas ainda não confirmadas.
- **Dados:** faltam jun–out/2023, e o WDO termina em 01/04/2026. No extra, os classificadores usam a base só de WIN; as variáveis deles não usam WDO.
- **Limiar instável:** o classificador de 1 min foi de 257 operações no teste para 1.303 no extra com o mesmo limiar.
- **Correção pendente:** notebooks e slides da v1 dizem 17 variáveis, mas o modelo usa 15 (5 lags + 10). A reprodução com 15 bateu com os resultados salvos.

## 11. Próximos passos

1. Completar o passo 4 da v2 e rodar os passos 5 a 7.
2. Decidir se a v2 será aplicada ao teste, deixando claro que ele já foi visto.
3. Corrigir "17 variáveis" para 15 nos materiais da v1.
4. Confirmar custo e margem.
5. Repetir o processo no WDO.

## 12. Onde está cada coisa

| Conteúdo | Caminho (a partir da raiz do projeto) |
|---|---|
| Frente original | `frentes/ml/frente_ml.ipynb`, `resultados/ml/relatorio.md`, `resultados/ml/teste_final.csv` |
| Otimização (classificador) | `frentes/ml/otimizacao_01` a `06`, `frentes/ml/otim_rf.py`, `resultados/ml/otimizacao/` |
| RF do zero, v1 | `frentes/rf_do_zero/v1_criterio_lucro/` |
| RF do zero, v2 | `frentes/rf_do_zero/passo_01` a `07`, `frentes/rf_do_zero/saidas/A3_*` e `A4_*` |
| Melhores cenários em 2025-2026 | `frentes/rf_do_zero/melhores_cenarios_teste.py`, `frentes/rf_do_zero/saidas/melhores_cenarios_teste.csv` |
| Apresentações | `apresentacoes/` |
