# Plano: como colocar os demais testes na apresentação de ML

Atualizado em 23/11/2026. A etapa 5 já está em `slides.html` (slides 22 a 25). Este plano cobre o que entra **depois do slide 25** e antes da frente de redes neurais. **Aplicado em 23/11 em versão resumida:** um único slide (26, IV.6), com característica e motivo de cada abordagem. Os três slides detalhados abaixo ficam como opção, não aplicada.

## Ideia central

> A etapa 5 não deu lucro. Testamos outros modelos, horizontes e regras. Nenhum teve lucro significativo, e isso mostra por que o critério de escolha importa.

Três slides principais e um de apoio. Tempo: cerca de 2 min 30 s.

## Slides

### IV.6a · O que mais testamos (45 s)
**Título:** Mudamos modelo, horizonte e regra de saída

| Grupo | Quando | O que mudou |
|---|---|---|
| RF 1 min original e variantes | 02–03/11 | opera todo candle; segurar até o sinal virar |
| Classificador de 3 classes | 11/11 | sobe / neutro / desce |
| RF 60 min (v1) | 12/11 | candle maior, 36 meses |
| Alvo em 30–120 min, caudas extremas, stop | 23/11, depois da validação | ajustes sobre a etapa 5 |

**Visual:** linha do tempo com a etapa 5 marcada. Base: `linha_do_tempo_ml.html`.
**Cuidado:** os três primeiros grupos vieram **antes** da etapa 5. Dizer "outras abordagens", e não "novos testes".

### IV.6b · Com qual critério (60 s)
**Título:** Nenhuma abordagem passou no critério de lucro em 2024

| Abordagem | Critério | Passou? |
|---|---|---|
| RF 1 min | lucro com t ≥ 2; senão, menor Theil U | Não; ficou pelo Theil U |
| Classificador | lucro, t ≥ 2 e 3 de 4 trimestres; senão, maior t do bruto | Não; 0 candidatas |
| Segurar até virar | registrado antes: lucro com t ≥ 2 | Não |
| **RF 60 min (v1)** | **critério escrito escolhia 1 min; trocado por lucro após custo** | Não; t = 1,8 |
| Alvo, caudas, stop | 2022–23, depois percentil ≥ 95 em 2024 | Não |

**Cuidado:** dizer em voz alta que na v1 o critério foi trocado. Está documentado no passo 4 da v1. O t ≥ 2 vale para as quatro primeiras linhas; alvo, caudas e stop usam percentil.

### IV.6c · O que aconteceu (45 s)
**Título:** Escolher pelo lucro de 2024 não se repetiu

| Líquido por operação | 2024 | 01/2025–03/2026 | 04–09/2026 |
|---|---|---|---|
| RF 60 min, v1 | +14,1 | +3,0 (t 0,29) | −4,8 |
| RF 1 min, segurar (90%) | −2,5 | −0,7 (t −0,35) | +1,9 (t 0,64) |
| Classificador, opera sempre | −4,5 | −4,8 | −4,1 |

**Destaque:** na grade original de 12 cenários, a ordem por **Theil U** se repetiu no teste (**0,99**); a ordem por **lucro**, não (**−0,24**).
**Linha de honestidade:** v1 no total fora da amostra = +R$ 441 em 21 meses (t = 0,08); só comprar nas mesmas entradas renderia +14,9 por operação em 01/2025–03/2026.

### Apoio · Testes posteriores (só se perguntarem)

| Teste | Regras | Finalistas em 2024, líquido por operação |
|---|---|---|
| Alvo em 30–120 min | 48 | +0,7, +0,4 e +1,0 |
| Caudas de 2% a 0,5% | 36 | −23,9, −7,4 e −4,9 |
| Stop de 0,25× a 1,5× e limite diário | 18 | +1,8 (repete a etapa 5) |
| Stop 1× e parar após stop (extra) | 1 das 18 | −17,5 |

## O que fica de fora dos slides
Pesos diferentes por lado, filtros, horizontes de 10 e 20 min, hiperparâmetros da v1 e v2 (incompleta). Uma frase cobre isso: *"170 regras distintas no pipeline da etapa 5, mais 24 combinações exploratórias; as outras abordagens têm contagem própria. Nenhuma foi aprovada."*

## O que precisa mudar em outros slides
A etapa 5 contradiz trechos que **não** foram alterados:

1. **Divisão dos dados:** "validadas uma única vez em 2025–26".
2. **Onde estamos:** "passos 1–4 de 6" e "Próximo: regras de compra e venda".
3. **Próximos passos:** as regras aparecem como "a testar" e o RF como ainda a validar.
4. **Capa da frente (Objetivo):** RF "em andamento".
5. **Redes neurais:** "dados nunca vistos" em 04–09/2026. Conferir com a frente, porque a frente de ML também já usou esse período.
6. **Roteiros de fala:** ainda descrevem as regras como próximo passo.
7. **`artigo_t1.tex` e `plano_de_acao.md`:** "rodada única" e "reserva não aberta".

## Ordem de execução
1. Aprovar este plano e a lista de "o que muda".
2. Aplicar IV.6a a IV.6c e o slide de apoio.
3. Corrigir os itens 1 a 5 e conferir com as outras frentes.
4. Atualizar roteiros, artigo e plano.
5. Conferir todos os slides no navegador e exportar o PDF.

## Fontes dos números
`resultados/passo5_*`, `passo7_*`, `redesenho_*`, `caudas_*`, `stop_*`; `resultados_ml/segurar_posicao.csv` e `otimizacao/teste_final_B.csv`; `rf_do_zero/saidas/melhores_cenarios_teste.csv`. A v1 e os classificadores foram retreinados e bateram; a v2 não foi reproduzida.
