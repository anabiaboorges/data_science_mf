# Roteiro geral — Frente IV · Random Forest

Slides 13 a 22 de `slides.html`. É um texto para falar com as próprias palavras: diz o que foi testado, como foi decidido e por quê, com analogias.

---

## Slide 13 · Seis passos, um grupo de decisões por vez

**O que é o modelo.** O Random Forest é um conjunto de árvores de decisão. Cada árvore faz perguntas sobre o momento do mercado, como "o preço subiu nos últimos 15 minutos?" ou "está perto da máxima do dia?", e chega a um grupo de situações parecidas do passado. A previsão é a média do que aconteceu depois nesse grupo, e o modelo tira a média de todas as árvores. A saída é quanto o preço deve subir ou cair no próximo candle, em porcentagem.

> Analogia: é como pedir a opinião de vários especialistas, cada um com seu próprio raciocínio, e ficar com a média.

**Como trabalhamos.** Em vez de testar todas as combinações de uma vez, o que geraria milhares de modelos e muita chance de achar algo por sorte, decidimos um grupo de escolhas por vez, nesta ordem: tamanho do candle, features, janela de treino e retreino, hiperparâmetros. Depois vêm as regras de compra e venda e a validação. Cada escolha fixa a anterior e não volta atrás.

**Onde medimos.** Tudo foi medido em 2024, em janela deslizante: o modelo treina só com o passado e prevê um período que nunca viu, como funcionaria na vida real.

**O critério de escolha.** Usamos a **correlação de Spearman média mensal**, que chamamos de rô, junto com a **estatística t**.

- **O rô** mede se o modelo põe os candles na ordem certa: se os que ele prevê que vão subir mais são os que de fato sobem mais. Rô perto de um é ordem perfeita; perto de zero, a previsão não tem relação com o que aconteceu, como sortear; negativo, o modelo erra o lado.
- **O t** diz se o resultado é real ou sorte. Ele compara a média com o quanto o rô oscila de um mês para o outro. Com t abaixo de dois, não dá para distinguir de sorte e a opção é descartada.

**Por que Spearman, e não a correlação comum ou o ganho em pontos:**
- **Ele olha a ordem, não o tamanho.** Um candle que dispara numa notícia conta como "o maior da fila", e não como um número gigante que domina a conta. O mercado de um minuto tem muitos saltos assim.
- **A ordem é o que a regra de compra e venda vai usar.** Ela vai operar só as previsões mais fortes, o topo da fila. O tamanho da previsão importa menos.
- **O ganho em pontos é ruidoso demais para escolher.** Ele depende de poucos candles grandes, e operando todo candle todas as opções perdem para o custo. Comparar por ele seria comparar prejuízos.

**Por que média mensal, e não o ano inteiro de uma vez:**
- **São doze provas, e não uma.** Um número único pode vir de dois meses muito bons e dez nulos. Com doze meses, dá para ver se o sinal se repete.
- **O erro fica honesto.** A agitação do mercado vem em ondas, e candles de um mesmo dia agitado se parecem. A margem de erro calculada pela variação real entre os meses não engana.

> Analogia do t: dois alunos com média 7. Um tirou sempre perto de 7; o outro tirou 2, 10, 3, 10. A média é igual, mas só o primeiro é de fato "um aluno 7". O t mede isso.

**Regra de desempate.** Se duas opções ficam a menos de um erro-padrão uma da outra, a diferença pode ser sorte. Nesse caso, fica a mais simples.

---

## Slide 14 · Candles

**O que testamos.** O horizonte da previsão: o próximo minuto, ou os próximos 5, 15, 30 ou 60 minutos. Usamos o mesmo modelo simples em todos, para que a única diferença fosse o candle.

**Como ler o gráfico.** Cada linha é um tamanho de candle. O ponto é o rô médio e a barra é a margem de erro. Se a barra cruza o zero, o resultado é indistinguível de ruído.

**Resultado e escolha.** Só o candle de 1 minuto ficou inteiro do lado positivo: rô de 0,018, t de 4,9, positivo nos doze meses. Os demais ficaram em cima do zero.

**Por quê.** Candles maiores têm uma vantagem: o custo pesa menos no movimento, 1,6% no de 60 minutos contra 12% no de 1 minuto. Mas custo baixo sem sinal não gera lucro.

---

## Slide 15 · Features

**O que são.** Features são as informações que o modelo recebe sobre o momento atual. Usamos 21, em sete grupos: retornos recentes, retornos acumulados, posição no dia, volatilidade, forma do candle, volume e gap, e dólar. Todas usam só o que já era conhecido no fechamento do candle.

> Analogia: é a ficha de um paciente que o médico lê antes do diagnóstico.

**O que testamos.** Primeiro, as 5 features básicas do passo dos candles contra as 21 juntas, com a mesma árvore.

**Testamos também cenários filtrando features pela análise de dados**, feita só com dados até 2023:
- **pela relação com o próximo retorno:** ficavam só as features com relação estável com a direção do próximo candle, com t acima de dois e o mesmo sinal em 2021, 2022 e 2023, ou que carregavam informação sobre o movimento;
- **pela correlação entre as próprias features:** quando duas passavam de 0,8 de correlação, ficava só uma. É o caso do retorno do candle com o corpo do candle (0,99) e das duas medidas de volatilidade (0,86).

> Fala sugerida: "Testei também cenários filtrando algumas features com base na correlação delas com o próximo retorno e na correlação entre elas mesmas, para tirar as redundantes. Mas, de todos os cenários, o das 21 features foi o melhor."

**Resultado.** As 21 tiveram rô de 0,023, contra 0,018 das básicas e 0,012 do conjunto filtrado.

**Por que filtrar piorou.** A análise olha cada feature sozinha, e o Random Forest ganha nas combinações. A volatilidade, sozinha, não diz se o preço sobe ou cai, mas ajuda a mostrar quando os outros sinais são mais fortes.

> Analogia: num time, um jogador que não faz gol pode ser o que dá o passe. Tirar quem não marca enfraquece o time.

---

## Slide 16 · Janela de treino

**O que testamos.** Quanto passado o modelo usa para aprender: 3, 6, 12 ou 24 meses, ou todo o histórico.

**O dilema.**
- **Janela curta** se adapta ao mercado atual, mas tem poucos dados.
- **Janela longa** tem muitos dados, mas mistura épocas diferentes de preço e de volatilidade.

> Analogia: para prever o trânsito de amanhã, olhar só a última semana é pouco, e olhar os últimos dez anos inclui uma cidade que já mudou.

**Resultado e escolha.** 12 meses venceu com folga: rô de 0,024 e t de 6. Com 24 meses, o rô caiu, e com todo o histórico ficou negativo.

---

## Slide 17 · Retreino

**O que testamos.** De quanto em quanto tempo refazer o modelo: todo mês, todo trimestre ou todo semestre.

**Resultado e escolha.** As três ficaram empatadas dentro da margem de erro, a faixa verde do gráfico. Pela regra, ficou a mais simples de operar: **semestral**, com dois treinos por ano. O semestral teve correlação um pouco menor, mas a diferença é indistinguível de sorte.

---

## Slide 18 · Número de árvores

**O que testamos.** 25, 50, 100, 200 e 400 árvores.

**Como funciona.** Cada árvore é um especialista dando opinião, e o modelo tira a média. Com poucas árvores, a média oscila. Com mais, ela se estabiliza.

> Analogia: uma pesquisa eleitoral. Ouvir mais gente deixa o resultado estável, mas, a partir de certo ponto, ouvir o dobro de pessoas quase não muda nada e só encarece a pesquisa.

**Resultado e escolha.** A partir de **50 árvores**, a correlação parou de mudar. Ficou o menor número a menos de um milésimo do resultado com 400.

---

## Slide 19 · Profundidade e folha

**O que é cada uma.**
- **Profundidade** é quantas perguntas a árvore faz em sequência. Com poucas perguntas, ela identifica situações amplas, que se repetem. Com muitas, chega a situações tão específicas que só existiram no passado.
- **Folha** é o grupo final de situações parecidas onde a árvore chega, e a previsão é a média desse grupo. A folha mínima é quantos candles o grupo precisa ter. Folha grande quer dizer menos grupos e cada um com mais casos: com folha de 10%, cada árvore tem no máximo dez grupos, ou seja, no máximo dez respostas diferentes, cada uma baseada em milhares de candles.

> Analogia da folha: avaliações de restaurante. Uma nota 5 com três avaliações não diz muito; uma nota 4,3 com duas mil é confiável.

> Analogia das duas juntas: árvore funda com folha pequena é o aluno que decora a prova antiga, que vai muito bem nela e mal na prova nova. Árvore rasa com folha grande é o aluno que aprende a regra geral.

**A ordem dos testes:**

1. **Grade inicial.** Profundidade 3, 5 e 8 × folha de 0,5%, 1%, 2% e 5%, com todas as features disponíveis em cada pergunta. Olhamos duas coisas:
   - o rô no teste;
   - o **gap**, que é o rô no treino menos o rô no teste. Ele mede o quanto o modelo decorou.

   As árvores fundas com folhas pequenas decoraram: profundidade 8 com folha de 0,5% teve 0,082 no treino e só 0,019 no teste. Os dois melhores cenários foram **profundidade 3 com folha de 5%** (0,025) e **profundidade 5 com folha de 5%** (0,024).

2. **Features por pergunta, nos dois melhores cenários.** Em cada pergunta, a árvore pode considerar todas as 21 features, metade ou a raiz quadrada (cerca de 4 ou 5), sorteadas a cada vez. As três opções empataram, e ficou a **raiz**, a mais simples.

   > Analogia: um júri. Se todos os jurados leem o mesmo jornal, pensam igual e erram juntos. Se cada um recebe documentos diferentes, as opiniões variam, e na média os erros se compensam.

3. **Extensão da grade.** O melhor cenário estava no canto da grade, com a menor profundidade e a maior folha testadas. Quando o melhor fica na borda, não dá para saber se o ótimo está além dela. Por isso testamos **profundidade 2 e folhas de 10% e 20%**, já com a raiz. Nessa extensão apareceu o **melhor cenário de todos: profundidade 2 com folha de 5%**, com rô de 0,026. Com folha de 20%, a árvore não consegue mais dividir os dados e a previsão vira uma constante.

**Escolha final: profundidade 2 com folha de 10%, e por quê:**
- **Empate.** A diferença para o melhor cenário (0,026 contra 0,024) fica dentro da margem de erro. Escolher o maior número entre empatados é escolher o mais sortudo.
- **É a mais simples entre as empatadas:** árvore mais rasa e folha maior, como manda a regra definida antes de rodar.
- **É a que menos decora:** tem o menor gap da grade, 0,014.
- **Os dados já apontavam nessa direção.** O sinal é tão fraco que nenhuma folha pequena tem média confiável; ele só aparece na média de muitos candles.
- **É o limite útil:** folha de 20% já não divide nada.
- **O lado honesto:** a folha de 5% teve correlação e ganho bruto maiores. Ela não foi descartada e segue como alternativa no passo das regras, onde o ganho decide.

---

## Slide 20 · Modelo final

**O modelo:** candle de 1 minuto, 21 features, janela de 12 meses, retreino semestral, 50 árvores, profundidade 2, folha de 10% e raiz das features por pergunta.

**Resultado em 2024:** rô de 0,024, t de 4,1, positivo em 11 dos 12 meses e acerto de direção de 51,2%.

**O ganho bruto caiu para 0,26 ponto por operação.** O modelo mais simples achata o tamanho das previsões, mas a ordem, que é o que o rô mede, se manteve.

---

## Slide 21 · Ganho acumulado

**O gráfico de cima** mostra o modelo operando todos os candles de 2024, sem custo: +36,8 mil pontos, quase tudo no 2º semestre. A linha sobe, então o sinal existe.

**O gráfico de baixo** mostra o mesmo modelo pagando 5 pontos por operação: −670 mil pontos. Operando todo candle, o sinal não paga o custo. Para isso, o rô precisaria ser perto de 0,11, mais de quatro vezes o nosso.

**É aqui que a escolha pela correlação se justifica.** O ganho operando todo candle mede uma estratégia que não vai ser usada. O que importa é se o modelo sabe ordenar os candles, porque a operação real vai usar só o topo da fila.

> Um exemplo do que acontece ao escolher pelo ganho: na frente estatística, o ARIMAX foi escolhido com +10 pontos por operação na seleção e caiu para −41 na validação.

---

## Slide 22 · Próximo passo

1. **Critérios de compra e venda:** operar só as previsões mais fortes, escolher o lado e definir a saída com alvo (take profit) e stop loss em múltiplos da volatilidade. Agora sim o critério é o dinheiro: pontos por operação depois do custo.
2. **Validação:** modelo e regras congelados, rodados uma única vez em 2025–26, em dados nunca usados.
3. **Comparação:** as mesmas regras aplicadas ao ARIMAX e à melhor rede neural, todos medidos com a mesma régua e contra sorteios aleatórios.

**Fechamento:** o Random Forest achou um sinal real e estável, ainda abaixo do custo. A pergunta agora é se, escolhendo bem quando entrar e como sair, esse sinal paga o custo.
