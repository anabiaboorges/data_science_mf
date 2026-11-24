# Roteiro falado — Frente IV · Random Forest

Texto para ler em voz alta, slides 13 a 22 de `slides.html`. Entre colchetes está para onde apontar. Os parágrafos marcados com **(se houver tempo)** aprofundam o ponto e podem ser pulados: sem eles, a fala leva cerca de 15 minutos; com eles, cerca de 17. A ideia é cortar depois.

---

## Slide 13 · Seis passos

Na frente de machine learning, usamos um Random Forest, ou floresta aleatória: um conjunto de árvores de decisão. Cada árvore faz perguntas sobre o momento do mercado, como "o preço subiu nos últimos quinze minutos?", chega a um grupo de situações parecidas do passado, que chamamos de folha, e prevê a média do que aconteceu depois. O modelo tira a média das árvores. A saída é quanto, em porcentagem, o preço deve subir ou cair no próximo candle: positivo indica compra, negativo, venda.

Decidimos um grupo de escolhas por vez, em seis passos; os quatro primeiros estão concluídos. Toda escolha foi medida em 2024 em janela deslizante: o modelo treina só com o passado e prevê um período que nunca viu, como funcionaria na vida real.

A métrica principal é a correlação de Spearman, que vou chamar de rô. Colocamos as previsões do mês em fila, da menor para a maior, e fazemos o mesmo com os retornos que aconteceram. O rô mede se as duas filas andam juntas. Rô igual a um é acertar a ordem inteira. Rô igual a zero é não ter relação nenhuma, como sortear. Rô negativo é errar o lado de forma sistemática. No candle de um minuto, que é quase todo ruído, dois centésimos já é um sinal relevante.

Calculamos um rô por mês e tiramos a média de 2024. A estatística t robusta é essa média dividida pelo seu erro-padrão: perto de zero, não dá para distinguir de sorte; acima de dois, o sinal dificilmente é acaso. E, se duas opções ficam a menos de um erro-padrão uma da outra, ficamos com a mais simples.

**(se houver tempo)** Dois indicadores de apoio aparecem nos quadros. O U de Theil compara o erro do modelo com o de prever sempre zero: igual a um, empata; abaixo de um, erra menos. O acerto de direção é a porcentagem de vezes em que o modelo acertou se o preço subiu ou caiu: cinquenta por cento é cara ou coroa.

## Slide 14 · Candles

O primeiro passo decide o horizonte: prever o próximo minuto ou os próximos cinco, quinze, trinta ou sessenta. **Como escolhemos:** o mesmo Random Forest simples em todos, com cinco features básicas e janela de doze meses, para que a única diferença fosse o candle. Venceria o de maior rô médio mensal no teste, desde que tivesse t acima de dois e U de Theil de no máximo um vírgula zero zero um, ou seja, no máximo um milésimo pior que prever zero.

[Apontar para as barras.] Cada linha é um tamanho de candle. O ponto é o rô médio mensal e a barra é a margem de erro, de dois erros-padrão. Se a barra cruza a linha do zero, o resultado não se distingue de ruído.

Só o candle de um minuto ficou inteiro à direita do zero: rô de zero vírgula zero dezoito, t de quatro vírgula nove, positivo nos doze meses. Os demais ficaram em cima do zero, com t entre menos meio e zero vírgula três. [Apontar para o quadro.] O modelo, nesse ponto, ganhava zero vírgula trinta e sete ponto por operação antes do custo.

**(se houver tempo)** Candles maiores têm uma vantagem: o custo de cinco pontos pesa doze por cento do movimento típico de um minuto e só um vírgula seis por cento no de sessenta. Mas custo baixo sem sinal não gera lucro.

## Slide 15 · Features

O segundo passo decide o que o modelo enxerga. Uma feature é uma informação que o modelo recebe sobre o momento atual para fazer a previsão, como a ficha de um paciente que o médico lê antes do diagnóstico. Todas são calculadas só com o que já era conhecido no fechamento do candle, para o modelo nunca espiar o futuro.

[Apontar para a lista embaixo do gráfico.] Montamos vinte e uma features, em sete grupos. Os retornos recentes: quanto o preço andou no último minuto e nos dois anteriores. Os retornos acumulados: quanto andou nos últimos cinco, quinze e sessenta minutos. A posição no dia: quanto o dia já subiu ou caiu, a hora, e onde o preço está entre a mínima e a máxima do dia. A volatilidade, que mede o quanto o mercado está agitado. A forma do candle: o tamanho, o corpo e onde ele fechou. O volume comparado ao normal do horário e o salto da abertura. E o dólar, no último minuto e nos últimos quinze.

**Como escolhemos:** usamos a mesma árvore do passo dos candles e mudamos só as features. Venceria o conjunto de maior rô no teste, e um conjunto menor ganharia se ficasse a até um erro-padrão do melhor. [Apontar para o gráfico.] Testamos três conjuntos. As cinco básicas, usadas na escolha dos candles, deram rô de zero vírgula zero dezoito. As vinte e uma juntas deram zero vírgula zero vinte e três, com t de cinco vírgula três. [Só falado, não está no gráfico.] Testamos também treze features pré-selecionadas por uma análise de dados, que olhava cada uma sozinha, e elas caíram para zero vírgula zero doze.

A linha tracejada marca um erro-padrão abaixo da melhor. Nenhum conjunto menor chegou nela, então o melhor cenário foi o das vinte e uma.

Filtrar piorou porque o Random Forest ganha nas combinações. Uma feature que sozinha parece fraca pode ser útil junto com outra: a volatilidade, por exemplo, não diz se o preço vai subir ou cair, mas ajuda a indicar quando os outros sinais são mais fortes. Cortar features com base nelas isoladas jogou essas combinações fora.

[Apontar para o quadro do modelo.] O que mudou no modelo: o rô subiu de zero vírgula zero dezoito para zero vírgula zero vinte e três, o acerto de direção, de cinquenta vírgula nove para cinquenta e um vírgula três por cento, e o ganho bruto, de zero vírgula trinta e sete para zero vírgula quarenta e três ponto por operação.

## Slide 16 · Janela de treino

Quanto passado usar. Janela curta se adapta ao mercado atual, mas tem poucos dados; janela longa mistura épocas diferentes de preço e de volatilidade. **Como escolhemos:** testamos três, seis, doze e vinte e quatro meses e todo o histórico, todos com retreino trimestral, e venceria o maior rô no teste, com t acima de dois e U de Theil dentro do limite.

[Apontar para o gráfico.] Três meses deram zero vírgula zero doze; seis, zero vírgula zero dezessete; doze, zero vírgula zero vinte e quatro, com t de seis, o melhor com folga. Com vinte e quatro meses, caiu para zero vírgula zero dez, e com todo o histórico ficou negativo. Escolhemos doze meses, e o ganho bruto foi a zero vírgula cinquenta e um ponto.

## Slide 17 · Retreino

De quanto em quanto tempo refazer o modelo. **Como escolhemos:** testamos retreino mensal, trimestral e semestral nas duas melhores janelas, e definimos antes que, se empatassem dentro de um erro-padrão, ficaria o menos frequente, por ser mais simples de operar.

[Apontar para a faixa verde.] A faixa verde é a zona de empate. Mensal e trimestral tiveram zero vírgula zero vinte e quatro, e semestral, zero vírgula zero vinte e dois: os três dentro da faixa. Pela regra, fica o mais simples de operar, o semestral, com dois treinos por ano. O trimestral fica anotado como alternativa.

## Slide 18 · Número de árvores

Começam os hiperparâmetros, que controlam o tamanho e a complexidade do modelo. O primeiro é o número de árvores. **Como escolhemos:** testamos vinte e cinco, cinquenta, cem, duzentas e quatrocentas árvores, com profundidade cinco e folha de um por cento fixas, e ficaria o menor número com rô a menos de um milésimo do obtido com quatrocentas.

[Apontar para a curva.] Com vinte e cinco árvores, o rô fica em zero vírgula zero dezenove; com cinquenta, zero vírgula zero vinte e dois, e daí em diante a curva fica plana, dentro da faixa de um milésimo do valor com quatrocentas. Mais árvores só custam tempo.

## Slide 19 · Profundidade e folha

A profundidade é quantas perguntas a árvore faz em sequência. A folha mínima é quantos candles cada grupo final precisa reunir. **Como escolhemos:** cruzamos profundidades de três, cinco e oito com folhas de meio, um, dois e cinco por cento do treino, doze combinações, e olhamos duas coisas: o rô no teste e a diferença para o rô no treino, que mostra sobreajuste.

[Apontar para a grade.] Linhas são profundidade, colunas são folha. O número grande é o rô no teste, e quanto mais laranja, maior. O gap, embaixo, é o rô no treino menos o rô no teste: mede o quanto o modelo decorou o treino em vez de aprender algo que se repete.

Árvores fundas com folhas pequenas decoram: profundidade oito com folha de meio por cento tem zero vírgula zero oitenta e dois no treino e só zero vírgula zero dezenove no teste. Na grade inicial, o melhor ficou no canto, árvore rasa com folha grande: profundidade três com folha de cinco por cento. Como caiu na borda, não sabíamos se o ótimo estava além dela, e estendemos a grade. [Apontar para as células tracejadas.] As células com contorno tracejado são essa extensão: profundidade dois e folha de dez por cento. [Apontar para a célula laranja forte.] O melhor cenário foi profundidade dois com folha de cinco por cento, com a maior correlação, zero vírgula zero vinte e seis. [Apontar para a célula de profundidade dois e folha de dez por cento.] Profundidade dois com folha de dez por cento deu zero vírgula zero vinte e quatro, empatada dentro da margem de erro, e com o menor gap da grade, zero vírgula zero catorze: é a que menos decora. [Só falado, não está na matriz.] Também testamos folha de vinte por cento, mas aí a árvore praticamente não consegue dividir os dados e a previsão vira uma constante. Pela regra, ficou a mais simples entre as empatadas: profundidade dois com folha de dez por cento.

**(se houver tempo)** Isso bate com os dados: uma folha precisaria de cerca de quinhentos e quarenta mil candles para ter média confiável, e uma janela de doze meses tem cento e trinta mil. Nenhuma folha sozinha é confiável; o sinal só aparece na média de muitas folhas e árvores.

## Slide 20 · Modelo final

**Como escolhemos:** primeiro testamos quantas features cada divisão da árvore pode considerar: todas as vinte e uma, metade ou a raiz quadrada do total. Depois estendemos a grade para profundidade dois e folhas de dez e vinte por cento. A regra: entre as opções a até um erro-padrão da melhor, fica a mais rasa e de folha maior, que é a mais simples.

[Apontar para o gráfico.] Considerar só a raiz quadrada das features deixa as árvores mais diferentes entre si, o que melhora a média. Várias configurações empataram. A maior correlação foi profundidade dois com folha de cinco por cento, zero vírgula zero vinte e seis, e a regra escolheu a mais simples entre as empatadas: profundidade dois e folha de dez por cento. Com folha de vinte por cento, a árvore não divide e a previsão fica constante.

O modelo final tem rô de zero vírgula zero vinte e quatro, t de quatro vírgula um, positivo em onze de doze meses e acerto de direção de cinquenta e um vírgula dois por cento. O ganho bruto caiu para zero vírgula vinte e seis ponto, porque o modelo mais simples achata o tamanho das previsões. A ordem, que é o que o rô mede, se manteve.

## Slide 21 · Ganho acumulado

[Apontar para o painel de cima.] O modelo final operando todos os candles de 2024, sem custo: trinta e seis vírgula oito mil pontos, quase tudo no segundo semestre. [Painel de baixo.] Pagando cinco pontos por operação, numa escala diferente, menos seiscentos e setenta mil. O sinal existe, mas operar todo candle não paga o custo: para isso, o rô precisaria ser perto de zero vírgula onze, mais de quatro vezes o nosso.

**(se houver tempo)** [Apontar para o quadro.] No próprio treino, os modelos tinham rô de zero vírgula zero trinta e quatro; no teste, zero vírgula zero vinte e quatro. É normal ir melhor nos dados já vistos, e a queda é moderada perto da árvore funda.

## Por que escolher pela correlação, e não pelo ganho

Por que todas as escolhas seguiram a correlação, e não o ganho em pontos? Quatro motivos.

Primeiro: o ganho bruto mede uma regra que nunca vai ser usada. Ele opera todos os cento e quarenta e um mil candles, e com custo todas as opções perdem entre quatro e meio e quatro vírgula sete pontos por operação. Escolher por ele é escolher o prejuízo menos ruim de uma estratégia descartada.

Segundo: a regra real vai operar só as previsões mais fortes, por exemplo os dez por cento do topo da fila. Para isso importa a ordem, saber quais candles tendem a andar mais, e é exatamente o que o rô mede. O tamanho da previsão não importa, porque o corte é pela posição na fila.

Terceiro: o ganho é ruidoso demais. O erro-padrão dele é de zero vírgula onze ponto, maior que quase todas as diferenças entre as opções. Ele depende de poucos candles grandes: no primeiro semestre, com rô positivo quase todo mês, o ganho foi praticamente zero. O rô usa a posição na fila, não o tamanho, e um candle extremo não domina.

Quarto: já vimos o que acontece ao escolher pelo ganho. O ARIMAX tinha mais dez pontos por operação na seleção e caiu para menos quarenta e um na validação. Escolher o maior ganho de um único ano é, muitas vezes, escolher quem teve mais sorte.

Por isso a seleção usa o rô, e o dinheiro é medido no passo seguinte, onde a folha de cinco por cento segue como alternativa.

## Slide 22 · Próximo passo

O modelo tem um sinal real, mas, operando todo candle, não paga o custo. O próximo passo é transformar esse sinal em operação, em três etapas.

[Apontar para a primeira coluna.] Primeiro, os critérios de compra e venda, que estão em desenvolvimento. Na entrada, em vez de operar todo candle, vamos operar só as previsões mais fortes, os candles em que o modelo está mais confiante. É aqui que a escolha pela correlação se paga: o que importa para esse corte é a ordem das previsões. No lado, vamos testar operar compra e venda, só compra ou só venda. E, na saída, um alvo de lucro, o take profit, e um stop loss, os dois medidos em múltiplos da volatilidade recente, para se ajustarem sozinhos a dias calmos e agitados. Também vamos testar a saída simples, no fechamento do candle seguinte. O critério de escolha, agora sim, é o dinheiro: pontos por operação depois do custo, em 2024.

[Apontar para a segunda coluna.] Segundo, a validação. O modelo e as regras são congelados e rodam uma única vez no período de janeiro de 2025 a março de 2026, que nunca foi usado em nenhuma escolha. O resultado é reportado com custo de cinco pontos por operação.

[Apontar para a terceira coluna.] Terceiro, a comparação com as outras frentes. As mesmas regras são aplicadas ao ARIMAX e à melhor rede neural, e todos são medidos com a mesma régua: pontos por operação depois do custo, custo de equilíbrio, número de operações e comparação com sorteios aleatórios de direção nos mesmos candles. Se o modelo não superar o sorteio, o ganho foi sorte.

[Apontar para a faixa de baixo.] E as alternativas que anotamos no caminho, a folha de cinco por cento e o retreino trimestral, também passam pelas regras. Lá, o ganho depois do custo decide.

Resumindo: o Random Forest achou um sinal real e estável no candle de um minuto, ainda abaixo do custo. A pergunta agora é se, escolhendo bem quando entrar e como sair, esse sinal paga o custo.
