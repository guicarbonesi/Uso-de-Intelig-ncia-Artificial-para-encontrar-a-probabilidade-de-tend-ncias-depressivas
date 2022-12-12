## Uso de Inteligência Artificial para encontrar a probabilidade de tendências depressivas

### Depressão

![Depression Illustration](https://s3.amazonaws.com/spectrumnews-web-assets/wp-content/uploads/2018/06/22080230/JuliaYellow-Depression_1120x550_acf_cropped_1120x550_acf_cropped.jpg)

Segundo a Organização Mundial da Saúde, a depressão é a principal causa de incapacidade em todo o mundo. Globalmente, mais de 300 milhões de pessoas de todas as idades sofrem com o distúrbio. E a incidência do distúrbio está aumentando em todos os lugares. A depressão é uma condição complexa, envolvendo muitos sistemas do corpo, incluindo o sistema imunológico, seja como causa ou efeito. Interrompe o sono e interfere no apetite; em alguns casos, causa perda de peso; em outros, contribui para o ganho de peso. A depressão também costuma ser acompanhada de ansiedade. A pesquisa indica que não apenas as duas condições ocorrem simultaneamente, mas também se sobrepõem nos padrões de vulnerabilidade.

Devido à sua complexidade, uma compreensão completa da depressão tem sido difícil de alcançar. Os pesquisadores têm algumas evidências de que a suscetibilidade à depressão está relacionada à dieta, tanto diretamente – por meio do consumo inadequado de nutrientes, como gorduras ômega-3 – quanto indiretamente, por meio da variedade de bactérias que povoam o intestino. Mas a depressão envolve humor e pensamentos, assim como o corpo, e causa dor tanto para aqueles que vivem com o distúrbio quanto para aqueles que se preocupam com eles.

Mesmo nos casos mais graves, a depressão é altamente tratável. A condição geralmente é cíclica e o tratamento precoce pode prevenir ou prevenir episódios recorrentes. Muitos estudos mostram que o tratamento mais eficaz é a terapia cognitivo-comportamental, que aborda padrões de pensamento problemáticos, com ou sem o uso de drogas antidepressivas. Além disso, estão se acumulando rapidamente evidências de que a meditação mindfulness regular, sozinha ou combinada com a terapia cognitiva, pode interromper a depressão antes que ela comece, diminuindo a reatividade a experiências angustiantes, permitindo efetivamente o desengajamento da atenção dos pensamentos negativos repetitivos que muitas vezes estabelecem a espiral descendente. de humor em movimento.


---
### Naive Bayes

Naive Bayes is a machine learning model that is used for large volumes of data, even if you are working with data that has millions of data records the recommended approach is Naive Bayes. It gives very good results when it comes to NLP tasks such as sentimental analysis. It is a fast and uncomplicated classification algorithm.

#### Teorema Naive Bayes

Naive Bayes é um modelo de aprendizado de máquina usado para grandes volumes de dados, mesmo se você estiver trabalhando com dados que possuem milhões de registros de dados, a abordagem recomendada é Naive Bayes. Fornece resultados muito bons quando se trata de tarefas de PNL, como análise sentimental. É um algoritmo de classificação rápido e descomplicado.

##### Probabilidade Condicional: 
![Bayes Theorem in cool led lamp](https://upload.wikimedia.org/wikipedia/commons/1/18/Bayes%27_Theorem_MMB_01.jpg)

Onde, <br>
P(A): A probabilidade da hipótese H ser verdadeira. Isso é conhecido como probabilidade a priori. <br>
P(B): A probabilidade da evidência. <br>
P(A|B): A probabilidade da evidência dada que a hipótese é verdadeira. <br>
P(B|A): A probabilidade da hipótese dado que a evidência é verdadeira. <br>

#### Classificador Naive Bayes
- É uma espécie de classificador que trabalha com o teorema de Bayes.
- A previsão de probabilidades de pertinência é feita para cada classe, como a probabilidade de pontos de dados associados a uma determinada classe.
- A classe com probabilidade máxima é avaliada como a classe mais adequada.
- Isso também é conhecido como Maximum A Posteriori (MAP).
- O MAP para uma hipótese é: 
  - 𝑀𝐴𝑃 (𝐻) = max 𝑃((𝐻|𝐸))  
  - 𝑀𝐴𝑃 (𝐻) = max 𝑃((𝐻|𝐸)  ∗ (𝑃(𝐻)) /𝑃(𝐸))  
  - 𝑀𝐴𝑃 (𝐻) = max(𝑃(𝐸|𝐻) ∗ 𝑃(𝐻))
  - 𝑃 (𝐸) é a probabilidade de evidência e é usada para normalizar o resultado. O resultado não será afetado pela remoção 𝑃(𝐸).
- Os classificadores NB concluem que todas as variáveis ​​ou características não estão relacionadas entre si.
- A existência ou ausência de uma variável não impacta a existência ou ausência de qualquer outra variável.
- Exemplo:
  - A fruta pode ser observada como uma maçã se for vermelha, redonda e com cerca de 4 ″ de diâmetro.
  - Neste caso também, mesmo que todas as características estejam inter-relacionadas entre si, o classificador NB observará todas elas independentemente, contribuindo para a probabilidade de que a fruta seja uma maçã.
- Experimentamos a hipótese em conjuntos de dados reais, com vários recursos.
- Assim, a computação torna-se complexa.

#### Tipos de Algoritmos 
1. Gaussian Naïve Bayes: Quando os valores característicos são contínuos por natureza, assume-se que os valores vinculados a cada classe são dispersos de acordo com Gaussian, que é a Distribuição Normal.

2. Multinomial Naive Bayes: Multinomial Naive Bayes é o preferido para usar em dados que são distribuídos multinomial. É amplamente utilizado na classificação de texto em PNL. Cada evento na classificação de texto constitui a presença de uma palavra em um documento.

 3. Bernoulli Naive Bayes: Quando os dados são distribuídos de acordo com as distribuições multivariadas de Bernoulli, então Bernoulli Naive Bayes é usado. Isso significa que existem vários recursos, mas supõe-se que cada um contenha um valor binário. Portanto, requer que os recursos tenham valor binário.
---
### Objetivos
O objetivo deste projeto é construir um modelo de aprendizado de máquina que possa classificar um texto em duas categorias: deprimido ou não deprimido(neutro) usando a fórmula de Bayes, calculando o sentimento das palavras de frequência nos dados de treinamento.

##### Autores
- Guilherme Carbonesi 
- Carlos Eduardo Vasconcelos Silva
- Renan Soares da Silva 
- Lucas Rabelo Moreaux 
- Vitor Selloti de Souza

Referências: [https://bvsms.saude.gov.br](https://bvsms.saude.gov.br/depressao-4/), [analyticssteps.com](https://www.analyticssteps.com/blogs/what-naive-bayes-algorithm-machine-learning) , [Mineracao-de-emocao-em-textos-com-python-e-nltk](https://www.udemy.com/course/mineracao-de-emocao-em-textos-com-python-e-nltk/) , [Introdução ao Processamento de Linguagem Natural](https://cienciaenegocios.com/processamento-de-linguagem-natural-nlp/) , [Uso de mineração de texto como ferramenta
de avaliação da qualidade informacional em laudos
eletrônicos de mamografia](https://www.scielo.br/j/rb/a/8jbwGc9KhMvBXHBxtgyW4Fz/?format=pdf&lang=pt) 
