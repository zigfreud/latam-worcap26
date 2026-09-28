# Método, resultados e limites

## Resultado principal

A família escolhida é **LightGBM com memória atmosférica de três meses e contexto local de pressão (E09)**. Na avaliação histórica de 2017 a 2022, obteve **RMSE de 1,831045 mm/dia**, contra **1,858171 mm/dia** da climatologia: redução relativa de **1,46%**.

Esses valores descrevem **desenvolvimento histórico**. Os mesmos três blocos foram reutilizados para estudar diferentes hipóteses; portanto, os resultados não constituem confirmação independente nem demonstração de significância estatística. O projeto conserva essa distinção ao apresentar os ganhos e as alternativas que não passaram pelos critérios de avanço.

## O que é previsto

O alvo é a **taxa média mensal de precipitação**, expressa em mm/dia, no mês seguinte. Cada exemplo corresponde a uma célula da grade e a um mês. A unidade mm/dia não torna o problema uma previsão de chuva para cada dia individual.

- Grade de 301 latitudes por 261 longitudes: **78.561 células**, incluindo oceano.
- Histórico disponível: janeiro de 1940 a dezembro de 2022.
- Horizonte de inferência: 24 meses, de janeiro de 2023 a dezembro de 2024.
- Saída: **1.885.464 previsões**, preservando os identificadores e sua ordem no arquivo de referência.
- Métrica: raiz do erro quadrático médio, **sem ponderação por área**, sobre toda a grade e todos os meses avaliados.

Para estimar o mês T, o modelo usa atmosfera de T-1 e memória de T-2 e T-3. Cada mês de inferência recebe o campo atmosférico anterior fornecido pelo conjunto de dados; não se trata de uma única previsão de 24 meses emitida no fim de 2022. A chuva observada durante 2023-2024 não entra como atributo nem como rótulo de ajuste. Os dados distribuídos não oferecem esses rótulos para calcular erro local nesse período.

## Arquitetura

O modelo aprende a diferença entre a chuva observada e a climatologia mensal de cada célula. Na inferência, soma a correção estimada à climatologia e limita o resultado inferior a zero.

Os **53 atributos** são:

| Grupo | Quantidade | Construção |
|---|---:|---|
| Anomalias atmosféricas | 27 | Nove variáveis em T-1, T-2 e T-3; médias sazonais estimadas somente no treino |
| Variação entre meses | 18 | Diferenças entre as anomalias de T-1 e as de T-2/T-3 |
| Contexto | 5 | Latitude, longitude, seno/cosseno do mês-alvo e climatologia da precipitação |
| Contexto de pressão | 3 | Valor bruto em T-1, média espacial 5 × 5 e diferença entre a célula e essa média |

As nove variáveis são temperatura a 2 m, cobertura de nuvens, umidade específica a 850 hPa, pressão de superfície, vento zonal e meridional a 850 hPa, temperatura e umidade relativa a 850 hPa e geopotencial a 850 hPa, conforme os nomes dos arquivos de entrada.

**Limite de interpretação:** os arquivos atmosféricos não trazem metadados suficientes para certificar todas as unidades físicas. O contexto de pressão utiliza valores brutos, sem conversão para altitude nem alegação de unidade certificada. A média 5 × 5 usa somente as células existentes nas bordas da grade.

A amostra de treino contém 2.500 pontos fixos da grade, escolhidos com semente 20260916. Uma janela de 30 anos fornece 900.000 exemplos. A avaliação usa todas as células, não apenas os pontos amostrados.

## Validação temporal

Os dados são separados por tempo, nunca por sorteio de linhas entre treino e avaliação. Cada bloco usa 30 anos de ajuste, precedidos por uma escolha interna de hiperparâmetros que só vê o passado.

| Reajuste do bloco | Avaliação histórica |
|---|---|
| 1987-2016 | 2017-2018 |
| 1989-2018 | 2019-2020 |
| 1991-2020 | 2021-2022 |

Exemplo do primeiro bloco: ajustar em 1983-2012; escolher entre 100 e 300 árvores em 2013-2016; reajustar em 1987-2016; avaliar em 2017-2018. As estatísticas de pré-processamento são recalculadas somente com o treino correspondente. O período de avaliação do bloco não é usado para escolher seu número de árvores.

Ao todo, os três blocos abrangem 72 meses e 5.656.392 pares célula-mês. Esse volume não corresponde a milhões de observações independentes: há dependência espacial e temporal.

## Comparação das hipóteses

Todos os RMSE abaixo estão em mm/dia da média mensal. Os resultados agregados são calculados sobre os erros de todas as células e meses; não são a média simples de RMSE regionais.

| Experimento | Receita | RMSE da grade | Leitura e decisão |
|---|---|---:|---|
| E00 | Climatologia mensal por célula | 1,858171 | Referência de comparação |
| E01 | Ridge local | 1,836497 | Relação linear regularizada por célula |
| E02 | LightGBM com atmosfera de T-1 | 1,835950 | Referência de árvores compartilhadas pela grade |
| E07 | LightGBM com três meses de memória | 1,831899 | Memória reduz o erro agregado |
| **E09** | **Memória e contexto de pressão** | **1,831045** | **Melhora os três blocos frente ao E07; família escolhida** |
| E10 | Combinação de Ridge com E09 | 1,830821 | Menor agregado, mas não passa sua regra de consistência |
| E11 | E09 com UF, bioma e vegetação | 1,831252 | Piora o agregado frente ao E09 |
| E12 | E11 no Brasil, E09 fora | 1,830952 | Ganho brasileiro instável; não promove |
| E13 | Especialista com 2.500 pontos brasileiros, E09 fora | 1,833960 | Piora os três blocos brasileiros; não promove |

A alternativa E10 com Ridge e E07 obteve 1,832558 mm/dia. A combinação de Ridge e E09 teve menor erro agregado, mas venceu essa alternativa em apenas dois dos três blocos, contrariando a regra previamente registrada de melhora agregada e em todos os blocos. Também perdeu um bloco frente ao E09 isolado. O menor valor agregado, sozinho, não determinou a decisão.

### E09 por período

| Avaliação | Climatologia E00 | Memória E07 | Pressão e memória E09 |
|---|---:|---:|---:|
| 2017-2018 | 1,853953 | 1,839644 | 1,838627 |
| 2019-2020 | 1,848451 | 1,810902 | 1,810317 |
| 2021-2022 | 1,872027 | 1,844967 | 1,844014 |

O acréscimo de pressão reduziu o RMSE agregado em cerca de 0,0466% frente ao E07. É um incremento pequeno, observado nos três blocos. A redução de 1,46% se refere à comparação mais ampla com a climatologia, não ao efeito isolado desses três atributos.

## Regionalização: o que foi testado

E09 já varia por localização, climatologia e estado atmosférico. E11 adicionou explicitamente três categorias: UF, bioma e vegetação de referência. A versão com contexto geográfico teve menor RMSE em **17 das 27 UFs**, mas piorou nas outras dez e no agregado da grade. Isso não identifica a contribuição causal de cada categoria.

Os mapas ambientais são estáticos e retrospectivos. Vegetação potencial/de referência não equivale à cobertura atual nem a uma série mensal de desmatamento. A disponibilidade binária dos mapas em cada data histórica não foi certificada. Fronteiras administrativas são máscaras de análise e não fronteiras físicas do clima.

E12 aplica a versão geográfica somente ao Brasil e mantém E09 fora do país. Seus componentes continuam treinados na grade ampla. O RMSE brasileiro caiu de **2,023047 para 2,022463**, mas a melhora ocorreu em apenas **1 de 3 blocos** e **34 de 72 meses**. O exterior foi preservado exatamente. O ganho pequeno e instável não justificou mudar a família final.

E13 treinou de fato especialistas apenas com pontos brasileiros. Os 53 atributos e os parâmetros das árvores foram mantidos. A escolha de 100/300 árvores usou as 11.351 células da máscara brasileira no período interno.

| Receita aplicada no Brasil, com E09 fora | Pontos de treino | RMSE Brasil | RMSE da grade | Blocos brasileiros melhores que E09 |
|---|---:|---:|---:|---:|
| E09 de referência | 2.500 na grade ampla | 2,023047 | 1,831045 | Referência |
| E13: subconjunto brasileiro da amostra original | 366 no Brasil | 2,037200 | 1,833311 | 1/3 |
| E13: mesmo orçamento, concentrado no Brasil | 2.500 no Brasil | 2,041238 | 1,833960 | 0/3 |

O resultado do braço principal é o produto combinado: especialista dentro da máscara brasileira e E09 fora. As previsões exteriores permaneceram idênticas, bit a bit. Essa experiência desfavorece as duas receitas testadas; não refuta toda especialização regional ou todo uso de regimes climáticos.

## Chuvas elevadas

A faixa observada de **pelo menos 20 mm/dia de média mensal** representa aproximadamente 0,61% dos pares célula-mês avaliados. Não se trata de eventos diários isolados.

| Modelo | RMSE nessa faixa | Viés: previsto menos observado |
|---|---:|---:|
| Climatologia | 9,892861 | -6,870146 |
| E09 | 9,776669 | -6,912580 |

O RMSE nessa faixa diminui, mas a subestimação média fica ligeiramente maior. Melhorar o erro agregado não resolve automaticamente os períodos de chuva muito elevada. O diagnóstico mantém ambas as medidas visíveis.

## Artefato final e reprodução

E14 é o reajuste da família E09, não uma nova família de modelo. A seleção final usa treino em 1989-2018 e escolha em 2019-2022. Foram escolhidas 100 árvores: RMSE de seleção de 1,827271, contra 1,827494 com 300 árvores. Esses números servem à escolha interna e não são uma nova avaliação independente.

O modelo final é ajustado em 1993-2022, preservando os 2.500 pontos e os 53 atributos. Para construir a memória inicial, utiliza os campos atmosféricos de outubro e novembro de 2022, além das entradas de teste cujo primeiro mês de origem é dezembro de 2022.

O processo verifica datas, grade, ordem dos atributos, hashes das entradas, ausência de valores não finitos e identificação de todas as linhas. A recarga dos artefatos reproduziu exatamente os 24 meses, e a serialização CSV preservou os valores float32.

- Forma de saída: colunas `id,tp_mm_day`, com 1.885.464 linhas.
- SHA-256 do CSV de referência: `6043f690f78ed2ce1993d05a2895eaa217b959858e803836fa7bc9ce05bb4071`.
- Hashes e contrato dos dados: `outputs/contract_audit.json`.
- Amostra congelada: `outputs/E09_spatial_v1/pressure/training_points.npy`.
- A versão de encerramento passou em 49 testes automatizados; os testes verificam propriedades do código, não a capacidade preditiva em um período novo.

## O que fica como investigação futura

Uma próxima avaliação precisaria de um período realmente não usado nas decisões de desenvolvimento. Outras hipóteses possíveis são regimes climáticos com seleção exclusivamente interna, atributos dinâmicos de cobertura do solo e modelos focados na distribuição da chuva. Elas exigiriam novos protocolos, verificação de disponibilidade histórica e controle de comparações. Correlação entre atividade humana e precipitação, sozinha, não demonstra influência causal.

A contribuição deste projeto é uma cadeia verificável entre contrato dos dados, validação temporal, comparadores, engenharia de atributos, decisões regionais e previsão reproduzível. O resultado inclui as hipóteses que falharam, porque elas também orientam a escolha do modelo.
