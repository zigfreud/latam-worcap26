# E01 — protocolo anterior à execução

Hipótese: anomalias atmosféricas locais do mês M permitem corrigir a climatologia
da precipitação do mês M+1. Esta é uma primeira regressão local linear; um resultado
negativo não elimina informação espacial, não linear ou oceânica.

- Mesma grade, alvo, pesos e três blocos do E00; nenhuma amostragem espacial.
- Uma Ridge por ponto, compartilhando coeficientes entre os 12 meses do ano.
- Entradas: as nove variáveis atmosféricas em M. Sem precipitação como atributo.
- Sazonalidade: climatologias mensais locais removidas de X e y no treino.
- X padronizado pela dispersão das anomalias, ajustada somente no treino.
- Localização: cada ponto possui seu próprio modelo. Sem interação explícita entre vizinhos.
- Janela: 30 anos de meses ALVO; as entradas são exatamente um mês anteriores.
- Alpha global por bloco escolhido entre 10, 100, 1000 e 10000 por RMSE nos quatro
  anos anteriores ao bloco externo. O treino interno também usa 30 anos anteriores.
- Seleção interna: 2013–2016, 2015–2018 e 2017–2020, respectivamente.
- Reajuste após escolher alpha: 1987–2016, 1989–2018 e 1991–2020.
- Não escolher parâmetros pelos resultados externos nem pelo ranking público.
- Previsão: climatologia do mês alvo + correção Ridge, limitada inferiormente a zero.
- Reportar RMSE agregado, por bloco e mês, viés, mapas de ganho e alpha escolhido.
- Salvar predições externas e parâmetros, versões, hashes e cópia do código.
- Critério descritivo: melhora agregada e consistência entre blocos. Não declarar
  significância estatística ou causalidade com esta comparação inicial.

A climatologia/normalização pode usar todos os anos do respectivo treino, inclusive
para transformar suas linhas de treinamento. Nenhum ano de validação entra nesses ajustes.
Nos blocos posteriores é legítimo treinar com anos que já seriam passado nessa data.
