# E02 — protocolo anterior à execução real

Hipótese: uma correção não linear compartilhada entre locais pode superar E00/E01.
Comparação de pipelines: E02 também muda compartilhamento espacial e amostragem;
não é uma ablação isolada da não linearidade.

- Alvo: média mensal da precipitação em mm/dia, M+1, mesma grade e pesos do E00.
- Previsão: climatologia mensal local de 30 anos + correção LightGBM; mínimo zero.
- Um modelo global por bloco. Entradas: nove anomalias atmosféricas locais em M,
  latitude, longitude, seno e cosseno do mês ALVO, climatologia da chuva do mês ALVO.
  Total de 14 atributos. Sem chuva recente, ano, dados externos ou campos vizinhos.
- Médias sazonais atmosféricas e climatologia ajustadas somente no treino de cada etapa.
  Sem padronização por desvio-padrão: árvores fazem cortes por limiares.
- Treino do corretor em 2500 pontos sorteados uniformemente sem reposição, seed 20260916,
  mesmos pontos em todas as etapas, todos os 360 meses de cada janela: 900000 linhas.
  A amostra reduz custo local. Climatologias calculadas na grade completa; métricas
  internas e externas sempre na grade completa, incluindo oceanos, sem ponderação de área.
- LightGBM CPU, erro quadrático, learning_rate 0.05, 31 folhas, profundidade máxima 8,
  mínimo 500 linhas por folha, lambda_l2 10, max_bin 127, quatro threads,
  deterministic=true e force_col_wise=true. Parâmetros históricos de referência descritos acima.
- Única seleção: 100 ou 300 árvores pelo menor RMSE das previsões finais truncadas,
  nos quatro anos anteriores a cada teste externo. Empate favorece 100 árvores.
  Candidatos avaliados nos prefixos de uma mesma sequência treinada de 300 árvores.
- Validações internas: 2013–2016, 2015–2018, 2017–2020; cada uma com 30 anos de
  treino anteriores. Reajuste com número escolhido nas janelas externas do E00/E01.
- Testes externos: 2017–2018, 2019–2020, 2021–2022; nenhuma escolha pelos seus resultados.
- Não fazer early stopping no teste externo nem ajuste pós-hoc.
- Salvar modelos, transformações, amostra, previsões, métricas por mês/bloco/local,
  versões, duração, hashes de entradas/fontes e cópia do protocolo/configuração/fontes.
- Verificar RMSE lendo previsões salvas e reprodução da primeira grade de cada bloco
  a partir do modelo e das transformações recarregados do disco.
- Reportar ganhos/perdas frente a E00 e E01, inclusive resultado negativo. Mesmos blocos
  são evidência de desenvolvimento, não confirmação independente da escolha entre E00/E01/E02.
  Não declarar significância, causalidade ou ranking a partir desta comparação.
