# E05 — candidato local 2023–2024

Protocolo registrado antes da execução. Escopo experimental: treino, inferência e
geração das previsões.

## Receita congelada

- Família E04: climatologia mensal/local + Ridge e LightGBM; combinar as previsões
  não negativas dos componentes. Somente nove variáveis atmosféricas oficiais.
- Seleção: treino dos alvos 1989–2018; validação 2019–2022, grade completa,
  RMSE sem ponderação espacial, incluindo oceano.
- Ridge: alphas 10, 100, 1000, 10000. LightGBM: 100 ou 300 iterações,
  mesmos parâmetros E02, amostra fixa de 2500 pontos, semente 20260916.
- Escolher primeiro os parâmetros de cada componente; depois o peso LightGBM
  em [0, 0.25, 0.5, 0.75, 1], na mesma validação. Empates favorecem o menor
  parâmetro/peso. Não otimizar outros parâmetros após observar os resultados.
- Reajuste final: alvos 1993–2022, com estatísticas sazonais estimadas somente
  nessa janela de 30 anos. Parâmetros/peso ficam congelados antes do reajuste.
- Inferência: alvos janeiro/2023–dezembro/2024. No teste, `time` é o mês-alvo e
  `time_origem` é o mês atmosférico anterior. Conferir ambos antes de renomear
  o eixo para reutilizar a inferência histórica. Ignorar `tp_alvo`, `tp_ultima_obs`
  e `lag_meses`; não usar SST nem chuva futura.

## Evidências e aceitação técnica

Persistir seleção, previsões internas, amostra, dois modelos, pré-processamento,
receita com caminhos relativos, fontes/config/lock, hashes das entradas e saídas.
Recarregar os modelos e reproduzir os 24 campos mensais dos dois componentes.
Verificar todas as linhas do CSV contra o sample: colunas, IDs, ordem, quantidade,
unicidade, valores finitos e não negativos. Serializar com nove algarismos
significativos e conferir recuperação exata das previsões float32.

O conjunto de modelos permite repetir a inferência e gerar o CSV sem depender
dos artefatos E01/E02/E04 nem de novo treinamento.

Não há alvos 2023–2024 disponíveis: não relatar RMSE, posição ou desempenho nesses
anos. Os escores internos foram usados para seleção e não estimam imparcialmente
o desempenho final. A escolha da família E04 também usou os blocos históricos de
desenvolvimento.
