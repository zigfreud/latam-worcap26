# E04 — combinação Ridge + LightGBM sem SST

Protocolo registrado antes de calcular o desempenho da combinação.

Hipótese: os erros da Ridge local e do LightGBM global sem SST se complementam;
uma combinação convexa pode melhorar a previsão histórica.

- Componentes: E01 (`outputs/E01_ridge_local_v1`) e E02 original sem SST
  (`outputs/E02_lightgbm_global_v1`). Não incluir E03, chuva recente ou fonte externa.
- Previsão = (1−w) × previsão Ridge + w × previsão LightGBM. Combinar as previsões
  finais já truncadas em zero. A soma convexa continua não negativa; salvar float32.
- Candidatos w = 0, 0.25, 0.5, 0.75, 1; um peso global por bloco, sem pesos regionais
  ou mensais. Empate favorece o menor w (Ridge). Os extremos permitem não combinar.
- Reconstituir as previsões INTERNAS da Ridge com o alpha originalmente escolhido
  e do LightGBM com o número de árvores originalmente escolhido. Esses parâmetros
  foram escolhidos nos quatro anos anteriores ao respectivo teste externo.
- Treino/normalização internos: 30 anos anteriores a esses quatro anos. Reutilizar
  o modelo interno LightGBM salvo; recomputar suas transformações com o treino original.
  Reajustar a Ridge interna porque seus parâmetros internos não foram persistidos.
- Exigir que os RMSEs internos reconstruídos reproduzam os valores originais de
  selection.json (tolerância absoluta 1e-7) antes de escolher o peso.
- Escolher w pelo menor RMSE interno da combinação float32, todos os pontos/meses.
  Usar a mesma validação interna para parâmetros dos componentes e peso é uma seleção
  sequencial de hiperparâmetros. Não é avaliação independente dentro dessa janela.
- Congelar o peso antes de abrir as previsões externas do respectivo bloco no runner.
  Aplicá-lo às previsões externas já existentes dos componentes reajustados no passado.
- Testes externos: 2017–2018, 2019–2020 e 2021–2022, sem usar seus alvos na seleção.
  Reproduzir RMSEs externos E01/E02; verificar mesma grade, datas e entrada de chuva.
- Salvar previsões internas dos componentes, candidatos/peso, previsões combinadas,
  receita do ensemble, hashes e caminhos dos modelos componentes, métricas, fontes e config.
  A receita depende dos diretórios E01/E02 preservados: não é um modelo autocontido.
- Verificar previsão salva, RMSE e primeira grade mensal externa reconstruída com
  os modelos componentes carregados do disco e o peso persistido na receita.
- Reportar E04 vs E00/E01/E02 por bloco/mês/ponto. E03 pode aparecer no histórico
  com a ressalva de SST revisada, mas não entra na escolha de pesos nem componentes.
- Sem novos pesos após ver resultados; sem alegação de significância estatística.
  Blocos externos reutilizados para desenvolver E00–E04 não são confirmação independente
  do processo de escolha da família do modelo. Resultado negativo também será preservado.
