# E09 — contexto espacial e pressão bruta

Congelado antes de treinar os novos braços. Avaliação de desenvolvimento.

## Perguntas e braços

Controle: previsões e modelos congelados de E07/memory3 (50 atributos), reutilizados,
sem novo ajuste. Verificar hashes, amostra de pontos, métricas e equivalência dos
50 atributos originais. Não há pretensão de confirmação independente.

- spatial: acrescentar nove médias locais 5 × 5 das anomalias atmosféricas em T−1.
- pressure: acrescentar pressão de superfície bruta T−1, sua média 5 × 5 e
  diferença ponto menos média (três atributos). Não acrescenta o braço spatial.

Janela centrada, inclui o ponto central, pesos iguais; bordas usam somente células
existentes, sem wrap/reflexão. Na grade 0,25°, centros extremos distam 1°; distância
física varia com latitude. Nenhum alvo de chuva da vizinhança é utilizado.
Anomalias calculadas usando climatologia atmosférica mensal ajustada apenas no treino.
As primeiras 50 colunas e o alvo residual permanecem exatamente iguais ao E07.

Pressão bruta não é altitude. As unidades/proveniência das nove variáveis continuam
ausentes no pacote local. Não aplicar limiar 85000 nem máscara física de 850 hPa.
Este braço testa contexto numérico associado a estado médio e variações de pressão;
não separa relevo de meteorologia. DEM e máscara de níveis ficam pendentes de fonte.

## Comparabilidade e decisão

Mesmos 2500 pontos, seed 20260916, parâmetros LightGBM, memória 3 e candidatos
100/300 árvores do E07. Seleção interna nos quatro últimos anos, treino de 30 anos.
Blocos externos 2017–18, 2019–20 e 2021–22; avaliar os 78561 pontos por mês.
Gate para avançar: RMSE agregado menor que E07 e melhora nos três blocos.
Relatar meses, mapas de RMSE/bias, regiões e faixa observada >=20 mm/dia,
inclusive se contradisserem a melhora global. Não selecionar escala após os resultados.
Não alterar E05, E07 ou E08. Não gerar novo candidato final neste bloco.

## Verificações

Testes de bordas, isolamento temporal, igualdade treino/inferência e serialização.
Releitura das previsões e primeira grade mensal de cada modelo recarregado.
Registrar configs, código, hashes de entradas e preservação dos experimentos anteriores.

## Trilha humana, separada

Nesta etapa fazer triagem de cobertura e disponibilidade de duas fontes oficiais:
IBGE PIM-PF (automóveis) e ANP (refino/processamento). Guardar metadados, séries
e hashes quando acessíveis. Referência econômica não equivale à emissão local.
Não interpretar correlação causal, nem anexar séries com cobertura menor ao backtest
de 30 anos. Se houver série utilizável, registrar protocolo próprio antes de calcular
correlações/ganhos, com janela comum e defasagem de divulgação. Datas históricas de
publicação e versões precisam ser auditadas; revisões atuais não provam disponibilidade.
