# Protocolo de ajuste final

Esta é a descrição pública da receita de reprodução da família E09. Os registros experimentais originais permanecem nos artefatos locais da pesquisa.

## Escolha da família

E09 combina LightGBM residual, memória atmosférica de três meses e contexto de pressão (53 atributos). Seu critério histórico exigia menor RMSE agregado e melhora em três blocos frente ao E07. A especialização E13 somente substituiria o componente brasileiro se melhorasse Brasil, grade global e os três blocos brasileiros. Não passou; a família E09 foi mantida.

## Seleção e reajuste

- Candidatos: 100 ou 300 árvores.
- Treino interno: 1989-2018; seleção: 2019-2022.
- Reajuste: 1993-2022, preservando amostra, semente e demais parâmetros.
- Amostra: 2.500 células da grade, semente 20260916, índices congelados no repositório.
- Cada janela de 30 anos produz 900.000 exemplos de treino.

A seleção interna não é um teste independente. As avaliações históricas 2017-2022 foram reutilizadas no desenvolvimento.

## Inferência e verificações

Cada alvo mensal de 2023-2024 recebe apenas as condições anteriores fornecidas no teste. Não usar os alvos de precipitação desse período no ajuste. A atmosfera é atualizada mensalmente.

Salvar modelo, pré-processamento, cauda atmosférica outubro/novembro de 2022, receita, código e hashes. Conferir sobreposição dezembro/2022, continuidade das memórias e recarga exata de todos os 24 meses.

Exportar CSV com 1.885.464 IDs na ordem do sample, colunas id,tp_mm_day, valores finitos e não negativos, precisão float32 preservada. Dados brutos são obtidos na fonte; não estão versionados.
