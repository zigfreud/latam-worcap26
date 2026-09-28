# Dados oficiais

Obtenha o pacote na página da competição: https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data

Extraia os arquivos em `data/raw/`, na raiz do projeto. Para o ajuste final:

- `treino_tp.nc`
- `treino_t2.nc`
- `treino_cloud_cover.nc`
- `treino_shum_850.nc`
- `treino_surface_pressure.nc`
- `treino_u_850.nc`
- `treino_v_850.nc`
- `treino_temperature_850.nc`
- `treino_rel_hum_850.nc`
- `treino_geopotential_850.nc`
- `teste_features.nc`
- `sample_submission.csv`

O pacote original também inclui `treino_tp_alvo.nc`; o ajuste final calcula o alinhamento temporal a partir de `treino_tp.nc`, sem depender desse arquivo auxiliar.

`outputs/contract_audit.json` contém os hashes e o contrato numérico do pacote de referência. O script de ajuste interrompe a execução se as entradas forem diferentes. Não substitua hashes para contornar divergências: verifique primeiro a versão dos dados.

Para repetir inferência de um modelo já ajustado, bastam `teste_features.nc` e `sample_submission.csv`. Os NetCDF e credenciais ficam fora do Git. Dados e derivados estão sujeitos aos termos de suas fontes.
