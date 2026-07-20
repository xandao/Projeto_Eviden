# Guia descrevendo os arquivos de configuração

## Descrição do arquivo de configuração <span style="color:red; font-weight: bold;">system_config.json</span>:

```json
{
  "predictors_path": "predictors",
  "templates_path": "templates",
  "dataset_path": "data",
  "predictors_info_config_filename": "predictors_info.json"
}
```

Campos (os caminhos relativo são em relação ao diretório preincipal em que estão o treinadir e o otimizador):

- <span style="color:blue; font-weight: bold">"predictors_path"</span>: Caminho relativo em que são armazenados os preditores gerados pelo treinador e usados pelo otimizador.
- <span style="color:blue; font-weight: bold;">"templates_path"</span>: Caminho relativo em que são armazenados os arquivos com os templates dos scripts de submissão de cada aplicação.
- <span style="color:blue; font-weight: bold;">"dataset_path"</span>: Caminho relativo em que são armazenadas as bases de dados usadas nos treinamentos.
  - <span style="color:red; font-weight: bold;">TODO</span>: Será que deveríamos ter um diretório para cada aplicação ao invés de colocar todos os arquivos em um mesmo diretório? Poderíamos usar o nome da aplicação como o nome do diretório.
- <span style="color:blue; font-weight: bold">"predictors_info_config_filename"</span>: Nome do arquivo de configuração JSON relacionando cada aplicação ao arquivo com o seu preditor treinado, usando o nome da aplicação como chave para descobrir o preditor correto.

## Descrição do arquivo de configuração <span style="color:red; font-weight: bold;">training_config.json</span>:

```json
{
  "filter": {
    "outlier_limit": 100.0
  },
  "models": {
    "ExtraTreesRegressor": {
      "grid_search_parms": {
        "max_depth": [5, 10, 15, null],
        "n_estimators": [10, 20, 50, 100, 120, 150]
      },
      "fixed_params": {
        "random_state": 42
      },
      "name": "ETR",
      "import_path": "sklearn.ensemble.ExtraTreesRegressor"
    },
    "GradientBoostingRegressor": {
      "grid_search_parms": {
        "max_depth": [5, 10, 15, null],
        "n_estimators": [10, 20, 50, 100, 120, 150]
      },
      "fixed_params": {
        "random_state": 42
      },
      "name": "GBTR",
      "import_path": "sklearn.ensemble.GradientBoostingRegressor"
    },
    "RandomForestRegressor": {
      "grid_search_parms": {
        "max_depth": [5, 10, 15, null],
        "n_estimators": [10, 20, 50, 100, 120, 150]
      },
      "fixed_params": {
        "random_state": 42
      },
      "name": "RFR",
      "import_path": "sklearn.ensemble.RandomForestRegressor"
    },
    "DecisionTreeRegressor": {
      "grid_search_parms": {
        "max_depth": [5, 10, 15, null]
      },
      "fixed_params": {
        "random_state": 42
      },
      "name": "DTR",
      "import_path": "sklearn.tree.DecisionTreeRegressor"
    }
  }
}
```
