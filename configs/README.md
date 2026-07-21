# Guia dos arquivos de configuração

## Exemplo do arquivo de configuração $\color{red}\textbf{system\_config.json}$

```jsonv
{
  "predictors_path": "predictors",
  "templates_path": "templates",
  "dataset_path": "data",
  "applications_path": "applications",
  "predictors_info_config_filename": "predictors_info.json"
}
```

Campos (os caminhos relativo são em relação ao diretório preincipal em que estão o treinadir e o otimizador) do objeto do sistema:

- $\color{blue}\textbf{predictors\_path}$: Caminho relativo em que são armazenados os preditores gerados pelo treinador e usados pelo otimizador.
- $\color{blue}\textbf{templates\_path}$: Caminho relativo em que são armazenados os arquivos com os templates dos scripts de submissão de cada aplicação.
- $\color{blue}\textbf{dataset\_path}$: Caminho relativo em que são armazenadas as bases de dados usadas nos treinamentos.
  - $\color{red}\textbf{TODO}$: Será que deveríamos ter um diretório para cada aplicação ao invés de colocar todos os arquivos em um mesmo diretório? Poderíamos usar o nome da aplicação como o nome do diretório.
- $\color{blue}\textbf{applcations\_path}$: Caminho relativo em que são armazenados os arquivos de configuração das aplicações.
- $\color{blue}\textbf{predictors\_info\_config\_filename}$: Nome do arquivo de configuração JSON relacionando cada aplicação ao arquivo com o seu preditor treinado, usando o nome da aplicação como chave para descobrir o preditor correto.

## Exemplo do arquivo de configuração $\color{red}\textbf{training\_config.json}$

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

Existem dois campos principais, $\color{blue}\textbf{filter}$ e $\color{blue}\textbf{models}$. O primeiro campo, $\color{blue}\textbf{filter}$, tem as informações para a parte do treinamento em que ocorre a filtragem da base de dados usada ao treinar o modelo:

- $\color{blue}{\textbf{outlier\_limit}}$: Valor real indicando como a faixa de exclusão para uma das características da base de dados é usada, sendo a faixa para a variável igual a $[m_a-outlier\_limit,m_a+outlier\_limit]$ onde $m_a$ é a mediana absoluta da couluna associada à característica na base de dados. As colunas usadas para a filtragrem dependerão da aplicação cujo modelo estamos treinando.

Já o segundo campo, $\color{blue}\textbf{models}$, tem as informações para cada modelo treinado, sendo este modelo idenficidado por um campo, o seu nome, no objeto JSON definido pelo campo $\color{blue}\textbf{models}$. Todos os modelos tem a mesma entrada, também um objeto, cujos os campos são os seguintes:

- $\color{blue}\textbf{grid\_search\_parms}$: Hiperparâmetros do modelo usados pela fase de otimização dos hiperparâmertros do modelo. É um objeto em que cada entrada é o hiperparâmetro do modelo, como usado quando o modelo é inicializado no Python, e o valor da entrada é um vetor com todos os valores a serem avaliados do hiperparâmetro. Por exemplo, para o modelo $\color{green}\textbf{ExtraTreesRegressor}$:
  - $\color{blue}\textbf{max\_depth}$: Profundidade de cada árvore da floresta gerada pelo modelo. Os possíveis valores são 5, 10, 15 ou ilimitado ($\color{gray}\textbf{null}$).
  - $\color{blue}\textbf{n\_estimators}$: Número de estimadores, ou seja, de árvores na floresta gerada pelo modelo. Os possíveis valores são 10, 20, 50, 100, 120 e 150.

- $\color{blue}\textbf{fixed\_parms}$: Hiperparâmetros fixos do modelo É um objeto em que cada entrada é o hiperparâmetro do modelo, como usado quando o modelo é inicializado no Python, e o valor da entrada é o valor para o hiperparâmetro. Por exemplo, para o modelo $\color{green}\textbf{ExtraTreesRegressor}$:
  - $\color{blue}\textbf{random\_state}$: Fixa a semente randômica. Durante o processamento do modelo, escolhas randômicas são feitas, como definido pelo [algoritmo do modelo][1]. O valor fixo, no caso 42, garante que ao treinar o modelo sempre pobteremos o mesmo estimador.

- $\color{blue}\textbf{name}$: Nome alternativo para o modelo. No caso do modelo $\color{green}\textbf{ExtraTreesRegressor}$, o nome é ``ETR''.
- $\color{blue}\textbf{import\_path}$: Caminho completo de importação dinâmica do modelo no Python, ou seja, se a importaçaõ for `from model_library import model` o valor do campo será $``model\_library.model''$. Por exemplo, para o modelo $\color{green}\textbf{ExtraTreesRegressor}$, o caminho é $``sklearn.ensemble.ExtraTreesRegressor''$, porque o comando usado ao importar o modelo no Python é `from sklearn.ensemble import ExtraTreesRegressor`

## Arquivo de configuração da aplicação

Existe um arquivo de configuração da aplicação para cada aplicação para a qual treinaremos um modelo. Os arquivos ficam no diretório $\color{darkgray}\textbf{applications}$. Por exemplo, para a aplicação RAxML, o arquivo de configuração é o $\color{red}\textbf{raxml\_config.json}$, descrito a seguir

```json
{
  "suggestions_parameters": ["NNodes", "Processo p/ no", "Thread p/ proc."],
  "application_parameters": ["Bootstrap", "Tamanho"],
  "name": "raxml",
  "estimated_parameters": {
    "suggestion": "EDP",
    "time": "ElapsedRaw"
  },
  "training": {
    "group_parameters": ["Bootstrap", "Tamanho"],
    "filter_parameters": ["ElapsedRaw", "Consumo de Energia Total (J)", "EDP"],
    "dataset_files": ["raxml.csv"]
  },
  "user": {
    "executable_names": ["raxmlHPC-PTHREADS-AVX-omp", "raxml"],
    "script_template_name": "raxml_template.sh",
    "suggestions_map": {
      "nodes": "NNodes",
      "process": "Processo p/ no",
      "threads": "Thread p/ proc."
    },
    "user_options": {
      "Bootstrap": {
        "params": ["-N", "-#"],
        "type": "integer",
        "help": "Bootstrap number."
      },
      "Arquivo": {
        "params": ["-s"],
        "type": "string",
        "help": "Input file."
      }
    },
    "conversions": {
      "Bootstrap": ["copy", "Bootstrap"],
      "Tamanho": ["filesize", "Arquivo"]
    },
    "slurm": [
      {
        "partition": "sequana_cpu_dev",
        "max_time": 1200,
        "max_memory": 367001600,
        "exclusive": true,
        "default": true,
        "nodes": 4,
        "process": 2,
        "threads": 48
      },
      {
        "partition": "sequana_cpu",
        "max_time": 3600,
        "max_memory": 367001600,
        "exclusive": true,
        "default": false,
        "nodes": 20,
        "process": 2,
        "threads": 48
      }
    ]
  }
}
```

[1]: https://doi.org/10.1007/s10994-006-6226-1
