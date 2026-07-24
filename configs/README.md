# Guia dos arquivos de configuração

## Exemplo do arquivo de configuração $\color{red}\text{\textbf{system\\\_config.json}}$

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

- $\color{blue}\text{\textbf{predictors\\\_path}}$: Caminho relativo em que são armazenados os preditores gerados pelo treinador e usados pelo otimizador.
- $\color{blue}\text{\textbf{templates\\\_path}}$: Caminho relativo em que são armazenados os arquivos com os templates dos scripts de submissão de cada aplicação.
- $\color{blue}\text{\textbf{dataset\\\_path}}$: Caminho relativo em que são armazenadas as bases de dados usadas nos treinamentos.
  - $\color{red}\text{\textbf{TODO}}$: Será que deveríamos ter um diretório para cada aplicação ao invés de colocar todos os arquivos em um mesmo diretório? Poderíamos usar o nome da aplicação como o nome do diretório.
- $\color{blue}\text{\textbf{applcations\\\_path}}$: Caminho relativo em que são armazenados os arquivos de configuração das aplicações.
- $\color{blue}\text{\textbf{predictors\\\_info\\\_config\\\_filename}}$: Nome do arquivo de configuração JSON relacionando cada aplicação ao arquivo com o seu preditor treinado, usando o nome da aplicação como chave para descobrir o preditor correto.

## Exemplo do arquivo de configuração $\color{red}\text{\textbf{training\\\_config.json}}$

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

Existem dois campos principais, $\color{blue}\text{\textbf{filter}}$ e $\color{blue}\text{\textbf{models}}$. O primeiro campo, $\color{blue}\text{\textbf{filter}}$, tem as informações para a parte do treinamento em que ocorre a filtragem da base de dados usada ao treinar o modelo:

- $\color{blue}{\text{\textbf{outlier\\\_limit}}}$: Valor real indicando como a faixa de exclusão para uma das características da base de dados é usada, sendo a faixa para a variável igual a $[m_a-outlier\\\_limit,m_a+outlier\\\_limit]$ onde $m_a$ é a mediana absoluta da couluna associada à característica na base de dados. As colunas usadas para a filtragrem dependerão da aplicação cujo modelo estamos treinando.

Já o segundo campo, $\color{blue}\text{\textbf{models}}$, tem as informações para cada modelo treinado, sendo este modelo idenficidado por um campo, o seu nome, como uma chave do objeto JSON definido pelo campo $\color{blue}\text{\textbf{models}}$. Todos os modelos tem a mesma entrada, também um objeto, cujos os campos são os seguintes:

- $\color{blue}\text{\textbf{grid\\\_search\\\_parms}}$: Hiperparâmetros do modelo usados pela fase de otimização dos hiperparâmertros do modelo. É um objeto em que cada entrada é o hiperparâmetro do modelo, como usado quando o modelo é inicializado no Python, e o valor da entrada é um vetor com todos os valores a serem avaliados do hiperparâmetro. Por exemplo, para o modelo $\color{green}\text{\textbf{ExtraTreesRegressor}}$:
  - $\color{blue}\text{\textbf{max\\\_depth}}$: Profundidade de cada árvore da floresta gerada pelo modelo. Os possíveis valores são 5, 10, 15 ou ilimitado ($\color{gray}\text{\textbf{null}}$).
  - $\color{blue}\text{\textbf{n\\\_estimators}}$: Número de estimadores, ou seja, de árvores na floresta gerada pelo modelo. Os possíveis valores são 10, 20, 50, 100, 120 e 150.

- $\color{blue}\text{\textbf{fixed\\\_parms}}$: Hiperparâmetros fixos do modelo É um objeto em que cada entrada é o hiperparâmetro do modelo, como usado quando o modelo é inicializado no Python, e o valor da entrada é o valor para o hiperparâmetro. Por exemplo, para o modelo $\color{green}\text{\textbf{ExtraTreesRegressor}}$:
  - $\color{blue}\text{\textbf{random\\\_state}}$: Fixa a semente randômica. Durante o processamento do modelo, escolhas randômicas são feitas, como definido pelo [algoritmo do modelo][1]. O valor fixo, no caso $\color{gray}\text{\textbf{42}}$, garante que ao treinar o modelo sempre obteremos o mesmo estimador.

- $\color{blue}\text{\textbf{name}}$: Nome alternativo para o modelo. No caso do modelo $\color{green}\text{\textbf{ExtraTreesRegressor}}$, o nome é $\color{gray}\text{\textbf{ETR}}$.
- $\color{blue}\text{\textbf{import\\\_path}}$: Caminho completo de importação dinâmica do modelo no Python, ou seja, se a importação for `from model_library import model` o valor do campo será $\color{gray}\text{\textbf{model\\\_library.model}}$. Por exemplo, para o modelo $\color{green}\text{\textbf{ExtraTreesRegressor}}$, o caminho é $\color{gray}\text{\textbf{sklearn.ensemble.ExtraTreesRegressor}}$, porque o comando usado ao importar o modelo no Python é `from sklearn.ensemble import ExtraTreesRegressor`

## Arquivo de configuração da aplicação

Existe um arquivo de configuração da aplicação para cada aplicação para a qual treinaremos um modelo. Os arquivos ficam no diretório $\color{darkgray}\text{\textbf{applications}}$.

Por exemplo, para a aplicação RAxML, o arquivo de configuração é o $\color{red}\text{\textbf{raxml\\\_config.json}}$, dado a seguir. Alguns parâmetros precisam ser convertidos e neste caso o nome do arquivo é convertido para o tamanho deste arquivo.

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

Por exemplo, para as aplicações do NAS, o arquivo de configuração é o $\color{red}\text{\textbf{ntb\\\_config.json}}$, dAdo a seguir. Os parâmetros precisam ser maeados e, neste caso, o par $\color{gray}\text{\textbf{Benchmark}}$, $\color{gray}\text{\textbf{Classe}}$ é convertido para os seis valores correspondentes $\color{gray}\text{\textbf{Zone X}}$, $\color{gray}\text{\textbf{Zone Y}}$, $\color{gray}\text{\textbf{Iterações}}$, $\color{gray}\text{\textbf{Iterações}}$, $\color{gray}\text{\textbf{Grid X}}$, $\color{gray}\text{\textbf{Grid Y}}$ e $\color{gray}\text{\textbf{Grid Z}}$, usando o arquivo $\color{gray}\text{\textbf{ntb\\\_map\\\_table.cvs}}$, com uma tabela que mapeia o par nos seis valores corrrespondentes.

```json
{
  "suggestions_parameters": ["NNodes", "Processo p/ no", "Thread p/ proc."],
  "application_parameters": [
    "Zone X",
    "Zone Y",
    "Iterações",
    "Grid X",
    "Grid Y",
    "Grid Z"
  ],
  "name": "ntb",
  "estimated_parameters": {
    "suggestion": "EDP",
    "time": "ElapsedRaw"
  },
  "training": {
    "group_parameters": ["Benchmark", "Classe"],
    "filter_parameters": ["ElapsedRaw", "Consumo de Energia Total (J)", "EDP"],
    "dataset_files": [
      "ntb_1.csv",
      "ntb_2.csv",
      "ntb_3.csv",
      "ntb_4.csv",
      "ntb_5.csv",
      "ntb_6.csv"
    ]
  },
  "user": {
    "executable_names": ["ntb.sh"],
    "script_template_name": "ntb_template.sh",
    "predictor_name": "ntb",
    "suggestions_map": {
      "nodes": "NNodes",
      "process": "Processo p/ no",
      "threads": "Thread p/ proc."
    },
    "user_options": {
      "Benchmark": {
        "params": ["-b"],
        "type": "string",
        "help": "Benchmark name."
      },
      "Classe": {
        "params": ["-c"],
        "type": "string",
        "help": "Class name."
      }
    },
    "conversions": {
      "Zone X": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"],
      "Zone Y": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"],
      "Iterações": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"],
      "Grid X": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"],
      "Grid Y": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"],
      "Grid Z": ["map", "ntb_map_table.cvs", "Benchmark", "Classe"]
    },
    "slurm": [
      {
        "partition": "sequana_cpu_dev",
        "max_time": 1200,
        "max_memory": 367001600,
        "exclusive": true,
        "default": false,
        "nodes": 4,
        "process": 2,
        "threads": 48
      },
      {
        "partition": "sequana_cpu",
        "max_time": 3600,
        "max_memory": 367001600,
        "exclusive": true,
        "default": true,
        "nodes": 20,
        "process": 2,
        "threads": 48
      }
    ]
  }
}
```

No arquivo de configuração de cada aplicação, existem seis campos principais, descritos a seguir:

- $\color{blue}\text{\textbf{suggestions\\\_parameters}}$: Características que representam os parâmetros do conjunto de dados que serão sugeridos pelo script que os usuários executarão para executar as suas aplicações de modo otimizado. No arquivo de configuração do RAxML, estas características são $\color{gray}\text{\textbf{NNodes}}$ (número de nós), $\color{gray}\text{\textbf{Processo p/ no}}$ (processos por nó) e $\color{gray}\text{\textbf{Thread p/ proc.}}$ (threads por processo).

- $\color{blue}\text{\textbf{application\\\_parameters}}$: Características que representam os parâmetros do conjunto de dados que serão fornecidos pelos usuários executarão ao executar as suas aplicações de modo otimizado. Para o caso do RAxML, o usuário fornece o $\color{gray}\text{\textbf{Bootstrap}}$ e o nome do arquivo, que irá gerar a outra característica, a $\color{gray}\text{\textbf{Tamanho}}$, o tamanho deste arquivo (veja a seguir o campo de conversão $\color{blue}\text{\textbf{conversions}}$).
- $\color{blue}\text{\textbf{name}}$: Um nome alternativo para a aplicação.
- $\color{blue}\text{\textbf{estimated\\\_parameters}}$: Objeto que define as variáveis alvo a serem treinadas no modelo. São dois campos:
  - $\color{blue}\text{\textbf{suggestion}}$: Variável alvo para o preditor usado para fazer as sugestões aos usuários. É a variável usada ao otimizar os hiperparâmetros dos modelos descritos no arquivo de configuração $\color{red}\text{\textbf{training\\\_config.json}}$ usando a busca em grade, a escolha escolha do melhor modelo usando a validação cruzada. FInalmente, o melhor modelo é treinado usando os melhores huperparâmetros e toda a base de dados. No caso do RAxML o alvo é a variavél $\color{gray}\text{\textbf{EDP}}$
  - $\color{blue}\text{\textbf{time}}$: Variável alvo para o preditor usado para predizer o tempo de execução após a escolha da melhor configuração para o usuário. O modelo treinado é o melhor modelo escolhido após a validação cruzada, usando os melhores huperparâmetros, e toda a base de dados. No caso da predição do tempo, a variável alvo é a $\color{gray}\text{\textbf{ElapsedRaw}}$.
- $\color{blue}\text{\textbf{training}}$: Objeto que define as variáveis usadas durante os passos que requerem treinamento de algum estimador, que ocorre durante a escolhe dos melhores hiperparâmetros, do melhor modelo e do treinamento do melhor modelo com todos os dados, para fazer a sugestão e o tempo estimado para a melhor sugestão. Também tem os arquivos do conjunto de dados a serem usados nestas fases e a filtragem. São três campos:
  - $\color{blue}\text{\textbf{group\\\_parameters}}$: Lista com as característcas usadas pelo tẽcnica LOGO usada durante os treinamentos para dividir o conjunto de dados em grupos de acordo com todas as combinações dos possíveis valores diferentes dessas características. Durante a busta de grade e a validação cruzada, são considerados $n$ testes onde$n$ é o número de grupos diferentes, sendo que em cada teste um grupo é usado para os dados de teste e os restantes para os dados de treinamento. O objetivo aqui é sempre usar um subconjunto do conjunto de características fornecidas pelo usuário e presente no conjunto de dados da aplicação. Para o arquivo da configuração do RAxML de exemplo, as características são o $\color{gray}\text{\textbf{Bootstrap}}$ e $\color{gray}\text{\textbf{Tamanho}}$
  - $\color{blue}\text{\textbf{filter\\\_parameters}}$: Lista das características usadas ao filtrar o conjunto de dados de entrada para remover os _outliers_ como descrito anteriormente no campo $\color{blue}\text{\textbf{filter}}$ do arquivo de configuração $\color{red}\text{\textbf{training\\\_config.json}}$. No caso do RAxML, os campos são $\color{gray}\text{\textbf{ElapsedRaw}}$, $\color{gray}\text{\textbf{Consumo de Energia Total (J)}}$ e $\color{gray}\text{\textbf{EDP}}$.
  - $\color{blue}\text{\textbf{dataset\\\_files}}$: Lista com os nomes dos arquivos com as bases de dados que serão lidos e concatenados em um único conjunto de dados que será o usado para todas as fases da geração do predior, ou seja, a filtragem, a busca em grade dos hiperparâmetros dos modelos, a escolha do melhor modelo pela validação cruzada e o treinamento dos modelos usados para fazer a sugestão e a estimativa do tempo da melhor sugestão. Os arquivos defem estar no diretório definido pelo campo $\color{blue}\text{\textbf{dataset\\\_path}}$ no arquivo de configuração do sistema $\color{red}\text{\textbf{system\\\_config.json}}$. Neste arquivo de configuração do RAxML, temos somente uma base de dados, no arquivo $\color{gray}\text{\textbf{raxml.csv}}$.
- $\color{blue}\text{\textbf{user}}$: Objeto que define as diversas configurações relacionadas a uma aplicação que são usadas pelo script usado pelos usuários para obter a execução otimizada de uma das possíveis aplicações suportadas. Este script faz esta otimização escolhendo a melhor configuração como descrito no artigo do article [SSCAD][2]. Este objeto tem os seguintes campos:
  - $\color{blue}\text{\textbf{executable\\\_names}}$: Lista com os nomes dos executáveis que podem ser usados pelo usuário do script de otimização ao definir a aplicação. Para esta arquivo de configuração do RAxML, podem uer usados os nomes $\color{gray}\text{\textbf{raxmlHPC-PTHREADS-AVX-omp}}$ e $\color{gray}\text{\textbf{raxml}}$.
  - $\color{blue}\text{\textbf{script\\\_template\\\_name}}$:
  - $\color{blue}\text{\textbf{suggestions\\\_map}}$:
  - $\color{blue}\text{\textbf{user\\\_options}}$:
  - $\color{blue}\text{\textbf{conversions}}$:
  - $\color{blue}\text{\textbf{slurm}}$:

[1]: https://doi.org/10.1007/s10994-006-6226-1
[2]: https://doi.org/10.5753/sscad.2025.16760
