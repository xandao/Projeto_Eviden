# Guia dos arquivos de configuração

## Exemplo do arquivo de configuração $\color{red}\text{\textbf{system\\\_config.json}}$

```json
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

Existe um arquivo de configuração da aplicação para cada aplicação para a qual treinaremos um modelo. Os arquivos ficam no diretório $\color{gray}\text{\textbf{applications}}$.

Por exemplo, para a aplicação RAxML, o arquivo de configuração é o $\color{red}\text{\textbf{raxml\\\_config.json}}$, dado a seguir. O parâmetro $\color{gray}\text{\textbf{Arquivo}}$, que dá o caminho do arquivo, precisa ser convertido o tamanho deste arquivo.

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

Arquivo de configuração do NTB (benckmarks do NAS) $\color{red}\text{\textbf{ntb\\\_config.json}}$ (dado depois da tabela). A conversão mapeia cada possível valor dos parâmetros $\color{gray}\text{\textbf{Benchmark}}$ e $\color{gray}\text{\textbf{Classe}}$ nas características $\color{gray}\text{\textbf{Zone X}}$, $\color{gray}\text{\textbf{Zone Y}}$, $\color{gray}\text{\textbf{Iteração}}$, $\color{gray}\text{\textbf{Grix X}}$, $\color{gray}\text{\textbf{Grid Y}}$ e $\color{gray}\text{\textbf{Grid Z}}$, usando a tabela a seguir:

| Benchmark | Classe | Zone X | Zone Y | Iterações | Grid X | Grid Y | Grid Z |
| --------- | ------ | ------ | ------ | --------- | ------ | ------ | ------ |
| bt-mz     | A      | 4      | 4      | 200       | 128    | 128    | 16     |
| bt-mz     | B      | 8      | 8      | 200       | 304    | 208    | 17     |
| bt-mz     | C      | 16     | 16     | 200       | 480    | 320    | 28     |
| bt-mz     | D      | 32     | 32     | 500       | 1632   | 1216   | 34     |
| lu-mz     | A      | 4      | 4      | 250       | 128    | 128    | 16     |
| lu-mz     | B      | 4      | 4      | 250       | 304    | 208    | 17     |
| lu-mz     | C      | 4      | 4      | 250       | 480    | 320    | 28     |
| lu-mz     | D      | 4      | 4      | 300       | 1632   | 1216   | 34     |
| sp-mz     | A      | 4      | 4      | 400       | 128    | 128    | 16     |
| sp-mz     | B      | 8      | 8      | 400       | 304    | 208    | 17     |
| sp-mz     | C      | 16     | 16     | 400       | 480    | 320    | 28     |
| sp-mz     | D      | 32     | 32     | 500       | 1632   | 1216   | 34     |

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
        "help": "Nome do benchmark."
      },
      "Classe": {
        "params": ["-c"],
        "type": "string",
        "help": "Nome da classe do benchmark."
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
  - $\color{blue}\text{\textbf{script\\\_template\\\_name}}$: Nome do arquivo com o template do script de execução para a aplicação, com o molde do script a ser submetido.
  - $\color{blue}\text{\textbf{suggestions\\\_map}}$: Objeto que mapeia cada opção do aplicativo do usuário, caso ele sugira as possíveis configurações das quais iremos escolher a melhor configuração. No caso do programa do usuário atual, temos três opções, temos três opções associadas `s características correspodentes da base usada para o treinamento do modelo para uma aplicação. No caso do RAxML, a opção do script do usuário $\color{gray}\text{\textbf{nodes}}$ está associada a característica $\color{gray}\text{\textbf{NNodes}}$, a opção do script do usuário $\color{gray}\text{\textbf{process}}$ está associada a característica $\color{gray}\text{\textbf{Processo p/ no}}$, e a opção $\color{gray}\text{\textbf{threads}}$ está associada a característica $\color{gray}\text{\textbf{Thread p/ proc.}}$.
  - $\color{blue}\text{\textbf{user\\\_options}}$: Objeto definindo os parâmetros da aplicação que o usuário precisará fornecer ao executar o script para otimizar essa aplicação. Tem uma chave para cada parâmetro necessário, que indica textualmente o parâmetro. Cada chave aponta para um objeto com as seguintes informações sonbre o parâmetro descrito pela chava:
  - È composto pelos seguintes campos:
    - $\color{blue}\text{\textbf{params}}$: Lista com sa opções de execução, passadas ao script de otimização, que definem o valor do parâmetro. No caso do RAxML e do parâmetro $\color{gray}\text{\textbf{Bootstrap}}$, as possíveis opções são $\color{gray}\text{\textbf{-N ou -\\\#}}$.
    - $\color{blue}\text{\textbf{type}}$: O tipo do valor passado pela opção (pode ser $\color{gray}\text{\textbf{integer}}$ para valores inteiros, $\color{gray}\text{\textbf{float}}$ para valores de ponto flutuante, ou $\color{gray}\text{\textbf{String}}$ para parâmetros que são strings). No caso do parâmetro $\color{gray}\text{\textbf{Bootstrap}}$, o tipo é $\color{gray}\text{\textbf{integer}}$.
    - $\color{blue}\text{\textbf{help}}$: Mensagem de ajuda que será mostrada no script de otimização quando o usuário desejar saber o que deve ser passado no parâmetro. parâmetro $\color{gray}\text{\textbf{Bootstrap}}$, a mensagem é $\color{gray}\text{\textbf{``Valor do bootstrap.''}}$
  - $\color{blue}\text{\textbf{conversions}}$: define, para cada parâmetro da aplicação passado pelo usuário que é uma das características usadas para treinar os modelos, como o parâmetro é convertido do valor passado pelo usuário para o valor adequado para a característica usada ao treinar o modelo. As conversões podem ser as seguintes:
    - $\color{blue}\text{\textbf{c: [``copy'', p]}}$: O valor da característica $\color{blue}\text{\textbf{c}}$ é uma cópia do parâmetro $\color{blue}\text{\textbf{p}}$ passado pelo usuário. No exemplo do RAxML, a característica $\color{gray}\text{\textbf{Bootstrap}}$ é igual ao valor do parâmetro $\color{gray}\text{\textbf{Bootstrap}}$.
    - $\color{blue}\text{\textbf{c: [``filsesize'', p]}}$. O valor da característica $\color{blue}\text{\textbf{c}}$ é o tamanho do arquivo cujo caminho foi definido pelo parâmetro $\color{blue}\text{\textbf{p}}$ passado pelo usuário. No exemplo do RAxML, a característica $\color{gray}\text{\textbf{Tamanho}}$ é igual ao tamanho do arquivo dado pelo caminho do parâmetro $\color{gray}\text{\textbf{Arquivo}}$.
    - $\color{blue}\text{\textbf{c: [``map'', arquivo\\\_mapeamento, p}}_1$,$\color{blue}\text{\textbf{p}}_2, \ldots, \text{\textbf{p}}_n\text{\textbf{]}}$: O valor da característica $\color{blue}\text{\textbf{c}}$ é dado pelo valor da característica na tabela do arquivo $\color{blue}\text{\textbf{arquivo\\\_mapeamento}}$, definido pelos valores dos parâmetros $\color{blue}\text{\textbf{p}}_1,\text{\textbf{p}}_2, \ldots, \text{\textbf{p}}_n$. O arquivo de mapeamento é uma tabela no formato CVS, com uma coluna para a característica $\color{blue}\text{\textbf{c}}$ e uma coluna para cada propriedade em $\color{blue}\text{\textbf{p}}_i, 1\leq i\leq n$. No arquivo de configuração do ntb (benckmarks do NAS), tem seis mapeamentos, usando a tabela dada em $\color{gray}\text{\textbf{ntb\\\_map\\\_table.cvs}}$ e os parâmetros $\color{gray}\text{\textbf{Benchmark}}$ e $\color{gray}\text{\textbf{Classe}}$ passados pelo usuário. Por exemplo, no arquivo de configuração do NTB, a configuração $\color{gray}\text{\textbf{``Zone X'': [``map'', ``ntb\\\_map\\\_table.cvs'',}}$ $\color{gray}\text{\textbf{``Benchmark'', ``Classe'']}}$, vai escolher o valor para a característica $\color{gray}\text{\textbf{``Zone X''}}$ usando o valor definido pelos parâmetros $\color{gray}\text{\textbf{Benchmark}}$ e $\color{gray}\text{\textbf{Classe}}$ da tabela (a tabela é dada antes do arquivo de configuração do NTB).
  - $\color{blue}\text{\textbf{slurm}}$: Lista com os objetos que definem as possíveis esquemas que podem ser uadas ao submeter a aplicação, sendo cada esquema composto por una partição e as suas configurações. A partição escolhida pel script executado pelo usuário será aquela tempo máximo da partição mais próximo do que o tempo predito para a configuração escolhida pelo script, sendo que somente serão consideradas as partição que possam executar essa sugestão feita. Para cada partição, o objeto tem os seguintes campos:
    - $\color{blue}\text{\textbf{partition}}$: Nome da partição, que é o mesmo usado ao submeter os trabalhos. No arquivo de configuração do RAxMl e do NTB, o nome da primeira partição da lista é $\color{gray}\text{\textbf{sequana\\\_cpu\\\_dev}}$, a partição de mesmo nome do Santos Dumont.
    - $\color{blue}\text{\textbf{max\\\_time}}$: Tempo máximo de execução da partição, em segundos. Para a partição $\color{gray}\text{\textbf{sequana\\\_cpu\\\_dev}}$, o tempo máximo é de $\color{gray}\text{\textbf{1200}}$ (20 minutos).
    - $\color{blue}\text{\textbf{max\\\_memory}}$: Tamanho máximo da memória em KB. Aqui, foi usado o tamanho máximo que usamos nos testes do NAS (NTB), $\color{gray}\text{\textbf{367001600}}$ (350GB).
    - $\color{blue}\text{\textbf{exclusive}}$: Se a execução do aplicativo na particão será de uso exclusivo ($\color{gray}\text{\textbf{true}}$) ou não ($\color{gray}\text{\textbf{false}}$). Nos arquivos de configuração do RAxML e do NAS, a partição será de uso exclusivo ($\color{gray}\text{\textbf{true}}$) porque executamos os testes com usu exclusivo.
    - $\color{blue}\text{\textbf{default}}$: Se a partição é a default ($\color{gray}\text{\textbf{true}}$) ou não ($\color{gray}\text{\textbf{false}}$), usada quando não for possível escolher uma partição que permita executar a aplciação no tempo predito para a sugestão. Neste caso, um aviso será mostrado ao usuário, indicando que não foi possível para executar a aplicação dentro do tempo predito. Somente uma partição da lista deve ser a default e deve sempre existir uma partição default. Nos arquivos de configuração do RAxML e do NAS, a partição default é a $\color{gray}\text{\textbf{sequana\\\_cpu}}$.
    - $\color{blue}\text{\textbf{nodes}}$: Número máximo de nós para o esquema. Não é necessáriamente igual ao número de nós máximo da partição, mas não pode ser maior do que o máximo de nós da partição: No esquema que usa a partição $\color{gray}\text{\textbf{sequana\\\_cpu\\\_dev}}$, o número máximo é $\color{gray}\text{\textbf{4}}$ (igual ao máximo da partição).
    - $\color{blue}\text{\textbf{process}}$: Número máximo de processos que podem ser executados em cada nó. Para os dois esquemas da lista, que usam, respectivamente, as partições $\color{gray}\text{\textbf{sequana\\\_cpu\\\_dev}}$ e $\color{gray}\text{\textbf{sequana\\\_cpu}}$, o valor é é $\color{gray}\text{\textbf{2}}$, porque cada nó, em ambas as partições, tem dois processadores físicos, implicando que podemos no máximo ter dois processos, cada um ligado a um processador diferente, caso tenham sido pedidos dois processos por nó.
    - $\color{blue}\text{\textbf{threads}}$: Número máximo de threads que podem ser executados em cada processo. Para os dois esquemas da lista, que usam, respectivamente, as partições $\color{gray}\text{\textbf{sequana\\\_cpu\\\_dev}}$ e $\color{gray}\text{\textbf{sequana\\\_cpu}}$, o valor é $\color{gray}\text{\textbf{48}}$, pois cada processador tem $\color{gray}\text{\textbf{24}}$ núcleos de processamento, e quando somente temos um processo por nó, caso em que não é necessário ligar um processo a um processador, podemos ter 48 threads no máximo, com cada processador executando 24 threads (1 thread para cada núcleo).

## Exemplo do arquivo de configuração $\color{red}\text{\textbf{user\\\_config.json}}$

```json
{
  "collect_consumed_energy": false,
  "default_script_name": "script.sh",
  "users_activity": {
    "enable": true,
    "data_file_prefix": "executed_jobs_data",
    "data_file_dir": "logs",
    "data_file_type": "csv"
  },
  "slurm": {
    "submission_program": "sbatch",
    "submission_message": ".*Submitted batch job (\\d+).*"
  }
}
```

O arquivo de configuração do usuário é composto pelos segintes campos descritos a seguir (precisamos ver se todos os campos serão mantidos):

- $\color{blue}\text{\textbf{collect\\\_consumed\\\_energy}}$: Este é um campo que não sei se é necessário. Seria no caso de não ser possível configurar p Sentos Dumont para coletar energia e introduziríamos isso de algum modo no script submetido para o usuário. O valor atual é $\color{gray}\text{\textbf{false}}$, mas o script do usuário está no momento ignorando o campo.
- $\color{blue}\text{\textbf{default\\\_script\\\_name}}$: Nome default para o script de submissão se o usuário não fornecer um nome pela opção -s do script de otimização usado pelo usário e se nçao pedir para submeter o script (opção -r). No caso de pedir para submeter, se a opção -s não for passada, o script será criado em um arquivo temporário que será removido após a submissão. O nome default é $\color{gray}\text{\textbf{script.sh}}$
- $\color{blue}\text{\textbf{users\\\_activity}}$: Objeto que define como será armazenado os dados das aplicações otimizadas pelo usuário. Para cada aplicação, o arquivo que armazenará os dados será dado pelo nome no campo $\color{blue}\text{\textbf{data\\\_file\\\_prefix}}$, sendo salvo no diretório definido pelo campo $\color{blue}\text{\textbf{data\\\_file\\\_dir}}$. Para cada aplicação, será armazenado o ID do Job, o nome do job dado pelo usuário, as sugestões (no momento, número de nós, processos por nó e threads por processo), os parâmetros da aplicação passados pelo usuário e usado no treinamento dos modelos, os parâmetros convertidos obtidos dos parâmetros passados pelo usuados e usados no treinamento dos modelos, o tempo predito para a sugestão, e a pontuação da sugestão (no momento é a variável predita definida pelo campo $\color{blue}\text{\textbf{suggestion}}$ do objeto definido pelo campo $\color{blue}\text{\textbf{estimated\\\_parameters}}$ definido no arquivo de configuração de cada aplicação, para o RAxML e o NTB, é o valor predito para o EDP). Por enquanto, o script ignora este objeto, mas vou em breve implementar o salvamento dos logs e usar o objeto. Os campos são os seguintes:
  - $\color{blue}\text{\textbf{users\\\_activity}}$: Define se deve ser habilidada ($\color{gray}\text{\textbf{true}}$) ou não ($\color{gray}\text{\textbf{false}}$) o log da execução de um aplicação por um usuário.
  - $\color{blue}\text{\textbf{data\\\_file\\\_prefix}}$: Prefixo do nome do arquivo do log. O nome será este prefixo mais um sublinhado ($``\\\_''$), mais o nome do aplicativo, mas a extensão ($``.csv''$). No caso da configuração de exemplo, o prefixo é $\color{gray}\text{\textbf{executed\\\_jobs\\\_data}}$.
  - $\color{blue}\text{\textbf{data\\\_file\\\_path}}$: Diretório em que serão armazenados os logs, como os outros diretórios, é relativo ao diretório principal em que todos os arquivos foram armazenados. No caso da configuração de exemplo, o diretório é $\color{gray}\text{\textbf{logs}}$.
- $\color{blue}\text{\textbf{slurm}}$: Objeto com as informações sobre como o script deve ser submetido. Tem os seguintes campos:
  - $\color{blue}\text{\textbf{submission\\\_program}}$: Nome do aplicatioo usdo para submeter os trabalhos no supercomputador. No exemplo do arquivo de configuração, o aplicativo é o $\color{gray}\text{\textbf{sbatch}}$.

[1]: https://doi.org/10.1007/s10994-006-6226-1
[2]: https://doi.org/10.5753/sscad.2025.16760
