# Projeto Escalonador

Neste repositório, estão os todos os arquivos de código e arquivos auxiliares do projeto que tem dois principais componentes, o script que faz os treinamentos para uma dada aplicação e o script usado para fazer as sugestões.

Os scripts desenvolvidos neste repositório são usados após o modulo correspondente do SLURM do script for carregado. Se o módulo não foi instalado no sistema, ainda será possível usar os módulos de modo local, usando o comando:

```bash
module use <diretório_do_git_clone>/modules
```

onde <diretório_do_git_clone> é o diretório em que o repositório foi
passado, ao usar o comando:

```bash
git clone https://github.com/xandao/Projeto_Eviden
```

## Script de treinamento

O script de treinamento será usado pelos administradores do sistema, quando desejar treinar um modelo para novas aplicações, ou atualizar o modelo de uma aplicação após novos dados serem obtidos e adicionados aos dados já existentes do treinamento anterior do modelo. Para usar o script de treinamento, primeiramente é necessário carregar o módulo do script, com o seguinte comando:

```bash
module load trainer
```

Depois, basta usar a aplicação trainer, que é um _wrapper_ para executar o script Trainer.py no diretório src. A ajuda de uso do script é a seguinte:

```bash
usage: Trainer.py [opções] Command [Parâmetros]

Script para teinar os modelos para todos os aplicativos que vamos otimizar o uso.

Opções principais:
  command        Comando a ser executado. Pode ser um dos seguintes comandos:

                 applications: lista todas as aplicações que podemos treinar os modelos.

                 models: lista os modelos que são avaliados quando os preditores forem gerados.

                 train app1, app2, ..., appn -> Faz todo o processo de treinamento, da filtragem dos dados, otimização dos
                 hiperparâmetros dos modelos, escolja do melhor modelo e treinamento deste melhor modelo com todos os
                 dados, sendo gerado um modelo para auxiliar a geração das sugestões e outro para predizer o tempo.

                 Cada aplicação da lista é considerada na ordem dada e os treinamentos são idependentes, ou seja,
                 passar a lista é equivalente a executar o script com o comando para cada aplicação isoladamemte.

  -v, --verbose  Habilita a verbosidade do script.

Ajuda:
  -h, --help     Mostra esta mensagem de ajuda e sai
```

Os possíveis comandos são:

- $\color{blue}\text{applications}$: Mostra uma lista com todas as aplicações, como identificadas pelos seus nomes definidos nos arquivos de configuração das aplicações. Exemplo:

```bash
user$ trainer applications
ntb
raxml
raxml-sscad
raxml-sscad-time
```

- $\color{blue}\text{models}$: Mostra todos os modelos que são considerados ao fazer os treinamentos, definidos no arquivo de configuração $\color{red}\text{\textbf{training\\\_config.json}}, sendo o nome de cada modelo a sua respectiva chave neste arquivo. Exemplo:

```bash
user$ trainer models
ExtraTreesRegressor
GradientBoostingRegressor
RandomForestRegressor
DecisionTreeRegressor
```

- $\color{blue}\text{\textbf{train lista\\\_nomes\\\_modelos}}$: para fazer os treinamentos que irão gerar o preditor para a aplicação identificada pelo nome $\color{blue}\text{\textbf{lista\\\_nomes\\\_modelos}}$. Cada nome da lista precisa ser um dos nomes dados nos campos $\color{blue}\text{\textbf{name}}$ dos arquivos de configurações das aplicações. A saída gerada pelos treinamentos feitos ao gerar o preditor para o modelo, dependerá da verbosidade estar ou não ativada. Se a verbosidade estiver habilitada, recomento fazer o treinamento de cada modelo com execuççoes separadas do trainer, porque todos os logs de todos os trainemtnos de todas as aplicações seriam gerados em sequência Exemplo, em que supomos que a clonagem do repositório foi feita no diretório raiz do usuário $\color{gray}\text{\textbf{user}}$:

```bash
user$ trainer train raxml
--> Salvando o preditor treinado como o modelo ExtraTreesRegressor (nome ETR) no arquivo /home/user/Projero_Evidem/predictors/raxml_ETR_EDP.pickle.

```

Opções do script:

- $\color{blue}\text{\textbf{-v ou --verbose}}$: Habilita a verbosidade.
- $\color{blue}\text{\textbf{-h ou --help}}$: Mostra a ajuda mostrada anteriormente.

## Script de otimização

O script de otimização será usado pelo usuário para obter as melhores configurações (no momento, número de nós, de processos por nó e de therads por processo) para obter um script de submissão que pode, se o usuário desejar, ser automaticamente submetido. ara usar o script de treinamento, primeiramente é necessário carregar o módulo do script, com o seguinte comando:

```bash
module load optimizer
```

Depois, basta usar a aplicação otimizer, que é um _wrapper_ para executar o script Optimizer.py no diretório src. A ajuda de uso do script é a seguinte:

```bash
usage: Optimizer.py [opções] -- [executável da aplicação] [-h] [opções obrigatórias da aplicação] [outras opções da aplicação]

Script para escolher a melhor configuração para aplicações selecionadas.

Opções principais:
  -r, --run             Submete o script com a melhor configuração de execução.
  -j JOBNAME, --jobname JOBNAME
                        Nome do trabalho registrado no sistema de submissão.
  -s SCRIPT, --script SCRIPT
                        Salva o script gerado em um arquivo.
  -S, --suggestion      Somente mostra a sugestão para os parâmetros do script.
  -n NODES [NODES ...], --nodes NODES [NODES ...]
                        Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
                        Usada conjuntamente com as opções -p e -t que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -n 1 2:10:2 -> Nós: 1, 2, 4, 6, 8, 10.
                                  -n :10:2    -> Nós: 1, 3, 5, 7, 9.
                                  -n 1:5      -> Nós: 1, 2, 3, 4, 5.
  -p PROCESS [PROCESS ...], --process PROCESS [PROCESS ...]
                        Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
                        Usada conjuntamente com as opções -n e -t que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -p 1 2:      -> Processos: 1, 2.
                                  -n 1 :3:1    -> Processos: 1, 2, 3.
                                  -n :3 6:12:3 -> Processos: 1, 2, 3, 6, 9, 12
  -t THREADS [THREADS ...], --threads THREADS [THREADS ...]
                        Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
                        Usada conjuntamente com as opções -n e -p que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -n 1 2:24:2 -> Threads: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
                                  -n 2 :24:8  -> Threads: 2, 24, 32, 40, 48.
                                  -n 2 24 48  -> Threads: 2, 24, 48.
  -v, --verbose         Habilita a verbosidade do script.
  -l, --list            Lista as aplicações cujas execuções podem ser otimizadas pelo script.

Ajuda:
  -h, --help            Mostra esta mensagem de ajuda e sai
```
