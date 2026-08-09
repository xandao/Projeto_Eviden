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

Depois, basta usar a aplicação *trainer*, que é um _wrapper_ para executar o script Trainer.py no diretório src. A ajuda de uso do script é a seguinte:

```bash
uso: trainer [opções] Command [Parâmetros]

Script para treinar os modelos para todos os aplicativos que vamos otimizar o uso.

Opções principais:
  command        Comando a ser executado. Pode ser um dos seguintes comandos:

                 applications: lista todas as aplicações que podemos treinar os modelos.

                 models: lista os modelos que são avaliados quando os preditores forem gerados.

                 train app1, app2, ..., appn -> Faz todo o processo de treinamento, da filtragem dos dados, otimização dos
                 hiperparâmetros dos modelos, escolha do melhor modelo e treinamento deste melhor modelo com todos os
                 dados, sendo gerado um modelo para auxiliar a geração das sugestões e outro para predizer o tempo.

                 Cada aplicação da lista é considerada na ordem dada e os treinamentos são independentes, ou seja,
                 passar a lista é equivalente a executar o script com o comando para cada aplicação isoladamente.

  -v, --verbose  Habilita a verbosidade do script.

Ajuda:
  -h, --help     Mostra esta mensagem de ajuda e sai.
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

- $\color{blue}\text{\textbf{train lista\\\_nomes\\\_modelos}}$: para fazer os treinamentos que irão gerar o preditor para cadaa aplicação identificada por um nome em $\color{blue}\text{\textbf{lista\\\_nomes\\\_modelos}}$. Cada nome da lista precisa ser um dos nomes dados nos campos $\color{blue}\text{\textbf{name}}$ dos arquivos de configurações das aplicações. A saída gerada pelos treinamentos feitos ao gerar o preditor para o modelo, dependerá da verbosidade estar ou não ativada. Se a verbosidade estiver habilitada, recomento fazer o treinamento de cada modelo com execuççoes separadas do trainer, porque todos os logs de todos os trainemtnos de todas as aplicações seriam gerados em sequência Exemplo, em que supomos que a clonagem do repositório foi feita no diretório raiz do usuário $\color{gray}\text{\textbf{user}}$:

```bash
user$ trainer train raxml
--> Salvando o preditor treinado como o modelo ExtraTreesRegressor (nome ETR) no arquivo /home/user/Projeto_Eviden/predictors/raxml_ETR_EDP.pickle.

```

Opções do script:

- $\color{blue}\text{\textbf{-v ou --verbose}}$: Habilita a verbosidade.
- $\color{blue}\text{\textbf{-h ou --help}}$: Mostra a ajuda mostrada anteriormente.

## Script de otimização

O script de otimização será usado pelo usuário para obter as melhores configurações (no momento, número de nós, de processos por nó e de therads por processo) para gerar o script de submissão que pode, se o usuário desejar, ser automaticamente submetido. Para usar o script de treinamento, primeiramente é necessário carregar o módulo do script, com o seguinte comando:

```bash
module load optimizer
```

Depois, basta usar a aplicação *otimizer*, que é um _wrapper_ para executar o script Optimizer.py no diretório src. A ajuda de uso do script é a seguinte:

````bash
uso: optimizer [opções] -- [executável da aplicação] [-h] [opções obrigatórias da aplicação] [outras opções da aplicação]

Script para escolher a melhor configuração para aplicações selecionadas.

Opções principais:
  -r, --run             Submete o script com a melhor configuração de execução.
  -j JOBNAME, --jobname JOBNAME
                        Nome do trabalho registrado no sistema de submissão.
  -s SCRIPT, --script SCRIPT
                        Salva o script gerado em um arquivo.
  -S, --suggestion      Somente mostra a sugestão para os parâmetros do script.
  -n NODES [NODES ...], --nodes NODES [NODES ...]
                        Lista com os possíveis números de nós, se a aplicação usa múltiplos nós.
                        Usada conjuntamente com as opções -p e -t que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -n 1 2:10:2 -> Nós: 1, 2, 4, 6, 8, 10.
                                  -n :10:2    -> Nós: 1, 3, 5, 7, 9.
                                  -n 1:5      -> Nós: 1, 2, 3, 4, 5.
  -p PROCESS [PROCESS ...], --process PROCESS [PROCESS ...]
                        Lista com os possíveis números de processos, se a aplicação usa múltiplos processos por nó.
                        Usada conjuntamente com as opções -n e -t que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -p 1 2:      -> Processos: 1, 2.
                                  -p 1 :3:1    -> Processos: 1, 2, 3.
                                  -p :3 6:12:3 -> Processos: 1, 2, 3, 6, 9, 12
  -t THREADS [THREADS ...], --threads THREADS [THREADS ...]
                        Lista com os possíveis números de threads, se a aplicação usa múltiplas threads por processo.
                        Usada conjuntamente com as opções -n e -p que terão os valores default se não usadas.
                        Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.
                        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
                        Default 1:1.
                        Exemplos: -t 1 2:24:2 -> Threads: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
                                  -t 2 :24:8  -> Threads: 2, 24, 32, 40, 48.
                                  -t 2 24 48  -> Threads: 2, 24, 48.
  -v, --verbose         Habilita a verbosidade do script.
  -l, --list            Lista as aplicações cujas execuções podem ser otimizadas pelo script.

Ajuda:
  -h, --help            Mostra esta mensagem de ajuda e sai.```

O formato do comando é:

```bash
optimizer <opções_do_otimizador> -- <executável_aplicação> <parâmetros_da_aplicação>
````

As opções do otimziador são as seguintes:

- $\color{blue}\text{\textbf{-l ou --list}}$: Lista as aplicações disponíveis para serem otimizadas. Um exemplo de saída é o seguinte, onde é mostrado, para cada aplicação que pode ser otimizada, o identificador da aplicação e os possíveis executáveis que podem ser usados no otimizador após o separador $\text{``--``}$ que separa os parâmetros do otimizador dos parâmetros da aplicação:

```bash
user$ optimizer --list
➡️  Aplicação ntb, possíveis nomes para os executáveis: ntb.sh
➡️  Aplicação raxml, possíveis nomes para os executáveis: raxmlHPC-PTHREADS-AVX-omp, raxml
➡️  Aplicação raxml-sscad, possíveis nomes para os executáveis: raxml-sscad
➡️  Aplicação raxml-sscad-time, possíveis nomes para os executáveis: raxml-sscad-time
```

- $\color{blue}\text{\textbf{-v ou --verbose}}$: Habilita a verbosidade.
- $\color{blue}\text{\textbf{-h ou --help}}$: Mostra a ajuda mostrada anteriormente.
- Os parâmetros $\color{blue}\text{\textbf{-n ou --nodes}}$, $\color{blue}\text{\textbf{-p ou --process}}$ e $\color{blue}\text{\textbf{-t ou --threads}}$ são usados conjuntamente para definir uam faixa customizada de números de nós, números de processos por nó, e números de threads a serem usadas ao avaliar a melhor sugestão para executar a aplicação que o usuário deseja otimizar. Como observado na ajuda, cada opção efetivamente define uma lista de valores. Se uma opção deixar de ser fornecida, será considerado o valor default 1. Se nenhuma opção for definida, as sugestões serão baseadas em todas as combinações feitas quando os dados de treinamento, usados para treinar os modelos da aplicação. Valores inadequados para o número de nós, processos e threads serão desconsiderados, com um aviso sendo emitido na tela. Exemplo de uso das opções:

```bash
user$ python Optimizer.py -r -s script.sh -j teste-customizado -n 1:10 -p 1 2 4 -t 2 4 8 16 3
2 64 -- raxml -N 100 -s ~xandao/Downloads/DENV_3-colombia-BVBRC_genome_sequence.mafft
⚠️  Descartando todos os valores para a opção process maiores do que 2 permitidos pelas possíveis partições sequana_cpu_dev, sequana_cpu da aplicação raxml!
⚠️  Descartando todos os valores para a opção threads maiores do que 48 permitidos pelas possíveis partições sequana_cpu_dev, sequana_cpu da aplicação raxml!
➡️  Script de submissão script.sh criado com sucesso!
➡️  Script de submissão submetido com sucesso!
➡️  O trabalho foi submetido com o identificador 386804.
```

- $\color{blue}\text{\textbf{-j ou --jobname}}$: Nome do trabalho associado à aplicação quando ela for executada.
- $\color{blue}\text{\textbf{-r ou --run}}$: Submete para execução o script gerado com a sugestão otimizada. A seguir está um exemplo diferente dado na descrição das opções anteriores, quando não são defindas opções customizadas para as sugestões a serem avaliadas:

```bash
user$ python Optimizer.py -r -s script.sh -j teste -- raxml -N 100 -s ~xandao/Downloads/DENV_3-colombia-BVBRC_genome_sequence.mafft
➡️  Script de submissão script.sh criado com sucesso!
➡️  Script de submissão submetido com sucesso!
➡️  O trabalho foi submetido com o identificador 493412.
```

- $\color{blue}\text{\textbf{-s <arquivo> ou }}$$\color{blue}\text{\textbf{--script <arquivo>}}$: Salva o script de submissão que deve ser usado para submeter a aplicação com as sugestões otimizadas no arquivo dado pelo caminho $\color{blue}\text{\textbf{<arquivo>}}$. Se a opção $\color{blue}\text{\textbf{-r ou --run}}$ for usada e essa opção não for passada, o script será gerado em um arquivo temporário. Em caso contrário, o script será salvo no arquivo cujo nome é definido na opção $\color{blue}\text{\textbf{default\\\_script\\\_name}}$ do arquivo de configuração do script de otimização $\color{red}\text{\textbf{user\\\_config.json}}$ (para mais detalhes, ver o arquivo de ajuda do diretório [configs](configs/README.md)), que por default é $\color{blue}\text{\textbf{script.sh}}$.
- $\color{blue}\text{\textbf{-S ou --suggestion}}$: Somente mostra a sugestão na tela do usuário, como mostrado no exemplo a seguir:

```bash
user$ otimizer -S -j teste -- raxml -N 100 -s ~xandao/Downloads/DENV_3-colombia-BVBRC_genome_sequence.mafft
➡️  Sugestão: nodes=4, process=2, threads=8
```

Quando o otimizador for usado com uma aplicação, existem opções desta aplicação que devem ser obrigatoriamente definidas. Para saber as opções, use o comando -h para a aplicação como mostrado a seguir:

```bash
user$ python Optimizer.py -- raxml -h
uso: raxml [-h] -N BOOTSTRAP -s ARQUIVO

Parser responsável pelos parâmetros da aplicação.

Opções principais:
  -N BOOTSTRAP, -# BOOTSTRAP
                        Valor do bootstrap.
  -s ARQUIVO            Arquivo de entrada com as sequências.

Ajuda:
  -h, --help            Mostra esta mensagem de ajuda e sai.```
