#!/bin/bash

# Inicialização das variáveis
benchmark=""
classe=""

ajuda()
{
  echo "Uso: $0 [-h] [-b] benchmark (bt-mz, lu-mz, sp-mz) [-c] classe (A, B, C ou D)]"
}

while getopts "hb:c:" opt; do
  case ${opt} in
    h)
      ajuda
      exit 0
      ;;
    b)
      benchmark="$OPTARG"
      ;;
    c)
      classe="$OPTARG"
      ;;
    \?)
      echo "Opção inválida." >&2
      exit 1
      ;;
  esac
done

# Validação: Verifica se foi passado um benchmark e se o nome é válido.
if [[ -z "$benchmark" ]]; then
  echo "Erro: A opção -b é obrigatória." >&2
  ajuda
  exit 1
else
  case "${benchmark,,}" in
    bt-mz|sp-mz|lu-mz)
    ;;
  *)
    echo "Erro: A opção -b precisa ser bt-mz, lu-mz ou sp-mz." >&2
    exit 1
  esac
fi

# Validação: Verifica se foi passado uma classe e se a classe é válida
if [[ -z "$classe" ]]; then
  echo "Erro: A opção -c é obrigatória." >&2
  ajuda
  exit 1
else
  case "${classe^^}" in
    A|B|C|D)
    ;;
  *)
    echo "Erro: A opção -c precisa ser A, B, C ou D!" >&2
    exit 1
  esac
fi

# Nome do aplicativo
aplicativo="$benchmark.$classe.x"

# Diretório do executável.
bindir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Exibe o resultado do processamento
echo "Benchmark: $benchmark"
echo "Classe: $classe"
echo "Executável: $aplicativo"

# Executando o aplicativo.
$bindir/$aplicativo
