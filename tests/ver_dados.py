import pandas as pd
import json
import sys

if (len(sys.argv) < 2):
  print(f"Sintaxe: {sys.argv[0]} <lista_arquivos_json>")
  exit(-1)

# Lê o json com o nome dos arquivos, dado em ver_dados.json
nome_arquivo = sys.argv[1]

with open(nome_arquivo, 'r') as file:
  lista_arquivos = json.load(file)

dados = pd.DataFrame()

for nome_arquivo in lista_arquivos["dataset_files"]:
  dados_arquivo = pd.read_csv(nome_arquivo)
  dados = pd.concat([dados, dados_arquivo])

dados = dados.reset_index(drop=True)		

print("--> Conjunto de dados original da aplicação:")
print("\n", dados.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
print("--> Dados estatísticos referentes ao conjunto de dados original:")
print("\n", dados.describe().to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
print("--> Valores únicos:")
print("\n", dados.nunique().to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

