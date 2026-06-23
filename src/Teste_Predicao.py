import pandas as pd
import numpy as np
from Utils.Suggestions import SuggestionsPredictor
from pprint import pprint
import sys
import argparse
from pathlib import Path

# Le os dados
variaveis_de_entrada = ['NNodes', 'Processo p/ no', 'Thread p/ proc.', 'Bootstrap', 'Tamanho']
variaveis_de_saida = ['ElapsedRaw', 'Consumo de Energia Total (J)', 'EDP']

parser = argparse.ArgumentParser(description="Test script to predict suggestion variables.")

parser.add_argument("-N", type=int, required=True, default=None, help="Bootstrap value")

parser.add_argument("-s", type=str, required=True, help="Input file path")

parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable output verbosity")

# Parse the parameters
args = parser.parse_args()

bootstrap = args.N
caminho_arquivo = Path(args.s)
tamanho = caminho_arquivo.stat().st_size
verbose = args.verbose

user_app_teste = {'Bootstrap': bootstrap, 'Tamanho': tamanho}

print("Lendo o preditor")

nome_arquivo_preditor = '../predictors/EDP_ExtraTreesRegressor_raxml.pickle'

predictor = SuggestionsPredictor.load_predictor(nome_arquivo_preditor)

print("Teste 1: Teste usando as configurações padrões (as mesmas do treinamento)")

suggestion = predictor.get_suggestion(user_app_teste, verbose=verbose)

print(f"Sugestão para o bootstap {bootstrap} e o tamanho do arquivo {tamanho} (arquivo {args.s})")
SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)

custom_configuratios = {
	'NNodes' : range(1, 11), 
	'Processo p/ no': [1, 2, 4], 
	'Thread p/ proc.': [2, 4, 8, 16, 32, 64],
}
custom_suggestion = predictor.get_suggestion(user_app_teste, custom_configuratios, verbose=verbose)

print(f"Teste 2: Teste usando as configurações passadas pelo usuário:")
pprint(custom_configuratios)
print(f"Sugestão para o bootstap {bootstrap} e o tamanho do arquivo {tamanho} (arquivo {args.s})")
SuggestionsPredictor.print_suggestion(custom_suggestion, show_score=True, show_X=True, show_y_pred=True)

user_app_teste_list = pd.DataFrame({'Bootstrap': [50, 500, 100, 1000], 'Tamanho': [123456, 654321, 231143, 198574]})

print("Testes 3 e 4, iguais ao 1 e 2, mas usando os parâmetros do usuário dados no seguinte dataframe:")
print(user_app_teste_list.to_markdown(tablefmt="grid"))
print("Teste 3: Teste usando as configurações padrões (as mesmas do treinamento):")

suggestions = predictor.get_suggestions(user_app_teste_list, verbose=verbose)

for pos, suggestion in enumerate(suggestions):
	print(f"Sugestão para o bootstap {user_app_teste_list.loc[pos, 'Bootstrap']} e o tamanho do arquivo {user_app_teste_list.loc[pos, 'Tamanho']})")
	SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)
  
custom_suggestions = predictor.get_suggestions(user_app_teste_list, custom_configuratios, verbose=verbose)

print(f"Teste 4: Teste usando as configurações passadas pelo usuário:")
for pos, suggestion in enumerate(custom_suggestions):
	print(f"Sugestão para o bootstap {user_app_teste_list.loc[pos, 'Bootstrap']} e o tamanho do arquivo {user_app_teste_list.loc[pos, 'Tamanho']})")
	SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)