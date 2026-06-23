import pandas as pd
import numpy as np
from Utils.Suggestions import FilterOutliers, BestHiperparams, SuggestionsPredictor, DiscoverBestModel
from pprint import pprint
import sklearn.ensemble as sken
import sklearn.tree as sktree
#from Utils.TestPredictors import BaselineByResource, BaselineMostCommonConfig

# Le os dados
variaveis_de_entrada = ['NNodes', 'Processo p/ no', 'Thread p/ proc.', 'Bootstrap', 'Tamanho']
variaveis_de_saida = ['ElapsedRaw', 'Consumo de Energia Total (J)', 'EDP']

dados = pd.DataFrame()

lista_arquivos = ['../data/raxml_processos.csv']

for arquivo in lista_arquivos:
	dados_arquivo = pd.read_csv(arquivo, usecols=variaveis_de_entrada + variaveis_de_saida)
	dados = pd.concat([dados, dados_arquivo])

dados = dados.reset_index(drop=True)

print("Dados de entrada.\n\n")
print(dados.to_markdown(tablefmt="grid"))

# Faz o filtro dos dados
limite_outliers = 1e2

data_filter = FilterOutliers()

dados_limpos = data_filter.Filter(dados, variaveis_de_entrada, variaveis_de_saida, limite_outliers)

print("Dados de entrada filtrados.\n\n")
print(dados_limpos.to_markdown(tablefmt="grid"))

variaveis_sugestao = ['NNodes', 'Processo p/ no', 'Thread p/ proc.']
variaveis_aplicativo = ['Bootstrap', 'Tamanho']
variaveis_usuario = variaveis_aplicativo
variaveis_preditas = ['EDP','ElapsedRaw']

# Obtem os melhores hiperparametros

# Descobre os melhores hiperparametros

param_grid_ensemble = {
	"max_depth": [5, 10, 15, None],
	'n_estimators': [10, 20, 50, 100, 120, 150],
}

param_grid_tree = {
	"max_depth": [5, 10, 15, None],
}

for variavel_predita in variaveis_preditas:
	print(f"Fazendo os treinamentos considerando a variável {variavel_predita}")
	modelos_arvore = {"ExtraTreesRegressor": sken.ExtraTreesRegressor(random_state=42),
										"GradientBoostingRegressor": sken.GradientBoostingRegressor(random_state=42),
										"RandomForestRegressor": sken.RandomForestRegressor(random_state=42),
										"DecisionTreeRegressor": sktree.DecisionTreeRegressor(random_state=42)}

	param_modelos = {}

	for modelo in modelos_arvore.keys():
		if modelo == "DecisionTreeRegressor":
			param_grid = param_grid_tree
		else:
			param_grid = param_grid_ensemble
		print(f"Otimizando os hiperparâmetros {list(param_grid.keys())} do modelo {modelo}")

		grid_search_modelo = BestHiperparams()   
		
		best_params, best_score = grid_search_modelo.optimize(dados_limpos, 
																variaveis_sugestao, 
																variaveis_aplicativo, 
																variaveis_usuario, 
																variavel_predita, 
																modelos_arvore[modelo], 
																param_grid)
		
		param_modelos[modelo] = {'BestHiperparamsObject': grid_search_modelo,
														'BestHiperparams': best_params,
														'BestScore': best_score}
		print(f"Melores hiperparâmetros para o modelo {modelo}: {best_params}")
		print(f"Melhor score para o modelo {modelo}: {best_score}")

	# Descobre o melhor modelo.
	final_models = {"ExtraTreesRegressor": sken.ExtraTreesRegressor(**param_modelos['ExtraTreesRegressor']['BestHiperparams'], 
																																	random_state=42),
									"GradientBoostingRegressor": sken.GradientBoostingRegressor(**param_modelos['GradientBoostingRegressor']['BestHiperparams'], 
																																	random_state=42),
									"RandomForestRegressor": sken.RandomForestRegressor(**param_modelos['RandomForestRegressor']['BestHiperparams'], 
																																	random_state=42),
									"DecisionTreeRegressor": sktree.DecisionTreeRegressor(**param_modelos['DecisionTreeRegressor']['BestHiperparams'], 
																																	random_state=42),
#									"MáximosRecursos": BaselineByResource(variaveis_sugestao, mode="max"),
#									"MínimosRecursos": BaselineByResource(variaveis_sugestao, mode="min"),
#									"BaselineReal": BaselineMostCommonConfig(variaveis_sugestao, variaveis_usuario, dados_limpos)
														}

	evaluate_models = DiscoverBestModel()
	best_model_name, best_model_scores, results_df, scores_df = evaluate_models.best_model(dados_limpos,
																																												variaveis_sugestao, 
																																												variaveis_aplicativo, 
																																												variaveis_usuario, 
																																												variavel_predita, 
																																												final_models)

	print(f"O melhor modelo é o {best_model_name}, com as pontuações, avaliadas da esquerda para a direita, {', '.join(best_model_scores)}")
	print("Tabela com as acurácias de todos os modelos")
	print(results_df.to_markdown(tablefmt="grid"))
	print("Tabela com as médias das acurácias de todos os modelos")
	print(scores_df.to_markdown(tablefmt="grid"))

	predictor = SuggestionsPredictor()
	predictor.fit(dados_limpos, 
								variaveis_sugestao, 
								variaveis_aplicativo, 
								variaveis_usuario, 
								variavel_predita, 
								final_models[best_model_name])
	
	print("Dataframe do oráculo")
	oracle = predictor.get_oracle()
	print(oracle.to_markdown(tablefmt="grid"))

	nome_arquivo_preditor = f'../predictors/{variavel_predita}_{best_model_name}_raxml.pickle'
	print(f"Salvando o preditor no caminho {nome_arquivo_preditor}")
	predictor.save_predictor(nome_arquivo_preditor)


