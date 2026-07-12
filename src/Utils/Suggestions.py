import pandas as pd
import scipy.stats as st
from scipy.stats import gmean
import numpy as np
import numpy.typing as npt
import sklearn.model_selection as skms
import sklearn.ensemble as sken
import sklearn.tree as sktree
import sklearn.preprocessing as skpp
import pickle
import itertools
from sklearn.base import BaseEstimator
from Utils.Common import debug_code

# Item 1: Diferença entre o EDP esperado da configuração sugerida e o EDP esperado da configuração do oráculo dividida pela EDP esperado da configuração do oráculo.
def min_edp_config_diff(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
	"""
	Função para calcular diferença pondenrada entre o valor mínimo em y_true e o valor 
	real associado ao menor valor predito em y_true.

	Parâmetros:

	  y_true : array_like(float)
        Vetor de entreda com os valores reais das medidas.
    y_pred : array_like(float)
        Vetor de entreda com os valores preditos das medidas.
        
  Retorna:

	  float
        A diferença ponderada entre o menor valor real e o valor real associado ao menor valor predito.
	"""
	y_true_min = y_true.min()
	y_pred_min_pos = y_pred.argmin()
	y_expected_min = y_true[y_pred_min_pos]

	return (y_expected_min - y_true_min) / y_true_min

def train_min_edp_config_diff(trained_estimator: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> float:
	"""
	Função para fazer a predição e depois calcular diferença pondenrada entre o valor mínimo em y_true e o valor 
	real associado ao menor valor predito em y_true usando a função min_edp_config_diff.

	Parâmetros:

	  trained_estimator: BaseEstimator
		    Estimador usado para fazer a predição. O estimador precisa seguir a interface do scikit-learn
				para os estimadores.
	  X_test : Pandas dataframe
        Dataframe do Pandas com as características.
    y_pred : array_like(float)
        Series do Pandas com os valores reais da variável alvo da predição.
        
  Retorna:

	  float
        A diferença ponderada entre o menor valor real e o valor real associado ao menor valor predito.
	"""
	df_test_mean_EDP = pd.concat((X_test, y_test), axis=1).groupby(list(X_test.columns))[y_test.name].median().reset_index()
	X_test = df_test_mean_EDP[X_test.columns]
	y_test = df_test_mean_EDP[y_test.name]
	y_pred = trained_estimator.predict(X_test)

	return min_edp_config_diff(y_test, y_pred)

def neg_train_min_edp_config_diff(trained_estimator, X_test, y_test):
	return -train_min_edp_config_diff(trained_estimator, X_test, y_test)

# Item 5: Frequência em que a configuração sugerida e a configuração do oráculo coincidem: os tais 80%.
def min_edp_config_accuracy(X, y_true, y_pred):
	y_pred_argmin = y_pred.argmin()
	y_true_argmin = y_true.argmin()

	return float((X.iloc[y_pred_argmin] == X.iloc[y_true_argmin]).all())

def train_min_edp_config_accuracy(trained_estimator, X_test, y_test):
	df_test_mean_EDP = pd.concat((X_test, y_test), axis=1).groupby(list(X_test.columns))[y_test.name].median().reset_index()
	X_test = df_test_mean_EDP[X_test.columns]
	y_test = df_test_mean_EDP[y_test.name]
	y_pred = trained_estimator.predict(X_test)

	return min_edp_config_accuracy(X_test, y_test, y_pred)

class FilterOutliers:
	def __init__(self):
		self.dados = None
		self.dados_filtrados = None
		self.input_variables = None
		self.output_variables = None
		self.outliers_limit = None
		self.make_range = lambda a, b: (a-b, a+b)

	def make_outliers_filter(self, outliers_limit, variables):
		def outliers_filter(df):
			masks = []
			for v in variables:
				masks.append(~df[v].between(*self.make_range(df[v].median(), outliers_limit * st.median_abs_deviation(df[v]))))
			return pd.DataFrame(masks).T
		return outliers_filter

	def Filter(self, dados, input_variables, output_variables, outliers_limit):
		self.dados = dados
		self.input_variables = input_variables
		self.output_variables = output_variables
		self.outliers_limit = outliers_limit

		outlier_masks = dados.groupby(input_variables).apply(self.make_outliers_filter(outliers_limit, output_variables))

		non_outliers_mask = ~outlier_masks.any(axis=1)

		self.dados_filtrados = dados[non_outliers_mask.reset_index(list(range(len(input_variables))), drop=True)].reset_index().copy()

		return self.dados_filtrados
		
class BestHiperparams:
	def __init__(self, n_jobs=-1, verbose=False):
		self.X = None
		self.y = None
		self.grid_search_model = None
		self.groups = None
		self.group_names = None
		self.n_jobs = n_jobs
		self.verbose = verbose
	
	def optimize(self, data, suggestion_names, application_names, user_names, 
							 predicted_name, model, hiperparams_grid, scoring=train_min_edp_config_accuracy):
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Invalid input data provided, data is not a Dataframe.")

		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError(f"Invalid input suggestion_names provided, not all {suggestion_names} suggestions params exists in {data.columns}.")
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError(f"Invalid input application_names provided, not all {application_names} applications params exists in {data.columns}.")
		if not pd.Index([predicted_name]).isin(data.columns).all():
			raise KeyError(f"Invalid input predicted_name provided, predicted param {predicted_name} doesn't exists in {data.columns}.")

        # Define X e y
		self.X = data[suggestion_names+application_names]
		self.y = data[predicted_name]

        # Cria os grupos
		lab_encoder = skpp.LabelEncoder()
		self.groups = lab_encoder.fit_transform(list(map(str, data[user_names].values)))
		self.groups_names = lab_encoder.classes_

        # Cria o objeto de grid para otimizar os hiperparâmetros.
		grid_search_model = skms.GridSearchCV(
			model,
			cv=skms.LeaveOneGroupOut(),
			param_grid=hiperparams_grid,
			scoring=scoring,
			refit=True,
			n_jobs=self.n_jobs,
			return_train_score=True,
			verbose=int(self.verbose),
		)

    # Otimiza os hiperparâmetros.
		self.grid_search_model = grid_search_model.fit(self.X, self.y, groups=self.groups)
		
		# Retorna os resultados da otimização.
		return (self.grid_search_model.best_params_, self.grid_search_model.best_score_)
	
	def get_hrperparams_scores(self):
		if self.grid_search_model is None:
			raise ValueError("The model's hyperparameters have not yet been optimized.!")
		hiperparams_df = pd.DataFrame(self.grid_search_model.cv_results_)
		return hiperparams_df
		
class DiscoverBestModel:
	def __init__(self, n_jobs=-1, verbose=False):
		self.results_df	= None
		self.X = None
		self.y = None
		self.cv_results = None
		self.groups = None
		self.group_names = None
		self.mean_scores_models_df = None
		self.best_model_name = None
		self.best_model_score = None
		self.n_jobs = n_jobs
		self.verbose = verbose
		
	def best_model(self, data, suggestion_names, application_names, user_names, predicted_name, models, 
	               scores_functions={'accuracy': train_min_edp_config_accuracy, 'difference': neg_train_min_edp_config_diff}):
	
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Invalid input data provided, data is not a Dataframe.")

		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError(f"Invalid input suggestion_names provided, not all {suggestion_names} suggestions params exists in {data.columns}.")
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError(f"Invalid input application_names provided, not all {application_names} applications params exists in {data.columns}.")
		if not pd.Index([predicted_name]).isin(data.columns).all():
			raise KeyError(f"Invalid input predicted_name provided, predicted param {predicted_name} doesn't exists in {data.columns}.")

        # Define X e y
		self.X = data[suggestion_names+application_names]
		self.y = data[predicted_name]

        # Cria os grupos
		lab_encoder = skpp.LabelEncoder()
		self.groups = lab_encoder.fit_transform(list(map(str, data[user_names].values)))
		self.groups_names = lab_encoder.classes_

		# Cria um dataframe vazio
		self.results_df = pd.DataFrame()
		
		self.cv_results = {}

		# Avalia os modelos
		for name, model in models.items():
			cv_results = skms.cross_validate(
				model,
				data[suggestion_names+application_names],
				data[predicted_name],
				scoring=scores_functions,
				groups=self.groups,
				cv=skms.LeaveOneGroupOut(),
				n_jobs=self.n_jobs,
        return_indices=True,
				error_score='raise',
				return_estimator=True,
				#verbose=int(self.verbose)
			)
			cv_results_df = pd.DataFrame({k.replace("test_", ""): cv_results[k] for k in cv_results if k not in ['indices','estimator']})
			cv_results_df['Model'] = name
			self.cv_results[name] = { 'cv_results': cv_results, 'results_dataframe': cv_results_df}

			if self.verbose:
				print(f"\n\n➡️  Model {name} table:\n")
				print(cv_results_df.to_markdown(tablefmt="grid"))
				print(f"\n\n➡️  Model {name} statistics:\n")
				print(cv_results_df.describe())
				print("\n\n")

			self.results_df = pd.concat([self.results_df, cv_results_df])

		# Torna os índices consecutivos.
		self.results_df = self.results_df.reset_index(drop=True)		
		
		# Cria um dataframe com a média para os modelos.
		scores_functions_names = list(scores_functions.keys())
		self.mean_scores_models_df = self.results_df.groupby(by=['Model'])[scores_functions_names].mean()
		
		# Descobre o(s) melhor(es) modelos, faz isso ordenando o dataframe mean_scores_models_df pelas
		self.mean_scores_models_df = self.mean_scores_models_df.sort_values(by=scores_functions_names, ascending=False)
		self.best_model_name = self.mean_scores_models_df.index[0]
		self.best_model_scores = self.mean_scores_models_df.iloc[0].to_dict()
		
		return (self.best_model_name, self.best_model_scores, self.results_df, self.mean_scores_models_df)
	
	def get_results_model(self, model_name):
		if self.cv_results is None:
			raise ValueError("Cross-validation of the models has not yet been executed!")
		if not model_name in self.cv_results.keys():
			raise KeyError(f"Invalid model name {model_name}! Must be any in {self.cv_results.keys()}.")
			
		return self.cv_results[model_name]
			      
class SuggestionsPredictor:
	def __init__(self):
		self.suggestion_names = None
		self.application_names = None
		self.user_names = None
		self.predicted_name = None
		self.predicted_time_name = None
		self.predicted_memory_name = None
		self.dataset = None
		self.model = None
		self.model_time = None
		#self.model_memory = None
		self.suggestion_params = None
		self.application_params = None
		self.user_params = None
		self.X = None
		self.y = None
		self.y_time = None
		#self.y_memory = None
	    	
	def fit(self, data, suggestion_names, application_names, user_names, estimated_parameters, model, model_params, verbose=False):
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Invalid input data provided, data is not a Dataframe.")

		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError(f"Invalid input suggestion_names provided, not all {suggestion_names} suggestions params exists in {data.columns}.")
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError(f"Invalid input application_names provided, not all {application_names} applications params exists in {data.columns}.")
		if not pd.Index(user_names).isin(data.columns).all():
			raise KeyError(f"Invalid input user_names provided, not all {user_names} applications params exists in {data.columns}.")
		if not pd.Index(estimated_parameters.values()).isin(data.columns).all():
			raise KeyError(f"Invalid input predicted_name provided, one of the {estimated_parameters.values()} doesn't exists in {data.columns}.")

		self.suggestion_names = suggestion_names
		self.application_names = application_names
		self.user_names = user_names
		self.predicted_name = estimated_parameters['suggestion']
		self.model = model(**model_params)
		self.X = data[self.suggestion_names+self.application_names].copy()
		self.y = data[self.predicted_name].copy()

		# Treina o model com os dados
		self.model.fit(self.X, self.y)

    # Se a variável de tempo foi definida, também treina um modelo para predizer o tempo.
		if 'time' in estimated_parameters:
			if estimated_parameters['time'] == estimated_parameters['suggestion']:
				self.model_time = self.model	
				self.predicted_time_name = self.predicted_name 
				self.y_time = self.y
			else:	
				self.model_time = model(**model_params)
				self.predicted_time_name = estimated_parameters['time']
				self.y_time = data[self.predicted_time_name].copy()
				self.model_time.fit(self.X, self.y_time)
		else:
			self,model_time = None		

		# Salva o dataframe usado para treinar o modelo.
		colunms_names = list(set(suggestion_names+application_names+user_names+list(estimated_parameters.values())))
		self.dataset = data[colunms_names].copy()

		# Define os possíveis parâmetros para cada sugestão.
		self.suggestion_params = {col: list(data[col].unique()) for col in suggestion_names}

		# Define os possíveis parâmetros para cada opção da aplicação do usuário;
		self.application_params = {col: list(data[col].unique()) for col in application_names}

		# Define os possíveis parâmetros para cada opção da aplicação do usuário;
		self.user_params = {col: list(data[col].unique()) for col in user_names}
		
		if verbose:
			print(f"➡️  Parâmetros de sugestão usados no treinamento: {self.suggestion_params}")
			print(f"➡️  Parâmetros de aplicação usados no treinamento: {self.application_params}")
			print(f"➡️  Parâmetros de usuário usados no treinamento: {self.user_params}")
			print(f"➡️  Variável alvo do modelo auxiliar usado na escolha da melor sugestão: {self.predicted_name}")
			if not self.predicted_time_name is None:
				print(f"➡️  Varíavel alvo do modelo predizer o tempo da melhor sugestão: {self.predicted_time_name}")
			print("➡️  X usado no treinamento dos modelos:\n")
			print(self.X.to_markdown(tablefmt="grid", floatfmt=".2f" ))
			print("\n\n➡️  y usado no treinamento do modelo auxiliar:\n")
			print(self.y.to_markdown(tablefmt="grid", floatfmt=".2f"))
			if not self.model_time is None:
				print("➡️  \n\ny usado pelo modelo para a predição dos tempos:\n")
				print(self.y_time.to_markdown(tablefmt="grid", floatfmt=".2f"))
			print("➡️  \n\nDataframe contendo todas as variáceis usadas nos treinamentos:\n")
			print(data.to_markdown(tablefmt="grid", floatfmt=".2f"))
			print("\n\n")
			
		return self 		
		
	def get_oracle(self, verbose=False):
		# Calcula o dataset do oraculo.
		df_aux = self.dataset.groupby(self.suggestion_names+self.user_names)[self.predicted_name].median().reset_index()
		if verbose:
			print(f"\n\n➡️  Medianas da variável {self.predicted_name} para todas as repetições de cada combinação dos valores das variáveis {self.suggestion_names+self.user_names}\n")
			df_aux.to_markdown(tablefmt="grid", floatfmt=".2f")
		df_oracle = df_aux.groupby(self.user_names).apply(lambda x: x[x[self.predicted_name] == x[self.predicted_name].min()], include_groups=False)
		if verbose:
			print("\n\n➡️  Dataframe do oráculo\n\n")
			df_oracle.to_markdown(tablefmt="grid", floatfmt=".2f")

		return df_oracle
	
	def get_importances(self, verbose=True):
		if self.model is None:
			raise ValueError("The model hasn't been trained yet!")
		
		if hasattr(self.model, "feature_importances_"):
			importances = getattr(self.model, "feature_importances_")
			if hasattr(self.model, "feature_names_in_"):
				importances_names = getattr(self.model, "feature_names_in_").tolist()
			else:
				importances_names = self.suggestion_names

			importances_dict = {
				'names': importances_names,
				'values': importances
			}
			importances_df = pd.DataFrame(importances_dict)	

			if verbose:
				print(f"➡️  Diciońario com as importâncias: {importances_dict}\n\n")		
				print("\n\n➡️  Dataframe das importâncias\n\n")
				importances_df.to_markdown(tablefmt="grid", floatfmt=".2f")
		else:		
			if verbose:
				print("➡️  O estimador usado não define as importâncias das variáveis!")		
			importances_df = None
		
		return importances_df
		    	
	def predict_suggestions_data(self, user_applicaion_params, custom_suggestions_params=None, verbose=False):
	    # Verifica se o fit foi feito
		if self.model is None:
			raise ValueError("The model hasn't been trained yet!")
	    	
		if sorted(user_applicaion_params.keys()) != sorted(self.application_params.keys()):
			raise KeyError(f"Invalid application {user_applicaion_params.keys()} param names! Must be {self.application_params.keys()}")
	  			
		if custom_suggestions_params is None:
		  # Cria um X usando as opções de configuração usadas para treinar o modelo.
			X = self.dataset.groupby(self.suggestion_names)[self.predicted_name].median().reset_index().copy().drop(columns=[self.predicted_name])
		else:
		  # Cria um X usando as opções de configuração passadas como parâmetro
			suggestions_cobinations = list(itertools.product(*custom_suggestions_params.values()))
			X = pd.DataFrame(suggestions_cobinations, columns=custom_suggestions_params.keys())
	
		for param_name in user_applicaion_params.keys():
			X[param_name] = user_applicaion_params[param_name]

		# Mantém a ordem das colunas do dataframe original,
		X = X[self.X.columns]	
   	
		if verbose:
			print(f"➡️  X usado quando foi predito todos os valores da variável alvo {self.predicted_name} para todas as possíveis sugestões:\n")	
			print(X.to_markdown(tablefmt="grid", floatfmt=".2f"))
			print("\n\n")

		# Faz a predição para o X_aux.
		y_pred = self.model.predict(X)

		if verbose:
			print(f"➡️  y predito da variável alvo {self.predicted_name} para todas as possíveis sugestões:\n")	
			print(pd.Series(y_pred).to_markdown(tablefmt="grid", floatfmt=".2f"))

		return (y_pred, X)
	
	def get_suggestion(self, user_applicaion_params, custom_suggestions_params=None, verbose=False):
		# Faz a predição dos valores para todas as configurações da base usada para o treinamento e os parâmetros da aplicação passados.
		y_pred, X = self.predict_suggestions_data(user_applicaion_params, custom_suggestions_params, verbose)
        
     	# Descobre a posicao do menor valor predito e esse valor, que indicará a posição da configuração predita.
		y_pred_posmin = y_pred.argmin()
		y_pred_min = y_pred.min()
		if verbose:
			print(f"y mínimo predito para os parâmetros da aplicação {user_applicaion_params}: {y_pred_min} está na posição {y_pred_posmin} do vetor de predições!")
		# A sugestão será a configuração associada ao menor valor da variável predita.	
		y_suggestion = X.loc[y_pred_posmin,self.suggestion_names].to_dict()
		
        # Retorna a sugestão com o manor valor predito para a variável predita pelo modelo.
		y_pred_s = pd.Series(y_pred)
		y_pred_s.name = self.predicted_name		
		info_suggestion = {"Suggestion": y_suggestion, "Score": y_pred_min, "X": X, "y_pred": y_pred_s, "y_pred_minimum": y_pred_min, "y_pred_minimum_position": y_pred_posmin}

		# Verifica se podemos predizer o tempo de execução e/ou o consumo de memória
		if not self.model_time is None:
			X_dict_aux = y_suggestion | user_applicaion_params
			X_aux = pd.DataFrame(X_dict_aux, index=[0])
			if verbose:
				print(f"➡️  X auxiliar usado ao predizer a o tempo da melhor sugestão {y_suggestion}, using {user_applicaion_params}\n")
				print(X_aux.to_markdown(tablefmt="grid", floatfmt=".2f"))
				print("\n\n")
			if not self.model_time is None:
				y_time = self.model_time.predict(X_aux)
				info_suggestion["Time"] = y_time[0]
				if verbose:	
					print(f"➡️  Tempo de execução predito para a melhor configuração {y_suggestion}, using {user_applicaion_params}: {y_time[0]}") 

		return info_suggestion	                     	

	def get_suggestions(self, user_applications_params_df, custom_suggestions_params=None, verbose=False):
		if not type(user_applications_params_df) is pd.DataFrame:
			raise ValueError("Invalid input user_params_df provided, not a Pandas DataFrame.")
		if not pd.Index(self.application_names).isin(user_applications_params_df.columns).all():
			raise KeyError(f"Invalid input user_params_df provided, not all {self.application_names} user params exists in user predictions dataset.")
			
		info_suggestions = []	

		for idx in user_applications_params_df.index:
			user_params = user_applications_params_df.loc[idx].to_dict()
			if verbose:
				print(f"➡️  Definindo a sugestão para os parâmetros {user_params} do usuário")
			info_suggestion = self.get_suggestion(user_params, custom_suggestions_params, verbose)
			info_suggestions.append(info_suggestion)

		return info_suggestions		
		
	def predict(self, X):
	    # Verifica se o fit foi feito
		if self.model is None:
			raise ValueError("The model hasn't been trained yet!")

		return self.model.predict(X)			

	def score(self, X, y):
	    # Verifica se o fit foi feito
		if self.model is None:
			raise ValueError("The model hasn't been trained yet!")

		return self.model.score(X, y)			
		
	def get_params(self, deep=False):
		return {}	

	def save_predictor(self, file_name):
		with open(file_name, 'wb') as file:
			pickle.dump(self, file)		

	@classmethod
	def print_suggestion(cls, info_suggestion, show_score=False, show_X=False, show_y_pred=False, 
											 show_time=False, show_memory=False, suggestion_map=None):
		if suggestion_map is None:
			formatted_suggestion = ", ".join(f"{k}={v}" for k, v in info_suggestion['Suggestion'].items())
		else:
			formatted_suggestion = ", ".join(f"{suggestion_map[k]}={v}" for k, v in info_suggestion['Suggestion'].items())	
		print(f"➡️  Sugestão: {formatted_suggestion}")
		if show_time:
			if 'Time' in info_suggestion:
				print(f"➡️  Tempo para a sugestão: {info_suggestion['Time']} s")
		if show_score:
			print(f"➡️  Pontuação da sugestão: {info_suggestion['Score']}")
		if show_X:
			print('➡️  X usado nas predições feitas quando estavamos escolhendo a malhor sugestão:\n')
			print(info_suggestion['X'].to_markdown(tablefmt="grid", floatfmt=".2f"))
			print("\n\n")
		if show_y_pred:
			print(f"➡️  y pedito usado para escolher a melhor sugestão, sendo que o mínimo {info_suggestion['y_pred_minimum']} está na posição {info_suggestion['y_pred_minimum_position']}:\n")
			print(info_suggestion['y_pred'].to_markdown(tablefmt="grid", floatfmt=".2f"))
			print("\n\n")
		
	@classmethod
	def load_predictor(cls, file_name):
		with open(file_name, 'rb') as file:
			predictor = pickle.load(file)
		return predictor