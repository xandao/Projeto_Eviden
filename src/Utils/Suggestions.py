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

def min_edp_config_diff(y_true, y_pred):
	"""
	Função para calcular diferença pondenrada entre o valor mínimo em y_true e o valor real 
	associado ao menor valor predito em y_true.

	Parâmetros:
  	y_true (array_like[float]): Vetor de entrada com os valores reais das medidas.
	  y_pred (array_like[float]): Vetor de entrada com os valores preditos das medidas.
        
	Retorna:
	  float: A diferença ponderada entre o menor valor real e o valor real associado ao menor 
	         valor predito.
	"""

	# Obtém o menor valor real da variável alvo.
	y_true_min = y_true.min()

	# Obtém a posição do menor valor predito da variável alvo.
	y_pred_min_pos = y_pred.argmin()

	# Obtém o valor real associado ao menor valor predito, ou seja, o valor real que idealmente 
  # seria o predito.
	y_expected_min = y_true[y_pred_min_pos]

	# Retorna a diferençao ponderada entre o valor real do valor predito e o menor valor real.
	#
	#  [valor real para o valor predito] y_expected_min - y_true_min [menor valor real] 
	#	                                   ---------------------------
	#                                            y_true_min  
	#
	return (y_expected_min - y_true_min) / y_true_min

def train_min_edp_config_diff(trained_estimator, X_test, y_test):
	"""
	Função para fazer a predição para um dos grupos das variáveis da aplicação compposto por um 
	possível conjunto de valores para essas variáveis, e depois calcular diferença pondenrada 
	entre o valor mínimo em y_true e o valor real associado ao menor valor predito em y_true 
	usando a função min_edp_config_diff.

	Parâmetros:
  	trained_estimator (BaseEstimator): Estimador usado para fazer a predição. O estimador 
	                                     precisa seguir a interface do scikit-learn para os 
		  																 estimadores.
	  X_test (DataFrame): Um objeto Dataframe do Pandas com as variáveis das sugestões de 
	                      configuração.
  	y_test (Series): Um objeto Series do Pandas com os valores reais da variável alvo da 
	                   predição.
        
	Retorna:
  	float: A diferença ponderada entre o menor valor real e o valor real associado ao menor 
	  	     valor predito.
	"""

	# Recria o dataframe original juntando X e y, sendo o valor da variável alvo das execuções 
  # para um mesmo conjunto de valores das variáveis de sugestão de configuração e das 
  # variáveis da aplicação, execuções essas que existem para mitigar a variabilidade da 
  # execução compartilhada em um supercomputador, será a mediana dos valores de todas essas 
  # execuções.
	df_test_mean_EDP = pd.concat((X_test, y_test), 
															 axis=1).groupby(list(X_test.columns))[y_test.name].median().reset_index()

	# Determina o X_test de teste usado na predição (é um dos possíveis grupos definidos pelas 
  # possíveis combinações de parâmetros para as variáveis da aplicação).
	X_test = df_test_mean_EDP[X_test.columns]

	# Determina o y_test de teste a ser predito, sendo como observamos os valores sendo as 
  # mediadas das execuções repetidas (a variável alvo).
	y_test = df_test_mean_EDP[y_test.name]

	# Utiliza o modelo para fazer a predição para o  X_test, retornada em y_pred.
	y_pred = trained_estimator.predict(X_test)

	# Agora que temos os valores y_test (mediana dos valores reais da variável alvo para cada 
  # execução em X), usamos a função min_edp_config_diff para calculcar a pontuação de 
  # diferença.
	return min_edp_config_diff(y_test, y_pred)

def neg_train_min_edp_config_diff(trained_estimator, X_test, y_test):
	"""
	Função para fazer a predição para um dos grupos das variáveis da aplicação compposto por um 
	possível conjunto de valores para essas variáveis, e depois calcular diferença pondenrada 
	entre o valor mínimo em y_true e o valor real associado ao menor valor predito em y_true 
	usando a função min_edp_config_diff.

	Parâmetros:
    trained_estimator (BaseEstimator): Estimador usado para fazer a predição. O estimador 
                                      precisa seguir a interface do scikit-learn para os 
                                      estimadores.
    X_test (DataFrame): Um objeto Dataframe do Pandas com as variáveis das sugestões de 
                        configuração.
    y_test (Series): Um objeto Series do Pandas com os valores reais da variável alvo da 
                     predição.
        
	Retorna:
    float: O negativo da diferença ponderada entre o menor valor real e o valor real associado 
          ao menor valor predito.
	"""

	# Usa a função train_min_edp_config_diff para calcular a pontuação da diferença e retorna o 
  # negativo, ou simétrico, da pontuação da diferença. Isso é necessário porque a pontuação de ]
  # diferença é uma pontuação em que o mínimo é o melhor (0 o ideal), enquanto que as funções 
  # que usamos na busca em grade (para determinar os melhores hiperparâmetros) e valodação 
  # cruzada precisam de uma função que o menor valor seja o pior e o maior valor possível o 
  # menhor de todos).
	return -train_min_edp_config_diff(trained_estimator, X_test, y_test)

def min_edp_config_accuracy(X, y_true, y_pred):
	"""
	Função para calcular a pontuação de acurácia, que será igual a 1 se a sugestão de
	configuração, definida pelas variáveis de X que definem o conjunto de possíveis
	configurações de sugestão, e obtida considerando o menor valor em y_pred, com os
	valores preditos para a variável alvo para cada configuração em X, for igual a 
	sugestão de configuração definida pelo menor valor em y_true, ou seja, se for igual a
	configuração definida pelo oráculo, ou 0 em caso contrário, ou seja, se a configuração
	for diferente da configuração do oráculo.

	Parâmetros:
	  X (DataFrame): Um objeto DataFrame do Pandas com uma columa para cada variável em uma
                   sugestão de configuração.
	  y_true (array_like[float]): Vetor de entrada com os valores reais das medidas.
	  y_pred (array_like[float]): Vetor de entrada com os valores preditos das medidas.
        
  Retorna:
	  float: 1 se a sugestão de configuração definida por y_pred for igual a definida por 
		y_real, ou seja, igual ao oráculo, ou 0 em caso contrário.
	"""
	# Obtém a posição do menor valor real da variável alvo.
	y_true_argmin = y_true.argmin()

	# Obtém a posição do menor valor predito da variável alvo.
	y_pred_argmin = y_pred.argmin()

  # Usa o pandas para verificar se todos os valores das colunas da posição em X dada por 
  # y_pred_argmin em (ou seja, a sugestão de configuração) coincidem com os valores da
  # posição em X dada por y_pred_argmin (ou seja, o oráculo).
	return float((X.iloc[y_pred_argmin] == X.iloc[y_true_argmin]).all())

def train_min_edp_config_accuracy(trained_estimator, X_test, y_test):
	"""
	Função para fazer a predição para um dos grupos das variáveis da aplicação composto por um 
	possível conjunto de valores para essas variáveis, e depois calcular a acurácia da 
	sugestão de configuração associada ao valor mínimo predito para a variável alvo e o
  oráculo definido pelo menor valor em y_test. 

	Parâmetros:
  	trained_estimator (BaseEstimator): Estimador usado para fazer a predição. O estimador 
	                                     precisa seguir a interface do scikit-learn para os 
		  																 estimadores.
	  X_test (DataFrame): Um objeto Dataframe do Pandas com as variáveis das sugestões de 
	                      configuração.
  	y_test (Series): Um objeto Series do Pandas com os valores reais da variável alvo da 
	                   predição.
        
	Retorna:
  	float: 1 se a sugestão de configuração do menor valor predito for o oráculo e 0 em caso
		contrário.
	"""

	# Recria o dataframe original juntando X e y, sendo o valor da variável alvo das execuções 
  # para um mesmo conjunto de valores das variáveis de sugestão de configuração e das 
  # variáveis da aplicação, execuções essas que existem para mitigar a variabilidade da 
  # execução compartilhada em um supercomputador, será a mediana dos valores de todas essas 
  # execuções.
	df_test_mean_EDP = pd.concat((X_test, y_test), 
															 axis=1).groupby(list(X_test.columns))[y_test.name].median().reset_index()

	# Determina o X_test de teste usado na predição (é um dos possíveis grupos definidos pelas 
  # possíveis combinações de parâmetros para as variáveis da aplicação).
	X_test = df_test_mean_EDP[X_test.columns]

	# Determina o y_test de teste a ser predito, sendo como observamos os valores sendo as 
  # mediadas das execuções repetidas (a variável alvo).
	y_test = df_test_mean_EDP[y_test.name]

	# Utiliza o modelo para fazer a predição para o  X_test, retornada em y_pred.
	y_pred = trained_estimator.predict(X_test)

	# Agora que temos os valores y_test (mediana dos valores reais da variável alvo para cada 
  # execução em X), usamos a função min_edp_config_accuracy para calculcar a acurária da
	# sugestão de configuração definida pelo menor valor em y_pred.
	return min_edp_config_accuracy(X_test, y_test, y_pred)

class FilterOutliers:
	"""
	Classe para fazer a filtragem dos outliers da base de dados a ser usada quando os modelos 
	forem treinados, com o objetivo de remover os valores das variáveis usadas no treinamento 
	que sejam muito discrepantes considerando todos os valores de cada uma das variáveis 
	escolhidas para fazer a filtragem. A filtragem, para cada uma dessas variáveis, será 
	feita usando o desvio absoluto em relação à mediada dos valores dessa variável.

	Atributos:
		dados (DataFrame | None): armazena a referência para o objeto do DataFrame do Pandas 
		com o conjunto de dados original, antes da filtragem.
		dados_limpos: (DataFrame | None): armazena a referência para o objeto do DataFrame do 
		Pandas com o novo conjunto de dados obtido após a flitragem do conjunto original.
		input_variables (list[str]): nomes das variáveis de entrada, ou características, usadas 
		nos treinamentos dos modelos. Este conjunto será composto pelas variáveis associadas as
		sugestões de configuração e as variáveis da aplicação usadas ao treinar os modelos e
		definidas pelos usuários, e as variáveis da aplicação usadas para construir os grupos
		da validação LOGO usada na busca em grade, na validação cruzada e no treinamento dos
		modelos.
		filter_variables (list[str]): variáveis usadas para fazer à filtragem do conjunto de 
		dados original. São todas variáveis obtidas pelas informação obtidas referentes às 
		execuções de cada teste do conjunto de dados original. A variável alvo do treinamento
		dos diversos modelos é uma dessas variáveis.
		outliers_limit (float): Valor de ponto flutuante com o fator multiplicador usado para 
		definir os limites inferior e superior de acordo com o desvio mediano absoluto, 
		sendo os limites definidos em relação à mediana. Todos os valores fora da faixa 
		definidos por estes limites, para cada variável em filter_variables, serão considerados 
		como outlires e serão removidos do novo conjunto de dados dados_filtrados.
		make_range (lambda): Função anômina que, dado dos valores a (float) e b (float), cria 
		uma tupla definindo o intervalo [a-b,a+b].

	"""
	def __init__(self):
		"""
    Função de inicialização da classe FilterOutliers.

		Parâmetros:
      Não tem parâmetros.
		"""

		# Armazena uma referência para o conjunto de dados original.
		self.dados = None
		# Armazena uma referência para o conjunto de dados flitrado.
		self.dados_filtrados = None
		# Armazena uma referência para a lista com as variáveis usadas ao treinar os diversos 
		# modelos.
		self.input_variables = None
		# Armazena uma referência para a lista com as variáveis usadas na filtragem
		self.filter_variables = None
		# Armazena o valor de ponto flutuante que define o limite ao redir do desvio mediano
		# absoluto. Os outliers estarão fora deste limite.
		self.outliers_limit = None
		# Define uma função anômina para, dados dois valores de ponto flutuante a e b, definir
		# em uma tupla o intervalo (a-b,a+b).
		self.make_range = lambda a, b: (a-b, a+b)

	def make_outliers_filter(self, outliers_limit, variables):
		"""
		Função para criar uma função de filtragem customizada do conjunto de dados a ser 
		filtrado, usando cada variável em v para fazer a filtragem dos outliers, sendo
		que, para cada variável em v e o desvido mediano absoluto dos valores dm em v, os
		outliers referente a v serão os testes que, considerando todos o valor de v para
		cada teste e a mediana m dos valores de v, serão os que estão fora do intervalo
		[d - outliers_limit x dm,d + outliers_limit x dm], onde outliers_limit é um valor 
		de ponto flutuante definindo a faixa de tolerância para os valores da variável v.

		Parâmetros:
			outliers_limit (float): Valor de ponto flutuante para definir o fator de 
			multiplicação ao definir os limites inferior e superior baseados no desvio
			mediano absoluto e a média.
			variables (list[str]): Variáveis em que a filtragem será baseada.

		Retorna:
			func: Uma função customizada do Python com a função que define a máscara para
			filtrar os testes do conjunto de dados com outliers.
		"""

		def outliers_filter(df):
			"""
			Função para filtrar o objeto DataFrame do Pandas passado como eferência em df 
			de acordo com os parâmetros outliers_limit e variables descrito anteriormente.

			Parâmetros:
				df (DataFrame): Referência para o objeto DataFrame do Pandas com o conjunto de
 												dados a ser filtrado. 
			Retorna:
				DataFrame: Uma referência para um objeto do Pandas com uma máscara para filtrar os
				outliers para cada variável v em variables.
			"""

      # Inicializa a lista com os índices, em df, dos testes para os quais existem outliers
			# pelo menos uma das variáveis de filtragem em variables. 
			masks = []

			# Atualiza a lista dos íncides dos testes que tem outliers para cada variável v em 
			# variables.
			for v in variables:
				# Determina os índices dos testes em df que possuem outliers para a variável v.
				# Para fazer isso, primeiramente geramos uma referência para um objeto Series do 
				# Pandas que, para cada teste em df, define um valor booleano (bool) True se o 
				# valor do teste para a variável v está dentro do intervalo [m - outliers_limit 
				# x mad, m + outliers_limit x mad], onde m e a mediana dos valores para todos os 
				# testes em v, mad é o desvio mediano absoluto dos valores para todos os testes 
				# em v, e False em caso contrário. Depois, basta fazer a negação booleana dos
				# valores obtidos, pois para filtrar um DataFrame precisamos que as posições a
				# serem escolhidas e, no caso, removidas, seja, True e não False. Finalmente,
				# os valores com as máscaras booleanas para a variável v serão adicionadas ao
				# vetor masks.
				masks.append(~df[v].between(*self.make_range(df[v].median(), 
																		outliers_limit * st.median_abs_deviation(df[v]))))

			# Retorna uma referência para um objeto DataFrame do Pandas com as colunas sendo
			# as máscaras para cada teste, sendo a coluna rotulada pelo índice do teste, e os 
			# índices das linhas sendo cada uma das variáveis em variable, implicando que una
			# linha com índice v e coluna com rótulo t indica se o valor da linha v e da 
			# coluna t é um outlier, se True, ou não, se False, ou seja, se True o valor do
			# teste indexado por t para a variável v é um outlier e este teste deverá portanto
			# ser removido.
			return pd.DataFrame(masks).T

		# Retorna a função que define as máscaras de exclusão dos outliers para cada 
		# variável v em variables.
		return outliers_filter

	def Filter(self, dados, input_variables, filter_variables, outliers_limit):
		"""
		Função para fazer a filtragem dos outliers no conjunto de dados referenciado por dados,

		Parâmetros:
			dados (DataFrame): Conjunto de dados para o qual os outliers serão filtrados.
				input_variables (list[str]): nomes das variáveis de entrada, ou características, usadas 
																		 nos treinamentos dos modelos. 
				filter_variables (list[str]): variáveis usadas para fazer à filtragem do conjunto de 
																  		dados original. 
				outliers_limit (float): valor de ponto flutuante para definir o fator multiplicador 
																que será usado para definir os limites inferior e superior com 
																o desvio mediano absoluto e a mediana.


		"""

    # Define a variável dados com o conjunto de dados original e não filtrado.
		self.dados = dados
		# Define a variável input_variables com as variáveis de entrada.
		self.input_variables = input_variables
		# Define a variável filter_variables com as variáveis usadas para fazer a filtragem do
		# conjunto de dados.
		self.filter_variables = filter_variables
		# Define o valor em ponto flutuante usado para definir, conjuntamente com o desvio médio
		# absoluto e a mediana.
		self.outliers_limit = outliers_limit

    # Usa a função make_outliers_filter para obter a máscara de testes do conjunto de dados 
		# com as posições dos testes que precisarão ser removidos devido aos seus valores serem
	  # para uma variável v serem outliers, para cada variável v em filter_variables. O 
		# DataFrame retornado será indexado pelos nomes das variáveis em filter_variables, e
		# os rótulos das colunas serão cada um dos possíveis índices dos testes em data, de
		# tal modo que uma célula (v, t) deste DataFrame, se True, indicará que o valor da 
		# variável v para o teste t é um outlier e, em caso contrário, que não é um outlier.
		# Como desejamos remover os outliers, devemos remover cada teste t para o pelo menos
		# para uma varíavel v o valor do teste para esta variável foi um outlier, ou seja, se
		# existir pelo menos uma variável v para a qual a célula (v, t) tem o valor True.
    #
		# Depois da função ser chamada e retornar o DataFrame descrito anteriormente, precisamos
		# preparar o conjunto de dados dados para a filtragem. Para isso, primeiramente agrupamos
		# todas as colunas referentes às variáveis usadas como características nos treinamentos
		# dos modelos, ou seja, as variáveis qie fazem parte de uma sugestão de configuração, as
		# variáveis da aplicação definidas direta ou indiretamente pelo usuário e as variáveis
		# usadas para definir os grupos usados pela validação LOGO, que sáo em geral as mesmas
		# qua foram convertidas, mas podem também ser as passadas pelo usuário sem uma conversão
		# Uma vez feito isso, as colunas restantes após o agrupamento serão somente as usadas pela
		# filttagem, pois supomos que o DataFrame somente tem as colunas citadas anterioremente e
		# as das variáveis usadas pela filtragem, que são as variáveis referentes à execução das
		# aplicações obtidas pelo sacct após a execução de cada teste. Quando lemos o conjunto de
		# dados, sempre lemos somente as variáveis usadas no treinamento (como variáveis alvo) ou 
		# na filtragem. Com o agrupamento feito, a função apply é usada para definir para cada
		# combinação das características e para cada variável de filtragem, o estado de cada um dos
		# testes feitos para a combinação, que são as repetições para mitigar oscilações nos dados
		# obtidos pelo sacct devido ao uso compartilhado do supercomputador.
		outlier_masks = dados.groupby(input_variables).apply(self.make_outliers_filter(outliers_limit, 
																																								   filter_variables))

    # Depois de obtido o dataframe outlier_masks anterior verificamos, para cada possível teste,
		# se a máscara indica que existe algum outlier para pelo menos uma das variáveis usadas para
		# a flitragem dadas em filter_variables. Depois de obter a máscara fina para cada combinação, 
		# obtemos a negação lógica dela, pois isso tornará as posições de todos os testes em que existe
		# pelo menos um outlier como False e as que nçao tem nenhum outlier como True, fazendo 
		# efetivamente com que a máscara agora escolha os testes para os quais não existiram outliers
		# para todas as variáveis em filter_variables.
		non_outliers_mask = ~outlier_masks.any(axis=1)

    # Usa a máscara para escolher somente os testes para os quais não foram encontrados outliers 
		# em todas as variáveis em filter_variables, e armazena o conjunto de dados obtido após a
		# filtragem, na varável dados_filtrados do objeto da classe instanciado.
		self.dados_filtrados = dados[non_outliers_mask.reset_index(list(range(len(input_variables))), 
																														   drop=True)].reset_index().copy()

    # Retorna uma referência para o conjunto de dados filtrado, sem os testes com pelo menos un
		# outlier.
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
				print(f"➡️  Model {name} table:")
				print("\n", cv_results_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
				print(f"➡️  Model {name} statistics:")
				print("\n", cv_results_df.describe().to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

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
			if self.predicted_time_name is not None:
				print(f"➡️  Varíavel alvo do modelo predizer o tempo da melhor sugestão: {self.predicted_time_name}")
			print("➡️  X usado no treinamento dos modelos:")
			print("\n", self.X.to_markdown(tablefmt="grid", floatfmt=".2f" ), "\n", sep="")
			print("➡️  y usado no treinamento do modelo auxiliar:")
			print("\n", self.y.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
			if self.model_time is not None:
				print("➡️  y usado pelo modelo para a predição dos tempos:")
				print("\n", self.y_time.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
			print("➡️  Dataframe contendo todas as variáceis usadas nos treinamentos:")
			print("\n", data.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
			
		return self 		
		
	def get_oracle(self, verbose=False):
		# Calcula o dataset do oraculo.
		df_aux = self.dataset.groupby(self.suggestion_names+self.user_names)[self.predicted_name].median().reset_index()
		if verbose:
			print(f"➡️  Medianas da variável {self.predicted_name} para todas as repetições de cada combinação dos valores das variáveis {self.suggestion_names+self.user_names}:")
			print("\n", df_aux.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
		df_oracle = df_aux.groupby(self.user_names).apply(lambda x: x[x[self.predicted_name] == x[self.predicted_name].min()], include_groups=False)
		if verbose:
			print("➡️  Dataframe do oráculo:")
			print("\n", df_oracle.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

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
				print(f"➡️  Diciońario com as importâncias: {importances_dict}")		
				print("➡️  Dataframe das importâncias")
				print("\n", importances_df.to_markdown(tablefmt="grid", floatfmt=".2f"))
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
			print(f"➡️  X usado quando foi predito todos os valores da variável alvo {self.predicted_name} para todas as possíveis sugestões:")	
			print("\n", X.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

		# Faz a predição para o X_aux.
		y_pred = self.model.predict(X)

		if verbose:
			print(f"➡️  y predito da variável alvo {self.predicted_name} para todas as possíveis sugestões:")	
			print("\n", pd.Series(y_pred).to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

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
		if self.model_time is not None:
			X_dict_aux = y_suggestion | user_applicaion_params
			X_aux = pd.DataFrame(X_dict_aux, index=[0])
			if verbose:
				print(f"➡️  X auxiliar usado ao predizer a o tempo da melhor sugestão {y_suggestion}, using {user_applicaion_params}:")
				print("\n", X_aux.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

      # Fazendo a predição do tempo.
			y_time = self.model_time.predict(X_aux)

      # Obtendo o tempo predito.
			info_suggestion["Time"] = y_time[0]
			if verbose:	
				print(f"➡️  Tempo de execução predito para a melhor configuração {y_suggestion}, using {user_applicaion_params}: {y_time[0]}") 

		return info_suggestion	                     	

	def get_suggestions(self, user_applications_params_df, custom_suggestions_params=None, verbose=False):
		if not isinstance(user_applications_params_df, pd.DataFrame):
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
											 show_time=False, suggestion_map=None):
		if suggestion_map is None:
			formatted_suggestion = ", ".join(f"{k}={v}" for k, v in info_suggestion['Suggestion'].items())
		else:
			formatted_suggestion = ", ".join(f"{suggestion_map[k]}={v}" for k, v in info_suggestion['Suggestion'].items())	
		print(f"➡️  Sugestão: {formatted_suggestion}")
		if show_time:
			if 'Time' in info_suggestion:
				print(f"➡️  Tempo para a sugestão: {info_suggestion['Time']:.2f} s")
		if show_score:
			print(f"➡️  Pontuação da sugestão: {info_suggestion['Score']:.2f}")
		if show_X:
			print('➡️  X usado nas predições feitas quando estavamos escolhendo a malhor sugestão:')
			print("\n", info_suggestion['X'].to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
		if show_y_pred:
			print(f"➡️  y predito usado para escolher a melhor sugestão, sendo que o mínimo {info_suggestion['y_pred_minimum']} está na posição {info_suggestion['y_pred_minimum_position']}:")
			print("\n", info_suggestion['y_pred'].to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
		
	@classmethod
	def load_predictor(cls, file_name):
		with open(file_name, 'rb') as file:
			predictor = pickle.load(file)
		return predictor