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
	Função para calcular diferença pondenrada entre o valor mínimo em 
	y_true e o valor real associado ao menor valor predito em y_pred.

	Parâmetros:
  	y_true (array_like[float]): Vetor de entrada com os valores reais
																das medidas.
	  y_pred (array_like[float]): Vetor de entrada com os valores preditos
																das medidas.
        
	Retorna:
	  float: A diferença ponderada entre o menor valor real e o valor real 
					 associado ao menor valor predito.
	"""

	# Obtém o menor valor real da variável alvo.
	y_true_min = y_true.min()

	# Obtém a posição do menor valor predito da variável alvo.
	y_pred_min_pos = y_pred.argmin()

	# Obtém o valor real associado ao menor valor predito, ou seja, o 
	# valor real que idealmente seria igual ao predito.
	y_expected_min = y_true[y_pred_min_pos]

	# Retorna a diferença ponderada entre o valor real do valor predito
	# e o menor valor real. Na equação a seguir, para calcular a 
	# diferença:
  #
  # y_expected_min -> valor real associado ao menor valor predito.
	# y_true_min -> menor valor real, ou seja, o valor do oráculo.
	# 
	#  y_expected_min - y_true_min
	#	 ---------------------------
	# 				 y_true_min  
	#
	return (y_expected_min - y_true_min) / y_true_min

def train_min_edp_config_diff(trained_estimator, X_test, y_test):
	"""
	Função para inicialmente fazer a predição para o conjunto de possíveis 
	valores das variáveis de configuração e da aplicação definidos em 
	X_test, sendo y_test os valores reais da variável alvo associadas às 
	possíveis combinações dos valores das variáveis. Depois, usa o y_test 
	como y_real e o y_pred com os valores preditos para X_test, e a função 
	train_min_edp_config_diff anterior para calcular a diferença 
	ponderada e retornar a diferença negada. 

	Parâmetros:
  	trained_estimator (BaseEstimator): Modelo usado para fazer a  
																			 predição. O modelo precisa seguir
																			 a interface do scikit-learn para
																			 os modelos.
	  X_test (DataFrame): Um objeto Dataframe do Pandas com todas as 
												combinações das variáveis de configuração e da 
												aplicação relevantes para o teste feito ao
												chamar a função (para o LOGO, por exemplo, 
												X_test terá todas as combinações de configuração
												e valores fixos para os parâmetros da aplicação,
												já que a função é chamada para cada grupo do
												LOGO, e cada grupo é definido por um conjunto
												fixo de valores para as variáveis da aplicação).
  	y_test (Series): Um objeto Series do Pandas com os valores reais da 
										 variável alvo da predição.
        
	Retorna:
  	float: O negativo da diferença ponderada entre o menor valor real e 
					 o valor real associado ao menor valor predito.
	"""

	# Recria o dataframe original juntando X_test e y_test, sendo que
	# calculamos, para cada combinação dos valores das variáveis de
	# configuração e da aplicação, a mediana de todos os testes repetidos
	# para essa combinação, sendo que esses testes existem para mitigar 
	# a variabilidade da execução compartilhada em um supercomputador.
	df_test_mean_alvo = (
		pd.concat((X_test, y_test), axis=1)
		.groupby(list(X_test.columns))[y_test.name]
		.median()
		.reset_index()
	)

	# Determina o X_test de teste a ser usado na predição (as combinações 
	# das variáveis de configuração e de aplicação), mas usando a mediana
	# dos testes repetidos ao invés de cada teste.
	X_test = df_test_mean_alvo[X_test.columns]

	# Determina o y_test do teste a ser predito, sendo como observamos os
	# valores sendo as mediadas das execuções repetidas (a variável alvo).
	y_test = df_test_mean_alvo[y_test.name]

	# Utiliza o modelo para fazer a predição para o X_test, retornada em
	# y_pred.
	y_pred = trained_estimator.predict(X_test)

	# Agora que temos os valores y_test (mediana dos valores reais da
	# variável alvo para cada execução em X), usamos a função 
	# min_edp_config_diff para calculcar a pontuação de diferença.
	return min_edp_config_diff(y_test, y_pred)

def neg_train_min_edp_config_diff(trained_estimator, X_test, y_test):
	"""
	Função para fazer a predição para o conjunto de possíveis valores
	das variáveis de configuração e da aplicação definidos em X_test,
	sendo y_test os valores reais da variável alvo associadas às possíveis
	combinações dos valores das variáveis. Depois, usa o y_test como 
	y_real e o y_pred com os valores preditos para X_test, e a função 
	min_edp_config_diff anterior para calcular a diferença ponderada.

	Parâmetros:
  	trained_estimator (BaseEstimator): Modelo usado para fazer a 
																			 predição. O modelo precisa seguir
																			 a interface do scikit-learn para
																			 os modelos.
	  X_test (DataFrame): Um objeto Dataframe do Pandas com todas as 
												combinações das variáveis de configuração e da 
												aplicação relevantes para o teste feito ao
												chamar a função (para o LOGO, por exemplo, 
												X_test terá todas as combinações de configuração
												e valores fixos para os parâmetros da aplicação,
												já que a função é chamada para cada grupo do
												LOGO, e cada grupo é definido por um conjunto
												fixo de valores para as variáveis da aplicação).
  	y_test (Series): Um objeto Series do Pandas com os valores reais da 
										 variável alvo da predição.
        
	Retorna:
  	float: A diferença ponderada entre o menor valor real e o valor real 
					 associado ao menor valor predito.
	"""

	# Usa a função train_min_edp_config_diff para calcular a pontuação 
	# da diferença e retorna o negativo, ou simétrico, da pontuação da 
	# diferença. Isso é necessário porque a pontuação de diferença é uma
	# pontuação em que o mínimo é o melhor (0 o ideal), enquanto que as
	# funções que usamos na busca em grade (para determinar os melhores
	# hiperparâmetros) e valodação cruzada precisam de uma função que o
	# menor valor seja o pior e o maior valor possível o menhor de todos).
	return -train_min_edp_config_diff(trained_estimator, X_test, y_test)

def min_edp_config_accuracy(X, y_true, y_pred):
	"""
	Função para calcular a pontuação de acurácia, que será igual a 1 se a 
	sugestão de configuração, definida pelas variáveis em X associada a 
	posição com o menor valor em y_pred, for igual a sugestão de 
	configuração definida pelo menor valor em y_true, ou seja, se for 
	igual a configuração definida pelo oráculo, ou 0 em caso contrário, ou 
	seja, se a configuraçãofor diferente da configuração do oráculo.

	Parâmetros:
	  X (DataFrame): Um objeto DataFrame do Pandas com uma columa para 
									 cada variável em uma sugestão de configuração.
	  y_true (array_like[float]): Vetor de entrada com os valores reais 
																das medidas.
	  y_pred (array_like[float]): Vetor de entrada com os valores preditos 
																das medidas.
        
  Retorna:
	  float: 1 se a sugestão de configuração definida por y_pred for 
					 igual a definida por y_real, ou seja, igual ao oráculo, ou 0
					 em caso contrário.
	"""

	# Obtém a posição do menor valor real da variável alvo.
	y_true_argmin = y_true.argmin()

	# Obtém a posição do menor valor predito da variável alvo.
	y_pred_argmin = y_pred.argmin()

  # Usa o Pandas para verificar se todos os valores das colunas da 
	# posição em X dada por y_pred_argmin em (ou seja, a sugestão de 
	# configuração) coincidem com os valores da posição em X dada por 
	# y_pred_argmin (ou seja, o oráculo).
	return float((X.iloc[y_pred_argmin] == X.iloc[y_true_argmin]).all())

def train_min_edp_config_accuracy(trained_estimator, X_test, y_test):
	"""
	Função para inicialmente fazer a predição para o conjunto de possíveis 
	valores das variáveis de configuração e da aplicação definidos em 
	X_test, sendo y_test os valores reais da variável alvo associadas às 
	possíveis combinações dos valores das variáveis. Depois, calcula a 
	pontuação de acurácia, que será igual a 1 se a sugestão de 
	configuração, obtida considerando	o menor valor em y_pred, for igual a
	sugestão de configuração definida pelo menor valor em y_true, ou seja, 
	se for igual a configuração definida pelo oráculo, ou 0 em caso 
	contrário, ou seja, se a configuraçãofor diferente da configuração do 
	oráculo.

Parâmetros:
		trained_estimator (BaseEstimator): modelo usado para fazer a
																			 predição. O modelo precisa seguir
																			 a interface do scikit-learn para
																			 os modelos.
		X_test (DataFrame): Um objeto Dataframe do Pandas com todas as 
												combinações das variáveis de configuração e da 
												aplicação relevantes para o teste feito ao
												chamar a função (para o LOGO, por exemplo, 
												X_test terá todas as combinações de configuração
												e valores fixos para os parâmetros da aplicação,
												já que a função é chamada para cada grupo do
												LOGO, e cada grupo é definido por um conjunto
												fixo de valores para as variáveis da aplicação).
		y_test (Series): Um objeto Series do Pandas com os valores reais da 
										 variável alvo da predição.
	        
  Retorna:
	  float: 1 se a sugestão de configuração definida por y_pred for 
					 igual a definida por y_real, ou seja, igual ao oráculo, ou 0
					 em caso contrário.
	"""

	# Recria o dataframe original juntando X_test e y_test, sendo que
	# calculamos, para cada combinação dos valores das variáveis de
	# configuração e da aplicação, a mediana de todos os testes repetidos
	# para essa combinação, sendo que esses testes existem para mitigar 
	# a variabilidade da execução compartilhada em um supercomputador.
	df_test_mean_EDP = (
		pd.concat((X_test, y_test), axis=1)
		.groupby(list(X_test.columns))[y_test.name]
		.median()
		.reset_index()
	)

	# Determina o X_test de teste a ser usado na predição (as combinações 
	# das variáveis de configuração e de aplicação), mas usando a mediana
	# dos testes repetidos ao invés de cada teste.
	X_test = df_test_mean_EDP[X_test.columns]

	# Determina o y_test do teste a ser predito, sendo como observamos os
	# valores sendo as mediadas das execuções repetidas (a variável alvo).
	y_test = df_test_mean_EDP[y_test.name]

	# Utiliza o modelo para fazer a predição para o X_test, retornada em
	# y_pred.
	y_pred = trained_estimator.predict(X_test)

	# Agora que temos os valores y_test (mediana dos valores reais da
	# variável alvo para cada execução em X), usamos a função 
	# min_edp_config_accuracy para calculcar a acurária da sugestão de 
	# configuração definida pelo menor valor em y_pred.
	return min_edp_config_accuracy(X_test, y_test, y_pred)

class FilterOutliers:
	"""
	Classe para fazer a filtragem dos outliers da base de dados a ser 
	usada quando os modelos forem treinados, com o objetivo de remover, 
	para cada teste definido por uma combinação dos possíveis valores das
	variáveis de configuração e da aplicação, as repetições deste teste
	que sejam muito discrepantes considerando a mediana dessas repetições
	para cada variável usada para fazer a filtragem. A filtragem, para 
	cada uma dessas variáveis, será feita usando o desvio absoluto em 
	relação à mediana dos valores dessa variável para as repetições do
	teste.

	Atributos:
		dados (DataFrame | None): Armazena uma referência para o objeto do  
		                          DataFrame do Pandas com o conjunto de 
															dados original antes da filtragem.
		dados_limpos: (DataFrame | None): Armazena a referência para o 
		                                  objeto do DataFrame do Pandas
																			com o novo conjunto de dados
																			obtido após a filtragem do 
																			conjunto original.
		input_variables (list[str]): Nomes das variáveis de entrada, ou 
																 características, usadas nos 
																 treinamentos dos modelos. Este conjunto
																 será composto pelas variáveis 
																 associadas as sugestões de configuração
																 e as variáveis da aplicação definidas 
																 pelos usuários.
		filter_variables (list[str]): Variáveis usadas para fazer à
																	filtragem do conjunto de dados 
																	original. São todas variáveis com as
																	informação obtidas referentes às 
																	execuções de cada teste e cada 
																	repetição deste teste do conjunto de
																	dados original relevantes à predição 
																	da variável alvo.
		outliers_limit (float): Valor em ponto flutuante com o fator
													  multiplicador usado para definir os limites
														inferior e superior de acordo com o desvio
														mediano absoluto, sendo os limites definidos
														em relação à mediana. Todos os valores fora
														da faixa definida por estes limites, para
														cada variável em filter_variables e cada
														repetição de um teste, serão considerados 
														como outlires e serão removidos do novo 
														conjunto de dados.
		make_range (lambda): Função anômina que, dado dos valores a (float) 
												 e b (float), cria uma tupla definindo o 
												 intervalo [a-b,a+b].
	"""

	def __init__(self):
		"""
    Função de inicialização da classe FilterOutliers.

		Parâmetros:
      Não tem parâmetros.
		"""

		# Armazena uma referência para o conjunto de dados original.
		self.dados = None
		# Armazena uma referência para o conjunto de dados filtrado.
		self.dados_filtrados = None
		# Armazena uma referência para a lista com as variáveis usadas ao 
		# treinar os diversos modelos.
		self.input_variables = None
		# Armazena uma referência para a lista com as variáveis usadas na 
		# filtragem
		self.filter_variables = None
		# Armazena o valor de ponto flutuante que define o limite ao redor 
		# do desvio mediano absoluto. Os outliers estarão fora deste limite.
		self.outliers_limit = None
		# Define uma função anômina para, dados dois valores de ponto 
		# flutuante a e b, definir em uma tupla o intervalo (a-b, a+b).
		self.make_range = lambda a, b: (a-b, a+b)

	def make_outliers_filter(self, outliers_limit, variables):
		"""
		Função para criar uma função de filtragem para ser usada com a 
		função apply aplicada a um grupo do Pandas, sendo que cada grupo tem
		um DataFrame composto por todas as repetições de um mesmo teste 
		definido pelas possíveis combinações das variáeis de configuração e
		da aplicação, pois o agrupamento em que a função apply será usada é
		feito por essas variáveis.

		Parâmetros:
			outliers_limit (float): Valor em ponto flutuante definindo o 
															fator de multiplicação usado ao definir os 
															limites inferior e superior baseados no
															desvio mediano absoluto e a mediana.
			variables (list[str]): Variáveis cujos valores serão usados para
														 filtrar os outliers.

		Retorna:
			func: Uma função customizada do Python com a função que define a 
						máscara para filtrar as repetições, de um mesmo teste, 
						consideradas como outliers, de acordo com a mediana e o 
						desvio mediano absoulto dos valores das variáveis em
						variables para essas repetições.
		"""

		def outliers_filter(df):
			"""
			Função para filtrar um objeto DataFrame do Pandas, passado como
			referência em df, de acordo com os parâmetros outliers_limit e 
			variables descrito anteriormente. Como esta função será a usada no
			apply após o agrupamento, df será um DataFrame em que as linhas
			serão as repetições feitas para um mesmo teste, definido pelo
			agrupamento das variáveis de configuração e da aplicação, e as
			colunas serão as variáveis usadas na filtragem.

			Parâmetros:
				df (DataFrame): Referência para um objeto DataFrame do Pandas  
												com os dados das repetições de um teste.

			Retorna:
				DataFrame: Uma referência para um DataFrame do Pandas com uma
				máscara para filtrar os outliers para cada variável v em
				variables do DataFrame df, ou seja, indicando, para cada 
				repetição, associada a uma linha do DataFarme, e para cada 
				variável v de variables, se a repetição tem (True) ou não 
				(False) um outlier para a variável v. 
			"""

      # Inicializa a lista com os índices, em df, da repetições para as
			# quais existem outliers para pelo menos uma das variáveis de 
			# filtragem em variables. 
			masks = []

			# Atualiza a lista dos índices das repetições que tem outliers 
			# considerando os valores dessas repetições para cada variável v 
			# em variables. 
			for v in variables:				
        # Para descobrir os outliers, primeiramente precisamos 
				# determinar, para a variável v, o intervalo em que os valores 
				# desta variável para o teste precisarão estar para não serem 
				# outliers. Depois de estudos, decidimos usar um intervalo 
				# baseado no desvido mediano absoluto, que é similar ao desvio
				# padrão, mas avalia a distância dos valores da variável v para 
				# as repetições do teste em relação à mediana de todos esses 
				# valores, ao invés da média. Depois de determinada a mediana 
				# mv e o desvio mediano absoulto mad, o intervalo que o valor de
				# v para uma repetição deve estar para não ser um outlier, 
				# armazendo em non_outliers_interval como uma tupla, é definido
				# do seguinte modo, usando mv, mad e outliers_limit, usando a
				# função do objeto make_range:
				#
				# [mv - outliers_limit * mad, mv + outliers_limit * mad]
				non_outliers_interval = self.make_range(
						df[v].median(),
						outliers_limit * st.median_abs_deviation(df[v]),
				)

        # Uma vez determinado o intervalo dos valores de v que não são
				# outliers para todas as repetições do teste, a função between
				# do Pandas será usada para retornar um objeto Series do Pandas
				# no qual as linhas serão as repetições do teste e o conteúdo de
				# cada linha indicará se o valor para v da repetição associada
				# à linha está (True) ou não (False) dentro do intervalo 
				# non_outliers_interval, ou seja, se não é um outlier (True) ou
				# é um outlier (False) considerando a variável v e o teste
				# associado à linha. Como usamos a operação between em df, os 
				# índices das linhas em non_outliers_positions serão os
				# mesmos índices das linhas das repetições do teste em df.
				non_outliers_positions = df[v].between(*non_outliers_interval)

        # Adiciona no vetor de máscaras uma nova entrada para a variável
				# v. Como desejamos filtrar os outliers, então armazenados a 
				# negação do objeto Series non_outliers_positions, como indicado
				# pelo operador de negação ~. Depois da operação de negação, 
				# agora as linhas com as repetições dos testes com outliers para
				# a variável v conterão valores True, enquanto que as linhas com
				# as repetições sem outliers serão False.
				masks.append(~non_outliers_positions)

			# Usa a lista masks para criar um DataFrame. Como existe uma 
			# entrada para cada variável v em variables, pois colocamos um
			# desses objetos para cada variável em masks, e como cada entrada 
			# na lista masks é um objeto Series do Pandas cujo nome é a 
			# variável v e cujas linhas são valores booleanos indicando se a 
			# repetição do teste tem um outlier (True) ou não (False) para a 
			# variável v, com os mesmos índices dessas repetições em df, então
			# o DataFrame criado teria como linhas as variáveis em variables
			# e como colunas os índices dos testes em df, o oposto do que 
			# desejamos, que é as linhas serem os índices das repetições em
			# df e as colunas as variáveis em variables. Logo, depois de criar
			# o DataFrame, usamos a operação T do objeto DataFrame para fazer
			# exatamente isso, trocar de posição as linhas com as colunas, 
			# como ocorre em uma operação de transposição de uma matriz.
			masks_df = pd.DataFrame(masks).T

			# Retorna o DataFrame com as máscaras a serem usadas para filtrar
			# o DataFrame, removendo as repetições do teste que possuam pelo
			# menos um outlier para uma das variáveis em variables.
			return masks_df

		# Retorna a função que define as máscaras de exclusão dos outliers 
		# para cada variável v em variables e para cada repetição do teste
		# dadas no DataFrame df.
		return outliers_filter

	def Filter(self, dados, input_variables, filter_variables, outliers_limit):
		"""
		Função para fazer a filtragem dos outliers no conjunto de dados.

		Parâmetros:
			dados (DataFrame): Conjunto de dados para o qual os outliers serão 
												 filtrados.
			input_variables (list[str]): Nomes das variáveis de entrada, ou 
																	 características, usadas nos 
																	 treinamentos dos modelos, ou seja, as
																	 variáveis de configuração e as
																	 variáveis de aplicação. 
			filter_variables (list[str]): Variáveis usadas para fazer a 
																		filtragem do conjunto de dados 
																		original, ou seja, para as quais
																		iremos avaliar os testes com 
																		outliers considerando os valores 
																		destas variáveis.
			outliers_limit (float): Valor de ponto flutuante para definir o 
															fator multiplicador que será usado para 
															definir os limites inferior e superior, 
															conjuntamente com o desvio mediano
															absoluto e a mediana, do intervalo de
															com os valores que não são outliers.

    Retorna:
      DataFrame: Se nenuhum erro ocorrer, retorna um ponteiro para o 
								 objeto DataFrame do Pandas com o conjunto de dados 
								 filtrado. Se algum erro ocorrer, gera uma exceção do
								 Python.
		"""

		# Verifica se dados é um DataFrame do Pandas.
		if not isinstance(dados, pd.DataFrame):	
			raise ValueError("Conjunto de dados com os dados do teste em um formato"
											 f"inválido {type(dados)}! Deveria ser uma referência"
											 "para um objeto DataFrame do Pandas.")
		
		# Verifica se cada nome em input_variables é o nome de uma das 
		# variáveis de configuração ou da aplicação definidas pelas colunas 
		# em dados.
		if not pd.Index(input_variables).isin(dados.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
		 							   "variáveis de configuração ou da aplicação "
										 f"{','.join(input_variables)} é uma das variáveis "
										 f"{','.join(dados.columns)} do conjunto de dados dos "
										 "testes!")

		# Verifica se cada nome em filter_variables é o nome de uma das 
		# variáveis com as informações das execuções dos testes definidas
		# pelas colunas em dados.
		if not pd.Index(filter_variables).isin(dados.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de filtragem, {','.join(filter_variables)} é "
										 f"uma das variáveis {', '.join(dados.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se outliers_limit é um inteiro ou número em ponto 
		# flutuante.
		if not isinstance(outliers_limit, (int, float)):
				raise TypeError("A variável que define o limite, outliers_limit, "
										    "precisa ser do tipo int ou float.")
		
    # Define a variável do objeto dados com o conjunto de dados original 
		# e não filtrado.
		self.dados = dados

		# Define a variável do objeto input_variables com as variáveis de 
		# entrada usada no treinamento dos modelos.
		self.input_variables = input_variables

		# Define a variável do objeto filter_variables com as variáveis 
		# usadas para fazer a filtragem do conjunto de dados.
		self.filter_variables = filter_variables

		# Define a variável do objeto outliers_limit com o valor em ponto 
		# flutuante usado para definir, conjuntamente com o desvio médio 
		# absoluto e a mediana, do intervalo de com os valores que não são
		# outliers.
		self.outliers_limit = outliers_limit

    # Para descobrir a máscara com os outliers, primeiramente agrupamos
		# os dados do conjunto de dados de acordo com as variáveis de 
		# entrada em input_variables, para separar os valores das repetições
		# de cada teste definido pela combinação destas variáveis em um
		# DataFrame associado ao grupo definido pelo teste. Este DataFrame
		# terá uma linha para cada repetição, indexada pelo índice que ela
		# originalmente tinha no conjunto de dados dados, e uma coluna para
		# cada variável que não é uma das variáveis de entrada em 
		# input_variables usadas para definir cada grupo. Depois, aplicamos 
		# a função apply sobre o grupo para gerar o DataFrame final, do 
		# seguinte modo:
		#
		#- Primeiramente a função, para cada DataFrame associado a um grupo
		# descrito anteriormente, utilizando a função make_outliers_filter 
		# passando como parâmetros outliers_limit e filter_variables, 
		# substituirá este DataFrame por um outro DataFrame em que, para
		# cada repetição dada por uma das linhas do DataFrame do grupo
		# indexadas pelo mesmo índice deste DataFrame, indicará se o valor
		# para cada variável de filtragem, nomeada pelo nome dado em
		# filter_variables, se o valor da repetição para esta variável é
		# (True) ou não (False) um outlier.
		#- Depois, para cada  grupo, o seu DataFrame será concatedado ao
		# novo DataFrame gerado pela função, inicialmente vazio, sendo que
		# o índice de cada repetição do teste associado ao grupo será, antes
		# da concatenação, substituído por um índice definido pelos valores
		# das variáveis em input_variables, usadas para deifnir o grupo, e
		# o índice dessa repetição no conjunto de dados dados.
		outlier_masks = ( 
			dados.groupby(input_variables)
			.apply(
				self.make_outliers_filter(
					outliers_limit, filter_variables
				)
			)
		)

    # Como vimos, para cada teste e uma das suas repetições, o DataFrame
		# outlier_masks indicará se existe, para cada variável em 
		# filter_variavbles, se o valor de uma repetição de um teste é um 
		# outlier se o valor da sua linha e da coluna dessa variável for
		# True. Logo, para ver se uma repetição de um teste tem algum 
		# outlier, basta verificar se pelo menos uma coluna tem um valor 
		# True para esta repetição e cada variável, o que podemos fazer
		# usando a funçaõ any do Pandas aplicada a todas as colunas (axis=1)
		# de outlier_masks, que retornará um objeto Series do Pandas que
		# indicará, para cada teste e uma das suas repetições, se pelo menos 
		# o valor de uma das variáveis de filtragem é (True) ou não (False)
		# um outlier. Agora, como desejamos filtrar os outliers, e como o
		# Pandas pernite somente filtrar as colunas que desejamos manter do
		# DataFrame, precisamos usar o operador ~ para negar os valores em
		# outlier_masks, de tal modo que agora uma linha com um teste e uma
		# das suas repetições em outlier_masks será True se a repetição
		# deste teste não tiver outliers para nenhuma das variáveis de 
		# filtragem. O resultado desta negação, ou seja, com uma 
		# máscara indicando (True) ou não (False) quais testes devem ser
		# mantidos, é armazenada em non_outliers_mask.
		non_outliers_mask = ~outlier_masks.any(axis=1)

    # Ajusta os índices para casar com os índices do conjunto de dados
		# em dados, usando somente o último valor do índice com multtplos
		# níveis n objeto Series non_outliers_mask original, e depois 
		# ordena segundo esses índices.
		non_outliers_mask = (
			non_outliers_mask.reset_index(
					level=non_outliers_mask.index.names[:-1], drop=True
			)
			.sort_index()
		)		

		# Usa a máscara para escolher somente as repetições dos testes para 
		# as quais não foram encontrados outliers em todas as variáveis em 
		# filter_variables, e armazena o conjunto de dados obtido após a 
		# filtragem, na variável dados_filtrados do objeto da classe 
		# instanciado.
		self.dados_filtrados = (
				dados[non_outliers_mask]
				.reset_index(drop=True)
				.copy()
		)

    # Retorna uma referência para o conjunto de dados filtrado, sem as 
		# repetições dos testes com pelo menos un outlier para uma das
		# variáveis em filter_variables.
		return self.dados_filtrados
		
class BestHiperparams:
	"""
	Classe para fazer a busca em grade e escolher os melhores valores para
	cada hiperparâmetro a ser otimizado para um dado modelo. Será criado
	um objeto para cada modelo a ser avaliado quando formos escolher o
	melhor modelo para ser usado, para garantir que os modelos ao serem
	comparados serão configurados com os melhores valores para os
	hiperparâmetros selecionados.

	Atributos:
		X (DataFrame): X contendo os testes usados para avaliar os melhores
									 valores dos hiperparâmetros com as linhas sendo os
									 testes e as colunas e as variáveis de configuração e
									 da aplicação.
		y (Series): Valores das variáveis alvo para cada teste em X, usado
								ao avaliar os resultados dos testes da validação cruzada
								LOGO usada ao avaliar cada combinação dos
								hiperparâmetros durante a busca em grade. 
		grid_searc_model (GridSearchCV): Objeto do scikit-learn com o 
																		 resultado da busca em grade para o
																		 modelo, já com todas as buscas em
																		 grades feitas, tendo como atributos
																		 informações sobre todas as
																		 combinações de hiperparâmetros 
																		 avaliadas durante a busca em grade
																		 e as suas pontuações, assim como 
																		 atributos com a melhor combinação
																		 dos	hiperparâmetros e sua
																		 pontuação.
																		
		group (array_like[str]): Rótulos codificados, identificando os 
														 grupos que serão usados na validação LOGO
														 quando a busca em grada avaliar cada
														 combinação possível dos valores dos
														 hiperparâmetros. As variáveis do usuário,
														 que são as variáveis da aplicação passadas
														 pelo usuário que podem ou não ser
														 convertidas para as variáveis finais da
														 aplicação usadas no treinamento, são usadas
														 para definir cada grupo da validação LOGO.
		group_names	(array_like[str]): Armazena os rótulos de cada uma das 
																	 classes representando cada um dos 
																	 rótulos codificados.
		n_jobs (int): Número de trabalhos paralelos criados quando a busca
									em grade for feita. Este parâmetro é passado
									diretamente ao objeto GridSearchCV, então o
									significado é o mesmo, ou seja, um valor positivo
									define o número de trabalhos paralelos, e um valor
									negativo indica que será usado o número de unidades de
									processamento na máquina menos o valor absoluto
									negativo mais 1 (ou seja, o valor -1 define o uso de
									todas as unidades de processamento).
		verbose (bool): Habilita/desabilita as informações de verbosidade.                   
	"""

	def __init__(self, n_jobs=-1, verbose=False):
		"""
		Função de inicialização da classe BestHiperparams.

		Parâmetros:
			n_jobs (int): Número de trabalhos paralelos criados quando a busca
										em grade for feita.
			verbose (bool): Habilita/desabilita as informações de verbosidade.
		"""

		self.X = None
		self.y = None
		self.grid_search_model = None
		self.groups = None
		self.group_names = None
		self.n_jobs = n_jobs
		self.verbose = verbose
	
	def optimize(self, data, suggestion_names, application_names, user_names, 
							 predicted_name, model, hiperparams_grid, 
							 scoring=train_min_edp_config_accuracy):
		"""
		Função que faz a otimização dos hiperparâmetros para um modelo, 
		usando a busca em grade, implementada pelo objeto GridSearchCV do
		scikit-learn, sendo que a avaliação de cada possível combinação dos
		valores dos hiperparâmetros do modelo avaliados usa a validação
		cruzada utilizando a validação LOGO para dar uma pontuação para cada
		combinação e escolher a melhor, que será a primeira combinação
		avaliada com a maior pontuação segundo a validação cruzada.

		Parâmetros:
			dados (DataFrame): Conjunto de dados usado para fazer as 
												 validações cruzadas para escolher os melhores
												 valores dos hiperparâmetros avaliados para um
												 modelo.
			suggestion_names (list[str]): Nomes das variáveis de usadas para
																		definir as configurações dos 
																		recursos usadas ao executar os
																		testes. São essas configurações que
																		serão as sugeridas pelo script de
																		otimização usado pelo usuário.
			application_names (list[str]): Nomes das variáveis da aplicação
																		 usadas ao treinar os modelos, seja
																		 na busca em grade, como nas outras
																		 fases como a validação cruzada e o
																		 treinamento final dos modelos.
			user_names (list[str]): Nomes das variáveis definidas pelos 
															usuários quando usarem o script de 
															otimização para escolher a melhor sugestão
															de configuração para executar a aplicação.
															Podem ou não ser as mesmas variávis de
															application_names, tudo dependerá de como
															as variáveis da aplicação definidas pelo
															usuário são convertidas em variáveis da
															aplicação efetivamente usadas nos 
															treinamentos. São essas variáveis que
															são usadas para definir os grupos usados
															pela validação LOGO durante a validação
															cruzada ao calcular a pontuação, segundo
															a função dada em scoring definida mais
															embaixo, para cada possível combinação dos
															valores dos hiperparâmetros para depois
															escolher a primeira combinação avaliada
															com a maior pontuação.
			predicted_name (str): Nome da variável alvo, que será a predita
														nos treinamentos e usada para auxiliar a
														descoberta da melhor sugestão de 
														configuração.
			model (BaseEstimator): Referência para o objeto definindo o
														 modelo (seguindo o formato da biblioteca 
														 scikit-learn) para o qual iremos escolher
														 a melhor combinação dos hiperparâmetros que
														 iremos avaliar.
			hiperparams_grid (dict): Dicionário em que cada chave é um dos
															 hiperparâmetros que desejamos otimizar
															 do modelo dado pelo objeto referenciado
															 por model. O valor da chave é a lista dos
															 possíveis valores que iremos avaliar para
															 o hiperparâmetro definido pela chave.
			scoring (func): Função que irá calcular as pontuações durante o
											processo de busca em grade, no formato usado pelo
											scikit-learn. Por default, é a função que calcula
											a acurácia que descrevemos anteriormente neste
											arquivo, a train_min_edp_config_accuracy.	A busca
											em grade supõe que é uma pontuação que desejamos
											maximizar, ou seja, a maior pontuação de todas 
											será a melhor.

    Retorna:
      tuple: Se não ocorrerem erros, retorna uma tupla com duas 
						 referências, a primeira para um dicionário em que as chaves
						 são os hiperparâmetros otimizados e, para cada chave, o seu
						 valor é o melhor valor escolhido para este hiperparâmetro,
						 e a segunda referência é para um número em ponto flutuante
						 com a pontuação, segundo a fundão dada pelo parâmetro
						 scoring, para os testes ao testar o modelo usando esses
						 hiperparâmetros e a validação cruzada usando a validação
						 LOGO. Se algum erro ocorrer, gera uma exceção do Python.
		"""

		# Verifica se data é um DataFrame do Pandas.
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Conjunto de dados com os dados do teste em um formato"
											 f"inválido {type(data)}! Deveria ser uma referência"
											 "para um objeto DataFrame do Pandas.")
		
		# Verifica se cada nome em suggestion_names é o nome de uma das 
		# variáveis de configuração definidas pelas colunas em data.
		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
		 							   f"variáveis de configuração {','.join(suggestion_names)} "
										 f"é uma das variáveis {','.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em application_names é o nome de uma das 
		# variáveis da aplicação definidas pelas colunas em data.
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(application_names)} é "
										 f"uma das variáveis {', '.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em user_names é o nome de uma das variáveis
		# da aplicação definidas pelas colunas em data, e usadas para
		# definir os grupos da validação LOGO.
		if not pd.Index(user_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(user_names)} é uma "
										 f"das variáveis {','.join(data.columns)} do conjunto de "
										 "dados dos testes!")
		
		# Verifica o nome em predicted_name é o nome da variável que será a 
		# usada como variável alvo dos modelos, que também será a variável
		# usada ao escolher a melhor sugestão de configuração.
		if not pd.Index([predicted_name]).isin(data.columns).all():
			raise KeyError("Nome inválido da variável alvo a ser predita! "
									   f"O Nome {predicted_name} não é o nome de uma das "
										 f"variáveis {','.join(data.columns)} do conjunto de dados"
										 "dos testes!")

    # Define o campo X do objeto como as colunas cujos nomes estão em
		# suggestion_names e application_names (pois estas são as variáveis
		# usadas como características ao treinar os modelos).
		self.X = data[suggestion_names+application_names]

		# Define o campo y do objeto como a coluna da variável alvo, cujo
		# nome é dado em predicted_name
		self.y = data[predicted_name]

    # Cria os grupos usados na validação LOGO, usando as variáveis de
		# aplicação dadas em user_names. Para criar os grupos, primeiramente 
		# é definida uma instância do objeto LabelEncoder do scikit-learn.
		# TODO: Eu tinha colocado este código somente para termos os grupos,
		# mas eles são recriados internamente pelo método fit do objeto da
		# classe GridSearchCV que criaremos a seguir. Não sei se devo
		# remover esta parte.
		lab_encoder = skpp.LabelEncoder()

		# Cria efetivamente os grupos, usando a função fit_transform do
		# objeto lab_encoder criado anteriormente, e salva uma referência
		# para um objeto do tipo vetor contendo cada um dos grupos criados.
		# Será este o objeto com os grupos definidos, que será 
		# posteriormente pasado à função que efeticamente fará a busca em
		# grade. Os grupos serão armazenados no campo groups do objeto.
		self.groups = lab_encoder.fit_transform(
    		list(map(str, data[user_names].values))
		)

		# Armazena os nomes internos dos grupos (chamados de classes), no
		# campo groups_names do objeto.
		self.groups_names = lab_encoder.classes_

    # Imprime uma mensagem para separar a verbosidade da função fit do
		# objeto da classe GridSearchCV.
		if self.verbose:
			print("*** Início da verbosidade da função GridSearchCV **** \n")

    # Cria o objeto de grid para otimizar os hiperparâmetros. A criação
		# somente cria a referência para o objeto e o inicializa. A busca em
		# grade é feita pela chamada de uma função do objeto criado, como
		# veremos a seguir.  
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

    # Imprime uma mensagem para separar a verbosidade da função fit do
		# objeto da classe GridSearchCV.
		if self.verbose:
			print("*** Fim da verbosidade da função GridSearchCV **** \n")
			
		# Depois de criado e inicializado o objeto, a função fit deste 
		# objeto será chamada para fazer a busca em grade, e retornar a
		# referência para o objeto com todas as informações sobre a busca
		# em grade feita pelo fit.
		self.grid_search_model = grid_search_model.fit(self.X, self.y, 
													  											 groups=self.groups)
		
		# Retorna uma tupla com dois objetos com informações sobre o 
		# resultado da otimização, ou seja, os melhores valores dos 
		# hiperparâmetros, dados e, grid_search_model.best_params_,  que é 
		# um dicionário em que cada chave é o hiperparâmetro e o valor 
		# associado à chave é o melhor valor para esse hiperparâmetro, e
		# grid_search_model.best_score_que armazena a pontuação, segundo 
		# calculada pela função dada no parâmetro scoring, para os valores
		# dis hiperparâmetros escolhidos como os melhores.
		return (self.grid_search_model.best_params_, 
						self.grid_search_model.best_score_)
	
	def get_hyperparams_scores(self):
		"""
		Função para retornar as informações da busca em grade feita ao 
		otimizar os hiperparâmetros do modelo definido após chamar a função
		optimize descrita anteriormente.

		Parâmetros:
				Não tem parâmetros.	
		"""

		# Verifica se já fizemos a busca em grade do modelo.
		if self.grid_search_model is None:
			raise ValueError("Os hiperparâmetros do modelo ainda não foram "
										   "otimizados!")
		
		# Converte o dicionário com todas as informações da busca em grade,
		# dado pelo campo cv_results_ do objeto GridSearchCV para o qual já
		# fizemos a busca em grade, referenciado pelo campo 
		# grid_search_model do objeto BestHiperparams, para um DataFrame do
		# Pandas.
		hiperparams_df = pd.DataFrame(self.grid_search_model.cv_results_)

		# Retorna o DataFrame obtido da conversão do dicionário.
		return hiperparams_df	
class DiscoverBestModel:
	"""
  Descobre o melhor modelo, dentre todos os modelos passados como 
	parâmetro ao chamar a função best_model para um objeto da classe. Para
	cada modelo, diversas pontuações serão calculadas, sendo que o melhor
	modelo será o que tiver a maior pontuação para a pontuação calculada
	pela primeira função definida pelo dicionário do parâmetro 
	scores_functions. Se dois os mais modleos tiverem a maior pontuação,
	a segunda será a usada e o seu máximo será considerado e assim por
	diante, para as outras pontuações, sempre com uma pontuação sendo
	usada como critério de desempate. Para calcular as pontuações para
	cada modelo, a validação cruzada, implementada usando um objeto da
	classe cross_validate do scikit-learn, será usado para cada modelo, 
	utilizando a validação LOGO. Como a validação LOGO é usada, então as
	pontuações serão calculadas para cada grupo usado como conjunto de
	teste, sendo que o modelo usado para as predições será o treinado com
	os dados dos outros grupos, e esse processo será repetido para cada
	grupo definido pela validação LOGO. Após de calculadas as pontuações
	para cada grupo, as pomtuações finais para o modelo serão as médias
	das pontuações de todos os testes dos grupos gerados pela validação
	LOGO. Por default, serão usadas as pontuações de acurária, calculada e
	descrita na função train_min_edp_config_accuracy, sendo a única função
	de desempate o a pontuação retornada pela função
	neg_train_min_edp_config_diff, que usa o negativo da pontuação de
	diferença calculada e descrita na função train_min_edp_config_diff,
	porque a pontuação de diferença é uma pontuação em que o melhor valor
	é o menor de todos e não o maior, como é necessário.

	Parâmetros:
		X (DataFrame): X contendo os testes usados para escolher o melhor
									 modelo usando a validação cruzada, com as variáveis
									 de configuração e da aplicação sendo as colunas do
									 Dataframe e os testes sendo as linhas do DataFrame.
		y (Series): Valores das variáveis alvo para cada teste em X, usado
								ao avaliar os resultados dos testes da validação cruzada
								com a validação LOGO usada ao avaliar cada modelo para
								depois escolher o melhor modelo. 
		cv_results (dict): Objeto do scikit-learn com os resultados da busca
											 em grade para cada modelo avaliado. A chave do
											 dicionário é o nome do modelo e o conteúdo é o
											 dicionário com as informações sobre a validação
											 cruzada para o modelo. Dentre as informações,
											 para cada grupo do LOGO, estão as pontuações,
											 os índices dos testes do conjunto de dados X 
											 usados no treinamento e no teste para cada grupo,
											 e as referências para os modelos treinados para
											 cada grupo que foram usados para fazer as
											 predições destes grupos.
		results_df (DataFrame): Dataframe construído a partir do dicionário
														cv_results, sem incluir as informações dos
														índices e dos modelos, para permitir um
														acesso faciliatado aos dados da validação
														cruzada para cada modelo. Tem uma coluna
														adicional 'Model' com o nome de cada modelo,
														pois os resultados das validações de todos
														os modelos são convertidos para um DataFrame
														separadamente e depois concatenados em um
														DataFrame final com as informações da
														validação cruzada para todos os modelos
														avaliados.
		group (array_like[str]): Rótulos codificados, identificando os grupos
														 que serão usados na validação LOGO quando a
														 busca em grada avaliar cada combinação 
														 possível dos valores dos hiperparâmetros. 
														 As variáveis do usuário, que são as
														 variáveis da aplicação passadas pelo
														 usuário que podem ou não ser convertidas
														 para as variáveis finais da aplicação
														 usadas no treinamento, são usadas para
														 definir cada grupo da validação LOGO.
		group_names	(array_like[str]): Armazena os rótulos de cada uma das 
														 			 classes representando cada um dos 
													 			 	 rótulos codificados.
		mean_scores_models_df (DataFrame): DataFrame com as pontuações 
																			 médias das pontuações definidas
																			 em pelo dicionário do parâmetro 
																			 scores_functions da função 
																			 best_model usada para avaluar os
																			 modelos. Os índices do DataFrame
																			 são os nomes dos modelos e as 
																			 colunas são as pontuações. Os
																			 modelos são ordenados no
																			 DataFrame, de cima para baixo, do
																			 melhor modelo segundo as
																			 pontuações para o pior modelo.
		best_model_name (str): Nome do melhor modelo segundo as pontuações
													 médias calculadas pelas validações cruzadas.
		best_model_score (dict): Dicionário com os valores das pontuações
														 do melhor modelo. Existe uma chave para
														 cada pontuação, sendo o valor associado à
														 chave o valor da pontuação para o melhor
														 modelo.
		n_jobs (int): Número de trabalhos paralelos criados quando a busca
									em grade for feita. Este parâmetro é passado
									diretamente ao objeto GridSearchCV, então o
									significado é o mesmo, ou seja, um valor positivo
									define o número de trabalhos paralelos, e um valor
									negativo indica que será usado o número de unidades de
									processamento na máquina menos o valor absoluto
									negativo mais 1 (ou seja, o valor -1 define o uso de
									todas as unidades de processamento).
		verbose (bool): Habilita/desabilita as informações de verbosidade.                   
	"""

	def __init__(self, n_jobs=-1, verbose=False):
		"""
		Função de inicialização da classe DiscoverBestModel.

		Parâmetros:
			n_jobs (int): Número de trabalhos paralelos criados quando a busca
										em grade for feita.
			verbose (bool): Habilita/desabilita as informações de verbosidade.
		"""

		self.X = None
		self.y = None
		self.cv_results = None
		self.results_df	= None
		self.groups = None
		self.group_names = None
		self.mean_scores_models_df = None
		self.best_model_name = None
		self.best_model_score = None
		self.n_jobs = n_jobs
		self.verbose = verbose
		
	def best_model(self, data, suggestion_names, application_names, 
			 					 user_names, predicted_name, models, 
								 scores_functions=None):
		"""
		Função para avaliar todos os modelos passados em models, que é um
		dicionário em que cada chave é o nome do modelo e o valor associado
		à esta chave um objeto do modelo já inicializado com os seus
		melhores hiperparâmetros definidos ao criar um objeto da classe
		BestHiperparams para cada modelo e chamar a função optimize para
		obter os melhores valores para os hiperparâmetros otimizados. O 
		conjunto de dados usado na validação cruzada de cada modelo é
		passado em data, as variáveis da sugestão de configuração em
		suggestion_names, as variáveis da aplicação definidas pelos usuários
		ou derivadas a partir das variáveis da aplicação definidas pelos
		usuários em application_names, as variáveis em user_names usadas
		para definir os grupos da validação LOGO usada, geralmente
		diretamente definidas pelos usuários ou obtidas a partir de uma
		simples conversão do que foi definido pelo usuário, a variável alvo
		definida em predicted_name que será a variável para a qual os
		modelos treinados farão predições, e o dicionário scores_functions
		com as funções de pontuação usadas, a serem maximizadas, sendo a
		chave o nome da função e o valor uma referência para a função que
		calcula a pontuação. Se o parâmetro não for passado, será usado o
		dicionário com duas funções default, a chave 'accuracy' que calcula
		a acurácia, como definimos nos artigos, usando a função
		train_min_edp_config_accuracy, e a chave 'difference' calculando o
		negativo da função de diferença que também definimos nos artigos,
		usando a função neg_train_min_edp_config_diff.

		Parâmetros:
			dados (DataFrame): Conjunto de dados usado para fazer as 
												 validações cruzadas para escolher os melhores
												 valores dos hiperparâmetros avaliados para um
												 modelo.
			suggestion_names (list[str]): Nomes das variáveis de usadas para
																		definir as configurações dos
																		recursos usadas ao executar os
																		testes. São essas configurações que
																		serão as sugeridas pelo script de
																		otimização usado pelo usuário.
			application_names (list[str]): Nomes das variáveis da aplicação
																		 usadas ao treinar os modelos, seja
																		 na busca em grade, como nas outras
																		 fases como a validação cruzada e o
																		 treinamento final dos modelos.
			user_names (list[str]): Nomes das variáveis definidas pelos 
															usuários quando usarem o script de 
															otimização para escolher a melhor sugestão
															de configuração para executar a aplicação.
															Podem ou não ser as mesmas variávis de
															application_names, tudo dependerá de como
															as variáveis da aplicação definidas pelo
															usuário são convertidas em variáveis da
															aplicação efetivamente usadas nos 
															treinamentos. São essas variáveis que
															são usadas para definir os grupos usados
															pela validação LOGO durante a validação
															cruzada ao calcular a pontuação, segundo
															a função dada em scoring definida mais
															embaixo, para cada possível combinação dos
															valores dos hiperparâmetros para depois
															escolher a primeira combinação avaliada
															com a maior pontuação.
			predicted_name (str): Nome da variável alvo, que será a predita
														nos treinamentos e usada para auxiliar a
														descoberta da melhor sugestão de 
														configuração.
			models (dict): Dicionário em que a chave é o nome do modelo e o
										 valor associado à chave é uma referência para um
										 objeto do modelo criado e inicializado, mas ainda
										 não treinado, com os melhores hiperparâmetros
										 definidos para o modelo quando foi criado um objeto
										 da classe BestHiperparams para esse modelo e a
										 função optimize do objeto foi chamada para
										 descobrir os melhores valores dos hiperparâmetros
										 otimizados para esse modelo, mais, se existirem, os
										 hiperparâmetros fixos e os seus valores.
			scores_functions (dict): Dicionário com as funções com as
															 pontuações usadas para pontuar todas as
															 avaliações feitas durante a validação
															 cruzada dos modelos usando a validação
															 LOGO e a posterior escolha do melhor
															 modelo devido a usarmos a médias das
															 pontuações de cada grupo da validação
															 LOGO. Se não for passado o dicionário com
															 as funções, serão usadas as pontuações de
															 acurácia, como a pontuação de escolha do
															 melhor modelo e o negativo da pontuação
															 de diferença para escolher o modelo em
															 caso de empate na maior pontuação de
															 acurácia.
    
		Retorna:
      tuple: Se não ocorrerem erros, retorna uma tupla com duas 
						 referências, a primeira o nome do melhor modelo segundo as
						 pontuções definidas em scores_functions, e a segunda 
						 referência com um dicionário com os valores das pontuações
						 do melhor modelo. Neste dicionário, existirá uma chave para
						 cada pontuação, sendo o valor associado à chave o valor da
						 pontuação para o melhor modelo. Se algum erro ocorrer, gera
						 uma exceção do Python.
		"""

		# Se não forem passadas funções de pontuação, então vamos usar as
		# funções que definimos no projeto a que calcula a acurácia
		# train_min_edp_config_accuracy e a que retorna o valor negativo de
		# pontuação de diferença, calculada pela função
		# neg_train_min_edp_config_diff.
		if scores_functions is None:
			scores_functions = {
				'accuracy': train_min_edp_config_accuracy, 
				'difference': neg_train_min_edp_config_diff
			}

		# Verifica se data é um DataFrame do Pandas.
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Conjunto de dados com os dados do teste em um formato"
											 f"inválido {type(data)}! Deveria ser uma referência"
											 "para um objeto DataFrame do Pandas.")
		
		# Verifica se cada nome em suggestion_names é o nome de uma das 
		# variáveis de configuração definidas pelas colunas em data.
		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
		 							   f"variáveis de configuração {','.join(suggestion_names)} "
										 f"é uma das variáveis {','.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em application_names é o nome de uma das 
		# variáveis da aplicação definidas pelas colunas em data.
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(application_names)} é "
										 f"uma das variáveis {', '.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em user_names é o nome de uma das variáveis
		# da aplicação definidas pelas colunas em data, e usadas para
		# definir os grupos da validação LOGO.
		if not pd.Index(user_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(user_names)} é uma "
										 f"das variáveis {','.join(data.columns)} do conjunto de "
										 "dados dos testes!")
		
		# Verifica o nome em predicted_name é o nome da variável que será a 
		# usada como variável alvo dos modelos, que também será a variável
		# usada ao escolher a melhor sugestão de configuração.
		if not pd.Index([predicted_name]).isin(data.columns).all():
			raise KeyError("Nome inválido da variável alvo a ser predita! "
									   f"O Nome {predicted_name} não é o nome de uma das "
										 f"variáveis {','.join(data.columns)} do conjunto de dados"
										 "dos testes!")

		# Define o campo X do objeto como as colunas cujos nomes estão em
		# suggestion_names e application_names (pois estas são as variáveis
		# usadas como características ao treinar os modelos).
		self.X = data[suggestion_names+application_names]

		# Define o campo y do objeto como a coluna da variável alvo, cujo
		# nome é dado em predicted_name
		self.y = data[predicted_name]

		# Cria os grupos usados na validação LOGO, usando as variáveis de
		# aplicação dadas em user_names. Para criar os grupos, primeiramente 
		# é definida uma instância do objeto LabelEncoder do scikit-learn.
		lab_encoder = skpp.LabelEncoder()

		# Cria efetivamente os grupos, usando a função fit_transform do
		# objeto lab_encoder criado anteriormente, e salva uma referência
		# para um objeto do tipo vetor contendo cada um dos grupos criados.
		# Será este o objeto com os grupos definidos, que será 
		# posteriormente passado à função cross_validate que fará a
		# validação cruzada para cada modelo.
		self.groups = lab_encoder.fit_transform(
    		list(map(str, data[user_names].values))
		)

		# Armazena os nomes internos dos grupos (chamados de classes), no
		# campo groups_names do objeto.
		self.groups_names = lab_encoder.classes_

		# Cria um dataframe vazio, que irá armazenar os resultados das
    # valições cruzadas de todos os modelos que serão avaliados.
		self.results_df = pd.DataFrame()

    # Dicionário que armazenará o campo cv_results_ do objeto da classe
		# cross_validate do scikit-learn, com os resultados da validação
		# cruzada para cada modelo avaliado. A chave do dicionário será o
		# nome do modelo, definido pela chava no dicinário models que
		# contém um objeto inicializado e não treinado deste modelo. Cada
		# chave terá um dicionário com duas informações, uma referência para
		# o dicionário retornado pela validação cruzada do modelo com todas
		# as informações desta validação cruzada, associada à chave
		# 'cv_results' e o DataFrame gerado a partir destas informações, 
		# igonrando os índices e estimadores de cada grupo avaliado, com
		# uma coluna Model preenchida com o nome do modelo, associada à
		# chave 'results_dataframe'. 
		self.cv_results = {}

		# Avalia os todos os modelos dados no dicionário models passado como
		# parâmetro para a função.
		for name, model in models.items():
			# Imprime uma mensagem para separar a verbosidade da função fit do
			# objeto da classe cross_validate.
			if self.verbose:
				print("*** Início da verbosidade da função cross_validate **** \n")

			# Faz a validação cruzada para o modelo cujo nome é dado em name,
			# e cujo objeto inicializado foi armazenado em model. A validação
			# cruzada irá usar a validação LOGO, sendo os grupos usados os que
			# foram definidos anteriormente usando um objeto da classe
			# LabelEncoder do scikit-learn e armazenados no campo groups do
			# objeto. 
			cv_results_model = skms.cross_validate(
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
				verbose=int(self.verbose)
			)

			# Imprime uma mensagem para separar a verbosidade da função fit do
			# objeto da classe cross_validate.
			if self.verbose:
				print("\n*** Fim da verbosidade da função cross_validate ****")

			# Converte o resultado da validação cruzada, ignorando os campos
			# do estimador e os índices para os testes de cada um dos grupos
			# usados pela validação LOGO. Para maior clareza, removemos todos
			# os "test_" nos nomes associados às pontuações de cada grupo.
			# TODO: Não usamos as informações excluídas (indices e estimator)
			# no momento. Talvez seja melhor não retornar estas informações,
			# defindo, na função fit, 'return_indices' como False e 
			# return_estimator como False.
			cv_results_df = pd.DataFrame(
					{
							k.replace("test_", ""): cv_results_model[k]
							for k in cv_results_model
							if k not in ["indices", "estimator"]
					}
			)

			# Adiciona uma coluna com o nome do modelo, no DataFrame
			# cv_results_df, para podermos diferenciar os dados das validações
			# cruzadas de cada modelo no DataFrame final com os dados de todas
			# as validações cruzadas feitas para determinar o melhor modelo.
			cv_results_df['Model'] = name

			# Salna no campo cv_results do objeto, na chave associada ao
			# modelo definida pelo seu nome em name, o dicionário com o
			# resultado da validação cruzada na chave 'cv_results' de um
			# dicionário, e o DataFrame obtido a partir dos dados do
			# dicionário, excluindo as chaves 'indices' e 'estimator', e com
			# uma coluna 'Model' definida com o nome do modelo, na chave
			# 'results_dataframe' deste dicionário. 
			# TODO: Se não usarmos os índices e os estimadores, como neste
			# caso o DataFrame e o dicionário teriam as mesmas informações,
			# podemos substituir o dicionário por uma referência direta ao
			# DataFrame cv_results_df. 
			self.cv_results[name] = { 'cv_results': cv_results_model, 
														    'results_dataframe': cv_results_df}

			# Imprime, se a verbosididade estiver habilitada, o DataFrame com 
			# as informações sobre as pontuações e tempos de execução do
			# modelo resultante da validação cruzada e as estatístcas deste
			# DataFrame usando a função describe.
			if self.verbose:
				print(f"➡️  Model {name} table:")
				print("\n", cv_results_df.to_markdown(tablefmt="grid", 
																					    floatfmt=".2f"), 
							"\n", sep="")
				print(f"➡️  Model {name} statistics:")
				print("\n", cv_results_df.describe().to_markdown(tablefmt="grid", 
																										     floatfmt=".2f"), 
							"\n", sep="")

			# Anexa o DataFarme gerado para o modelo atualmente avaliado e
			# identificado pelo nome em name no DataFrame results_df 
			# armazenado no campo do objeto, com as informações das validações
			# cruzadas dos outros modelos já avaliados.
			self.results_df = pd.concat([self.results_df, cv_results_df])

		# Como o DataFrame armazenado no campo results_df do objeto é uma
		# concatenação dos DataFrames com as informações das validações
		# cruzadas de cada um dos modelos avaliados, precisamos resetar os
		# índices deste DataFrame para que sejam consecutivos.
		self.results_df = self.results_df.reset_index(drop=True)		
		
		# Descobre os nomes das colunas do DataFrame armazenado no campo
		# results_df do objeto que tem as pontuações anteriormente
		# calculadas durante o processamento das validações cruzadas feitas
		# para os modelos, e armazena os nomes na lista 
		# scores_functions_names.
		scores_functions_names = list(scores_functions.keys())

		# Cria um DataFrame com a média de todas as pontuações avaliadas e
		# definidas em scores_functions_names, agrupando as informações do
		# DataFrame armazenado no campo results_df do objeto com todas as
		# pontuações de todas as validações cruzadas dos modelos feitas
		# anteriormente. O DataFrame foi, para acesso posterior, armazenado
		# no campo mean_scores_models_df do objeto.
		self.mean_scores_models_df = (
				self.results_df.groupby(by=['Model'])[scores_functions_names].mean()
		)
		
		# Para descobrir os melhores modelos, usamos as funções de pontuação
		# para ordenar, em ordem decrescente, porque desejamos as maiores
		# pontuações, as linhas do DataFrame mean_scores_models_df 
		# armazenado no objeto de acordo com as pontuaçẽos calculadas, 
		# usando a ordem definida por como as chaves foram armazenadas, no
		# dicionário scores_functions com as pontuações usadas (que é a
		# mesma dos elementos da lista scores_functions_names), ao ordenar
		# pelas pontuações.
		#
		# TODO: As versões mais recentes do Python mantpem a ordem das
		# chaves dos dicionários, mas as antigas não mantém, então para
		# versẽos antigas do Python a ordem pode ser aleatória. Vamos manter
		# a ordenalão deste modo, ou definir um modo de escolher a ordem das
		# funções de pontuação?
		self.mean_scores_models_df = self.mean_scores_models_df.sort_values(
				by=scores_functions_names, 
				ascending=False
		)

		# Depois de ordenar o DataFrame armazenado em mean_scores_models_df,
		# a primeira linha conterá as informações do modelo com as maiores
		# pontuações, de acordo com a ordem definida na lista
		# scores_functions_names. Armazenanos o nome desse modelo, que será
		# o índice desta linha, no campo best_model_name do onjeto, as 
		# pontuações, que são as colunas desta primeira linha, no campo
		# best_model_scores do objeto, que será um diciponaŕio em que,
		# devido a conversão feita, terá uma chava para cada pontuação, com
		# a chave sendo o nome da pontuação definido originalmente como uma
		# das chaves do dicionário scores_functions e o valor associado à
		# chave o valor desta pontuação para o melhor modelo.
		self.best_model_name = self.mean_scores_models_df.index[0]
		self.best_model_scores = self.mean_scores_models_df.iloc[0].to_dict()

		# Retornauma tupla com todas as informações relavantes referentes a
		# descoberta do melhor modelo, ou seja, o nome do modelo, o
		# dicionário com as pontuações paara este modelo, o DataFrame com
		# os resultados de todas as validações cruzadas, e o DataFrame
		# com as pontuações médias, indexado pelos nomes dos modelos e
		# ordenado e acordo com as pontuações.
		return (self.best_model_name, self.best_model_scores, 
					  self.results_df, self.mean_scores_models_df)

	def get_cv_results_model(self, model_name):
		"""
		Função para retornar todas as informações, ou seja, pontuações e
		tempos de execução, referentes à execução da validação cruzada para
		o modelo cujo nome é passado como parâmetro.

		Parâmetros:
			model_name (str): Nome do modelo que foi avaliado, pela chamada à
			função best_model, para o qual desejamos obter as informações
			referentes à validação cruzada feita, ou seja, as pontuações e os
			tempos de treinamento e de teste de cada grupo da validação LOGO.

		Retorna:
			dict: Dicionário com as informações sobre a validação cruzada
						feita para o modelo. É o mesmo dicionário retornado pela
						função cross_validade do scikit-learn quando foi chamada na
						função best_model para o modelo cujo nome foi passado em
						model_name. Se algum erro ocorrer, exceções serão geradas, 
						como não existirem dados porque a função best_model não foi
						chamada ou o nome do modelo não existe porque ele não foi
						avaliado.
		"""

		# Verifica se as validações cruzadas já foram feitas, ou seja, se 
		# já foi executada a função best_model para determinar qual é o
		# melhor modelo.
		if self.cv_results is None:
			raise ValueError("O melhor modelo ainda não foi descoberto e as " 
										   "validações cruzadas de todos os modelos ainda "
											 "não foram feitas!")

		# Verfica se o nome do modelo, passado como parâmetro em model_name,
		# é válido, ou seja, é uma das chaves do dicionário best_model
		# armazenado no campo do objeto, pois usamos o nome do modelo,
		# quando executamos a função best_model, como a chave do dicion para
		# acessar as suas informações.
		if not model_name in self.cv_results.keys():
			raise KeyError(f"Nome {model_name} inválido para um dos modelos " 
									   "treinados! Deveria ser um dos modelos da lista "
										 f"{', '.join(self.cv_results.keys())}.")

		# Retorna o dicionário com as informações sobre a validação cruzada
		# do modelo cujo nome foi passsado como parâmetro, criado quando a
		# função best_model chamou a função do scikit-learn cross_validate
		# para o modelo cujo nome foi passado em model_name.	
		return self.cv_results[model_name]
			      
class SuggestionsPredictor:
	"""
	Classe que implementa as duas fases descritas no artigo, a construção
	do preditor pelo script de treinamento e o uso do preditor pelo script
	de otimização. A função fit treina o preditor usando os dados de 
	entrada, usando como características as variáveis de configuração e as
	variáveis da aplicação. As variáveis do usuário são somente para
	permitir obter o oráculo, se for necessário no futuro usar o oráculo,
	pois atualmente ele somente é usado para calcular as pontuações de
	acurácia e de diferença que definimos nos artigos, que são somente
	usadas quando otimizamos os hiperparâmetros de cada modelo a ser
	avaliado e posteriormente, após obter os melhores valores para os
	hiperparâmetros, para escolher o melhor modelo que será o usado para
	ser treinado para obter o preditor. Dois modelos são treinados, um
	que irá auxiliar a escolha da melhor sugestão de configuração para
	executar a aplicação do usuário, e outro que será usado para obter o
	tempo de execução estimado para esta condiguração, que será usado para
	escolher, junto da sugestão de configuração, a melhor partição do
	supercomputador (por enquanto, o Santos Dumont 1 e 2) ao configurar o
	script de submissão que pode ser posteriormente salvo e/ou 
	automaticamente submetido na fila da partição escolhida. As variáveis
	alvo usadas em cada caso são configuradas no arquivo de configuração
	da aplicação e passadas no dicionário estimated_parameters, sendo que 
	a chave 'suggestion' está associada ao nome da variável alvo do modelo
	a ser treinado para auxiliar na escolha da melhor sugestão de
	configuração (nos nossos testes preliminares, configuramos a varíavel 
	alvo para o 'EDP'), e a chave 'time' está associada ao nome da
	variável alvo do outro modelo a ser treinado para predizer o tempo
	estimado de execução da sugestão de configuração escolhida (nos nossos
	testes preliminares, configuramos para 'ElapsedRaw').

	Atributos:
		X (DataFrame): X contendo os testes usados para treinar os modelos
									 usados para auxiliar a escoha da melhor sugestão de
									 configuração e as colunas com as variáveis de
									 configuração e da aplicação.
		y (Series): Valores da variável alvo para cada teste em X, usados ao
								treinar o modelo usado para auxiliar na escolha da
								melhor sugestão de configuração. O nome da série será o
								nome da variável alvo, que nos nossos testes 
								preliminares será a EDP.
		y_time (Series): Valores da variável alvo para cada teste em X, 
										 usados ao treinar o modelo usado para estimar o
										 tempo de execução para a sugestão de configuração
										 escoçhida e a auxiliar na escolha ds partição para
										 executar a aplicação com esta sugestão. O nome da
										 série será o ome da variável alvo, que nos nossos
										 testes preliminares será a ElapsedRaw.
		suggestion_names (list[str]): Nomes das variáveis de usadas para
																	definir as configurações dos
																	recursos usadas ao executar os
																	testes. São essas configurações que
																	serão as sugeridas pelo script de
																	otimização usado pelo usuário.
		application_names (list[str]): Nomes das variáveis da aplicação
															 	   usadas ao treinar o modelo para 
																	 auxiliar a escolha da melhor sugestão
																	 de configuração e o modelo para
																	 estimar o tempo de execução da 
																	 sugestão de configuração escolhida.
		user_names (list[str]): Nomes das variáveis definidas pelos usuários
														quando usarem o script de otimização para
														escolher a melhor sugestão de configuração
														para executar a aplicação. Podem ou não ser
														as mesmas variávis de application_names,
														tudo dependerá de como as variáveis da
														aplicação definidas pelo usuário são
														convertidas em variáveis da aplicação
														efetivamente usadas nos treinamentos. Esas
														variáveis serão usadas somente para calcular
														o DataFrame com os oráculos para cada
														combinacão dos parâmetro da aplicação usados
														ao treinar o modelo para auxiliar a escolha
														da melhor sugestão de configiração e o
														modelo para estimar o tempo de execução
														desta melhor sugestão.
		predicted_name (str): Nome da variável alvo, que será a predita no 
													modelo treinado para auxiliar a escolha da
													melhor sugestão de configuração.
		predicted_time_name (str): Nome da variável alvo, que será a predita
															 no modelo treinado para fazer a
															 estimativa do tempo de execução da melhor
															 sugestão de configuração.
		dataset (DataFrame): DataFrame com todas as colunas usadas, ou
												 seja, as colunas em suggestion_names,
												 application_names, user_names, predicted_name e
												 predicted_time_name, que é internamente usadas
												 para calcular o oráculo e para determinar todas
												 as sugestões de configuração consideradas para
												 escolher a melhor sugestão quando os usuários
												 não definem valores customizados para cada
												 possível variável da configuração, o que
												 permite definir sugestões de configuração
												 customizadas usando todas as possíveis
												 combinações desses valores. 
		model (BaseEstimator): Referência para a classe do objeto do modelo 
													 treinado que será usado para auxiliar na
													 escolha da melhor sugestão de configuração.										 
		model_time (BaseEstimator): Referência para o objeto do modelo
																treinado que será usado para estimar o
																tempo de execução da melhor sugestão
																de configuração.										 
		suggestion_param (dict): Diconário cujas chaves são as variáveis e,
														suggestion_names sendo o objeto associado a
														cada chave uma lista com os possíveis 
														valores para a variável de sugestão que tem
														como nome essa chave.
		applcartion_param (dict): Diconário cujas chaves são as variáveis e,
															application_names sendo o objeto associado
															a cada chave uma lista com os possíveis 
															valores para a variável de aplicação que
															tem como nome essa chave.
		user_param (dict): Dicionário cujas chaves são as variáveis e,
											 user_names sendo o objeto associado a cada chave
											 uma lista com os possíveis valores para a
											 variável do usuário que tem como nome essa chave.
	"""
	def __init__(self):
		"""
    Função de inicialização da classe SuggestionsPredictor.

		Parâmetros:
      Não tem parâmetros.
		"""
		self.X = None
		self.y = None
		self.y_time = None
		self.suggestion_names = None
		self.application_names = None
		self.user_names = None
		self.predicted_name = None
		self.predicted_time_name = None
		self.dataset = None
		self.model = None
		self.model_time = None
		self.suggestion_params = None
		self.application_params = None
		self.user_params = None
	    	
	def fit(self, data, suggestion_names, application_names, user_names, 
				  estimated_parameters, model, model_params, verbose=False):
		"""
		Função para fazer a parte final da construção do preditor, que é o
		treinamento do melhor modelo com os seus melhores hiperparâmetros, 
		utilizando todo o conjunto de dados para gerar os dois modelos que
		compõem o preditor. O primeiro destes modelos, o mais importante e
		descrito no artigo, será usado para predizer a variável alvo que
		usaremos para escolher a melhor sugestão de configuração, ao
		escolher ao configuração associado ao menor valor predito para a
		variável alvo. Nos nossos testes preliminares, as configurações dos
		treinamentos dos aplicativos configuram esta variável para a 'EDP'.
		O segundo modelo será usado para estimar o tempo de execução para a
		sugestão de configuração escolhida com o auxílio do primeiro modelo.
		Nos nossos testes preliminares, as configurações dos treinamentos
		dos aplicativos configuram esta variável para a 'ElapsedRaw'. 
		
		Parâmetros:
			dados (DataFrame): Conjunto de dados usado usado para treinar os
												 dois modelos do preditor, o que prediz a 
												 variável alvo que permitirá escolher a melhor
												 sugestão de configuração ao escolher o menor
												 valor predito para um conjunto de sugestões
												 de configuração e usar como a sugestão de
												 configuração ótima, ou a escolhida, a
												 configuração assiciada a este menor valor
												 predito. O segundo modelo será o usado para
												 estimar o tempo de execução da melhor sugestão
												 de configuração.
			suggestion_names (list[str]): Nomes das variáveis de usadas para
																		definir as configurações dos
																		recursos usadas ao treinar os
																		modelos. São essas configurações que
																		serão as sugeridas pelo script de
																		otimização usado pelo usuário.
			application_names (list[str]): Nomes das variáveis da aplicação
																		 usadas ao treinar os modelos, que
																		 são passadas ou derivadas das
																		 variáveis de aplicação passadas
																		 pelos usuários.
			user_names (list[str]): Nomes das variáveis definidas pelos 
															usuários quando usarem o script de 
															otimização para escolher a melhor sugestão
															de configuração para executar a aplicação.
															Podem ou não ser as mesmas variávis de
															application_names, tudo dependerá de como
															as variáveis da aplicação definidas pelo
															usuário são convertidas em variáveis da
															aplicação efetivamente usadas nos 
															treinamentos. 
			estimated_parameters (dict): Diciońario cujas chaves indicam as
																	 variáveis alvo treinar os dois
																	 modelos. O nome da variável alvo do
																	 primeiro modelo, o que auxilia na
																	 escolha da melhor sugestão de
																	 configuração, é referenciado pela 
																	 chave 'suggestion' do dicionário. Já
																	 a variável alvo do segundo modelo,
																	 que estima os tempos de execução das
																	 melhores seguestões de configuração,
																	 será definida na chave 'time' do 
																	 dicionário. 
			model (BasicEstimator): Referência para o objeto não inicializado
															do melhor modelo escolhido pelo critério
															de avaliação dos modelos que, no caso do
															script treinador, é a validação cruzada,
															utilizando a validação LOGO, dos modelos
															escolhidos para serem treinados e para
															os queis escolhemos os melhores valores
															para os seus hiperparâmetros.
			model_params (dict): Dicionário em que a chave é o nome de um dos
													 hiperparâmetros otimizados para o modelo com
													 os melhores valores escolhidos mais, se
													 existirem, os hiperparâmetros fixos e os seus
													 valores.
			verbose (bool): Habilita/desabilita as informações de verbosidade.                   

		Retorna:
      tuple: Se não ocorrerem erros, a função retorna uma referência
						 para o próprio objeto da classe SuggestionsPredictor para o
						 qual invocamos a função fir, como é comum em alguns modelos
						 da sckit-learn. Se algum erro ocorrer, gera uma exceção do
						 Python.
		"""

		# Verifica se data é um DataFrame do Pandas.
		if not isinstance(data, pd.DataFrame):	
			raise ValueError("Conjunto de dados com os dados do teste em um formato"
											 f"inválido {type(data)}! Deveria ser uma referência"
											 "para um objeto DataFrame do Pandas.")
		
		# Verifica se cada nome em suggestion_names é o nome de uma das 
		# variáveis de configuração definidas pelas colunas em data.
		if not pd.Index(suggestion_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
		 							   f"variáveis de configuração {','.join(suggestion_names)} "
										 f"é uma das variáveis {','.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em application_names é o nome de uma das 
		# variáveis da aplicação definidas pelas colunas em data.
		if not pd.Index(application_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(application_names)} é "
										 f"uma das variáveis {', '.join(data.columns)} do "
										 "conjunto de dados dos testes!")

		# Verifica se cada nome em user_names é o nome de uma das variáveis
		# da aplicação definidas pelas colunas em data, e usadas para
		# definir os grupos da validação LOGO.
		if not pd.Index(user_names).isin(data.columns).all():
			raise KeyError("Nem todas os nomes de variáveis dados na lista de "
									   f"variáveis de aplicação {','.join(user_names)} é uma "
										 f"das variáveis {','.join(data.columns)} do conjunto de "
										 "dados dos testes!")
		
		# Verifica o nome em predicted_name é o nome da variável que será a 
		# usada como variável alvo dos modelos, que também será a variável
		# usada ao escolher a melhor sugestão de configuração.
		if not pd.Index(estimated_parameters.values()).isin(data.columns).all():
			raise KeyError("Nome inválido de uma das variáveis alvo a serem "
									   "preditas! Um dos nomes em "
										 f"{', '.join(estimated_parameters.values())} não é o "
										 f"nome de uma das variáveis {','.join(data.columns)} do "
										 "conjunto de dados dos testes!")

    # Inicializa o campo do objeto com uma referência para a lista com
		# os nomes das variáveis de configuração.
		self.suggestion_names = suggestion_names

    # Inicializa o campo do objeto com uma referência para a lista com
		# os nomes das variáveis de aplicação usadas nos treinamentos.
		self.application_names = application_names

    # Inicializa o campo do objeto com uma referência para a lista com
		# os nomes das variáveis de aplicação definidas diretamente pelos
		# usuários ou por conversão do que foi definido pelo usuário.
		self.user_names = user_names

		# Inicializa o campo do objeto para o nome da variável alvo usada
		# para auxiliar na escolha da melhor sugestão de configuração (nos
		# nossos testes preliminares, é a 'EDP').
		self.predicted_name = estimated_parameters['suggestion']

		# Inicializa o modelo a ser treinado para auxiliar a escolha da
		# melhor sugestão de configuração, cuja classe é passada em model,
		# e utilizando os hiperparâmetros definidos no dicionário 
		# model_params.
		self.model = model(**model_params)

		# Inicializa o DataFrame X a ser usado como características ao
		# treinar os dois modelos.
		self.X = data[self.suggestion_names+self.application_names].copy()

		# Inicializa a Series y a ser usadaa ao treinar o modelo que auxilia
		# na escolha da melhor sugestão de configuração.
		self.y = data[self.predicted_name].copy()

		# Treina o modelo usado para auxiliar a escolha da melhor sugestão 
		# de configuração, usando X e y.
		self.model.fit(self.X, self.y)

    # Se a variável alvo usada para predizer o tempo da melhor sugestão
		# de configuração foi definida, o que é indicado pelo dicionário 
		# estimated_parameters ter a chave 'time', também treina um modelo
		# para predizer o tempo da melhor sugestão de configuração. 
		if 'time' in estimated_parameters:
			# Vamos teinar também um modelo para estimar o tempo de execução
			# da melhor sugestão de configuração. Como náo existe nenhum 
			# impedimento de a variável alvo usada para escolher a melhor
			# sugestão de configuração ser a mesma para estimar o tempo de
			# execução desta sugestão, primeiramente verificamos se os nomes
			# das variáveis albo são os mesmos.
			if estimated_parameters['time'] == estimated_parameters['suggestion']:
				# Os nome da variável alvo para auxiliar na escolha da melhor
				# sugestão de configuração é o mesmo do que o da variável
				# alvo para estimar o tempo de execução desta sugestão, ou seja,
				# a variável alvo é a mesma.

				# O nome da variável alvo do modelo treinado para estimar o
				# o tempo de execução da melhor sugestão terá o mesmo nome do da
				# variável alvo usada pelo modelo que auxilia a escolha dessa
				# confiuração. Logo, o campo predicted_time_name será igual ao
				# campo predicted_name.
				self.predicted_time_name = self.predicted_name 

				# O y também será o mesmo usado ao treinar o modelo que auxilia
				# a escolha da melhor sugestão de configuração. Logo, o campo
				# y_time será igual ao campo y.
				self.y_time = self.y

				# O modelo para estimar o tempo será simplesmente uma referência
				# para o modelo já treinado que auxilia na escolha da melhor
				# configuração. Logo o campo do objeto model_time será igual ao
				# camnpo model do objeto.
				self.model_time = self.model	
			else:	
				# Neste caso, os nomes das variáveis alvo do modelo para
				# auxiliar	na escolha da melhor sugestão de configuração e na
				# estimativa do tempo de execução desta sugestão serão variáveis
				# alvo diferentes.
				
				# Inicializa o campo do objeto para o nome da variável alvo 
				# usada para estimar o tempo de execução da  melhor sugestão de
				# configuração (nos nossos testes preliminares, é a 
				# 'ElapsedRaw').
				self.predicted_time_name = estimated_parameters['time']

				# Como as variáveis alvo são diferentes, precisaremos 
				# inicializar um novo modelo, cuja classe é passada em model,
				# e utilizando os hiperparâmetros definidos no dicionário 
				# model_params.
				self.model_time = model(**model_params)


        # Como a variável alvo é diferemte. inicializa a Series y_time 
				# a ser usadaa ao treinar o modelo que estima o tempo de
				# execução da melhor sugestão de configuração.
				self.y_time = data[self.predicted_time_name].copy()

				# Treina o modelo usado para estimar o tempo de execução da
				# melhor sugestão  de configuração, usando X e y_time.
				self.model_time.fit(self.X, self.y_time)
		else:
			# Se a variável não for definida, então não teremos um modelo para
			# estimar os tempos das melhores sugestões de condifuração.
			# TODO: Será que teria problema em tornar o que é opcional 
			# obrigatório? Isso simplificaria um pouco o código e não afetaria
			# o que já fizemos, pois usamos um modelo para fazer as
			# estimativas de tempo.
			self.model_time = None		

		# Descobre os nomes de todas as variáveis relevantes, que são as
		# variáveis de configuração, as de aplicação obtidas diretamente do
		# usuário ou convertidas a partir do que o usuário irá fornecer, 
		# as váriáveis da aplicação definidas pelo uauŕio ou obtidas delas,
		# e as variáveis alvo usadas nos dois modelos. A conversão de lista
		# para conjunto e depois novamente para lista é porque podem existir
		# variáveis comuns nas listas application_names e user_names.
		columns_names = list(
			set(
				suggestion_names
				+ application_names
				+ user_names
				+ list(estimated_parameters.values())
			)
		)

		# Salva o dataframe usado para treinar o(s) modelo(s), com todas as
		# colunas usadas nos treinamentos, e as colunas definidas em
		# user_names que serão usadas para criar o DataFrame do oráculo.
		self.dataset = data[columns_names].copy()

		# Cria uma lista no campo suggestion_params do objeto com todos os
		# possíveis valores para cada variável de configuração definida em
		# suggestion_names. 
		self.suggestion_params = {
			col: list(data[col].unique())
			for col in suggestion_names
		}

		# Cria uma lista no campo application_params do objeto com todos os
		# possíveis valores para cada variável da aplicação definida em
		# application_names. 
		self.application_params = {
			col: list(data[col].unique())
			for col in application_names
		}

		# Cria uma lista no campo user_params do objeto com todos os 
		# possíveis valores para cada variável da aplicação definida em
		# user_names. 
		self.user_params = {
			col: list(data[col].unique())
			for col in user_names
		}

		# Imprime os parâmetros usados para treinar o(s) modelo(s) e os 
		# usados para gerar a tabela do oráculo na função que retorna o
		# DataFrame com o oráculo para o conjunto de dados com os testes.
		if verbose:
			print("➡️  Parâmetros de sugestão usados no treinamento: "
				    f"{self.suggestion_params}")
			print("➡️  Parâmetros de aplicação usados no treinamento: "
				    f"{self.application_params}")
			print("➡️  Parâmetros de usuário usados no treinamento: "
				    f"{self.user_params}")
			print("➡️  Variável alvo do modelo auxiliar usado na escolha da melor "
				    f"sugestão: {self.predicted_name}")
			if self.predicted_time_name is not None:
				print(f"➡️  Varíavel alvo do modelo predizer o tempo de execução da "
					     "melhor sugestão: {self.predicted_time_name}")
			print("➡️  X usado no treinamento dos modelos:")
			print("\n", self.X.to_markdown(tablefmt="grid", floatfmt=".2f" ), 
				    "\n", sep="")
			print("➡️  y usado no treinamento do modelo auxiliar para escolher a :")
			print("melhor sugestão de configuraçaõ:", "\n", 
				    self.y.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
			if self.model_time is not None:
				print("➡️  y usado pelo modelo para a predição dos tempos das ")
				print("melhores sugestões de coinfiguração", "\n", 
					    self.y_time.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
							sep="")
			print("➡️  Dataframe contendo todas as variáceis usadas nos "
				    "treinamentos:")
			print("\n", data.to_markdown(tablefmt="grid", floatfmt=".2f"), 
				    "\n", sep="")

		# Retorna uma referência para o próprio objeto para o qual foi chamada a
		# função fit.	
		return self 		
		
	def get_oracle(self, verbose=False):
		"""
		Função para retornar o DataFrame com os oráculos para cada grupo
		que seria formado com as variáveis de configuração (que foram
		armazenadas, pela função fit, no campo suggestion_names do obejto) e
		as variáveis do usuário (que foram armazenadas no campo user_names),
		cosiderando a variável alvo do modelo auxiliar usado para escolher a
		melhor sugestçao de configuração, já que o oráculo é baseado na
		mediana desta variável alvo para todas as repetições de um mesmo
		teste definido por uma das possíveis combinações dos valores dessas
		variáveis (dados nos campos suggestion_params e user_params). A
		função fit precisa ter sido chamada, pois a função usa os campos
		suggestion_names e user_names do objeto para gerar o DataFrame com
		os oráculos.

		Parâmetros:
			verbose (bool): Habilita/desabilita as informações de verbosidade.                   

		Retorna:
			DataFrame: Se não ocorrerem erros, retorna uma referência para um
								 objeto DataFrame com o oráculo para cada combinação dos
								 possíveis valores para os parâmetros do usuário
								 definidos pelo campo user_params do objeto. Se algum
								 erro ocorrer, gera uma exceção do Python.
		"""

	  # Verifica se o fit foi feito, ou seja, se a função fit foi chamada,
		# pois apesar de o oráculo não depender do fit, os campos usados do
		# objeto somente esterão definidos depois do fit.
		if self.model is None:
			raise ValueError("O modelo ainda não foi treinado!")

		# Lista auxiliar com todas os nomes das variáveis de sugestão (que
		# estão na lista do campo suggestion_names do objeto) e de uruário (
		# que estão no campo user_names do usuário, porque são estas
		# variáveis que definem os possíveis testes diferentes).
		group_cols = self.suggestion_names + self.user_names

		# Cria o DataFrame auxiliar para calcular o oráculo com as medianas,
		# de todos as repetições, para cada possível combinação de valores
		# das varíaveis de aplicação e do usuário, cujos nomes estão na
		# lista group_cols.

		df_aux = (
			self.dataset.groupby(group_cols)[self.predicted_name]
			.median()
			.reset_index()
		)

		# Se a verbosidade estiver habilitada, imprime o DataFrame df_aux
		# criado anteriormente com as medianas.
		if verbose:
			print("➡️  Medianas da variável {self.predicted_name} para todas as "
				    "repetições de cada combinação dos valores das variáveis "
						f"{self.suggestion_names+self.user_names}:")
			print("\n", df_aux.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
				    sep="")

		# Cria o DataFarme do oráculo, em que os índices inicialmente serão
		# compostos, sendo definidos por possíveis valores das variáveis de
		# usuário definidas no campo user_names do objeto, e um valor que
		# define a ??? e que será descartado, pois não é uma das variáveis
		# do usuário.
		df_oracle = df_aux.groupby(self.user_names).apply(
				lambda x: x[
						x[self.predicted_name] == x[self.predicted_name].min()
				],
				include_groups=False,
		)

		# Remove o último nível do índice, pois não está associado aos
		# valores de uma das variáveis do usuário, e depois usa a função
		# reset_index do Pandas, para tornar os índices, que são os valores
		# das variáveis do usuário, em colunas, e criar um índice sequencial
		# para o DataFrame, para uma melhor organização.
		df_oracle = df_oracle.droplevel(level=-1).reset_index()

		# Se a verbosidade estiver habilitada, imprime o DataFrame com os
		# oráculos.
		# TODO: Será que esta impressão é necessário? Não sei mais se é
		# logico imprimir o retorno da função que pode ser impresso pelo
		# programa que chamou a função get_oracle do objeto, como faz o
		# script de treinamento.
		if verbose:
			print("➡️  Dataframe do oráculo:")
			print("\n", df_oracle.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
				    sep="")

		# Retorna uma referÊncia para o DataFrame criado com as informações
		# dos oráculos.
		return df_oracle
	
	def get_importances(self, verbose=True):
		"""
		Função para retornar o DataFrame com as importâncias do modelo 
		treinado com a variável alvo usada para auxliar na escolha da melhor
		sugestão de configuração, se o modelo escolhido definir importâncias
		para cada característica usada para treinar o modelo. As
		características são, no modelo auxiliar, as variáveis de
		configuração e as variáveis da aplicação. 

		Parâmetros:
			verbose (bool): Habilita/desabilita as informações de verbosidade.                   

		Retorna:
			DataFrame | None: Se não ocorrerem erros, retorna uma referência
			                  para um objeto DataFrame com as importâncias de
												cada uma das características ao treinar o modelo,
												ou seja, as importânias das variáveis de
												configuração e das variáveis da aplicação, se o
												modelo calcula as importâncias, e None se o
												modelo não calcula as importâncias. Se algum
												erro ocorrer, gera uma exceção do Python.
		"""

	  # Verifica se o fit foi feito, ou seja, se a função fit foi chamada,
		# pois as importâncias somente são geradas, quando definidas, após o 
		# modelo ser treinado.
		if self.model is None:
			raise ValueError("O modelo ainda não foi treinado!")

		# Verifica se o objeto da classe do modelo tem o campo
		# feature_importances_ no qual são definidas as importâncias das
		# características (variáveis de configuração e da aplicação) usadas
		# ao treinar o modelo que auxilia na escolha da melhor sugestão de
		# configuração.
		if hasattr(self.model, "feature_importances_"):
			# Obtém o vetor com as importâncias de cada característica, na
			# ordem em que foram considerdas, ou seja, a ordem definida pelas
			# colunas do DataFrame X usado ao treinar o modelo
			importances = getattr(self.model, "feature_importances_")
			# Verifica se o ojeto da classe do modelo tem o campo
			# feature_names_in_ com a ordem em que as características foram 
			# consideradas, ou seja, a ordem definida pelas colunas do
			# DataFrame X usado ao treinar o modelo.
			if hasattr(self.model, "feature_names_in_"):
				# Se existir o campo, usa a ordem dada nele, sendo que é
				# necessário converter o campo para uma lista.
				importances_names = getattr(self.model, "feature_names_in_").tolist()
			else:
				# se o campo não existir, usa a ordem definida pelas colunas do
				# campo X usado ao treinar o modelo.
				importances_names = list(self.X.columns)

			# Cria um dicionário auxiliar para a criação do DataFrame final.
			# A chave 'nome' armazena os nomes das características na ordem
			# definida pelas colunas do DataFrame X usado ao treinar o modelo,
			# e a chave 'values' que armazena as importâncias das
			# características, também na mesma ordem das colunas de X.
			importances_dict = {
				'names': importances_names,
				'values': importances
			}

			# Cria o DataFrame com as importâncias a partir do dicionário
			# criado. Ele terá duas colunas, 'names' com os nomes das
			# importâncias, e 'values' com os valores das importâncias.
			importances_df = pd.DataFrame(importances_dict)	

			# Se a verbosidade estiver habilitada, imprime o dicionário
			# auxiliar criado no qual a chave 'names' tem os nomes das
			# características e a chave 'values' tem os valores das
			# importâncias e também o DataFrame final com as importâncias
			# gerado a partir deste dicionário.
			if verbose:
				print(f"➡️  Diciońario com as importâncias: {importances_dict}")		
				print("➡️  Dataframe das importâncias")
				print("\n", importances_df.to_markdown(tablefmt="grid", 
																					 		 floatfmt=".2f"))
		else:		
			# Se o objeto da classe do modelo não possuir um campo
			if verbose:
				print("➡️  O modelo usado não define as importâncias das variáveis!")		
			importances_df = None

		# Retorna o DataFrame criado com as características e suas
		# importâncias.
		return importances_df
		    	
	def _predict_suggestions_data(self, user_applicaion_params, 
															  custom_suggestions_params=None, verbose=False):
		"""
		Função auxiliar para construir um DataFrame X com todas as possíveis
		configurações para executar uma apluiação. Se o parâmetro opcional 
		custom_suggestions_params com um dicionário com as configurações
		customizadas do usuário não for passado, X será composto por todas
		as configurações avaliadas durante o treinamento do modelo que preve
		a variável alvo usada para escolher a sugestão de configuração para
		a aplicação do usuário do script de otimização. Se o dicionário for
		passado, ele terá uma chave para cada componente de uma das
		configurações, e o valor associado ao componente será uma lista com
		os possíveis valores para este compomente, definidos pelo usuário.
		Neste caso, todas as combinações dos valores de todos os componentes
		serão usadas para definir quais configurações serão avaliadas. Esta
		função é auxiliar e somente faz a predição da variável alvo usando o
		modelo auxiliar para todas as configurações (usadas no treinamento
		ou customizadas) e não foi projetada para ser usada externamente ao
		criar um objeto da classe.

		TODO: Quando os usuários passam as configurações costomizadas em
		custom_suggestions_params, podem existir combinações que não seriam
		válidas, mas desconsiderei porque, como definir as configurações
		customizadas seria atilizado por usuários avançados, considerei que
		o usuário não faria combinações inválidas. Podemos fazer uma regra
		simples como, por exemplo, quando a configuração é composta por
		nós, processos por nó e threads por processo, que o número de 
		threads multiplaco pelo número de procesos não pode ser maior do que
		o maior número de threads passado para a função (os valores já são
		ajustados considerando os limites de todas as partições que podem
		ser usadas). Acredito que isso possa ser feito com um campo especial
		no dicionário custom_suggestions_params, por exemplo, uma chave com
		o nome 'rules'.

		Parâmetros:
			user_applicaion_params (dict): Dicionário com os parâmetros da 
																		 aplicação definidos pelo usuário do 
 																		 script de otimização, sendo as
																		 variáveis as usadas ao treinar o
																		 modelo. Estes valores serão fixos
																		 ao fazer a predição para o conjunto
																		 de possíveis configurações, pois
																		 estamostentando escolher a melhor
																		 sugestão de configuração para a
																		 execução da aplicação do usuário
																		 deifnida.
			custom_suggestions_params (dict): Parâmetro opcional que, se for
																				passado conterá, para cada
																				componente de uma possível
																				configuração, os valores para
																				esse componente. É um dicionário
																				em que a chave é o componente e
																				o valor assiciado à chave é a
																				lista com os possíveis valores
																				para o componente. As
																				configurações consideradas serão
																				as obtidas por todas as
																				possíveis combinações dos
																				valores dos componentes. Se o 
																				parâmetro não for passado, serão
																				usadas todas as configurações
																				usadas ao treinar o modelo
																				auxiliar para escolher a melhor
																				sugestão de configuração.
			verbose (bool): Habilita/desabilita as informações de verbosidade.                   

		Retorna:
			tuple: Se não ocorrerem erros, retorna uma tupla em que o primeiro
						 componente é o y predito pelo modelo auxiliar para cada
						 configuração e o segundo componente e o DataFrame com
						 todas as configurações consideradas. Se algum erro ocorrer,
						 gera uma exceção do Python.
		"""

	  # Verifica se o fit foi feito, ou seja, se a função fit foi chamada,
		# pois apesar de o oráculo não depender do fit, os campos usados do
		# objeto somente esterão definidos depois do fit.
		if self.model is None:
			raise ValueError("O modelo ainda não foi treinado!")

		# Cria uma variável auxiliar user_applicaion_params com os
		# parâmetros da aplicação passados pelo usuário, ordenados e
		# convertidos para uma lista.
		user_applicaion_params = sorted(user_applicaion_params.keys())

		# Cria uma variável auxiliar self_params com os parâmetros da
		# aplicação usados ao treinar o modelo, ordenados e convertidos
		# para uma lista.
		self_params = sorted(self.application_params.keys())

		# Verifica se os parâmetros dd aplicação definodos pelo usuário e 
		# passados como chaves no dicionário user_applicaion_params são
		# exatamente os mesmos parâmetros da aplicação usados ao treinar o
		# modelo (não podem ter parâmetros ausentes, pois as predições das
		# variáveis alvo precisam usar todos os parâmetros da aplicação). 
		# Isso será feito usando as variáveis auxiliares criadas, para 
		# verificar se os parâmetros da aplicação passados pelo usuário são
		# os mesmos parâmetros usados ao treinar o modelo, e se não forem,
		# gera uma exceção do tipo KeyError.
		if (user_applicaion_params != self_params):
			raise KeyError("Os parâmetros de aplicação "
									   f"{', '.join(user_applicaion_params)} inválidos! "
										 f"Deveriam ser os parâmerros {', '.join(self_params)}")

		# Constrói o X usado para fazer a predição da variável alvo do
		# modelo auxiliar para escolher a melhor sugestão de configuração. 
		# Se o parâmetro custom_suggestions_params não for passado, o X será
		# construído com todas as configurações usadas ao treinar o modelo
		# auxiliar, e se o parâmetro for passado, o X será construído com
		# todas as combinações dos valores de cada componente da
		# configuração, que são passados como listas no dicionário
		# custom_suggestions_params, como cada componente da configuração
		# sendo uma chave do dicionário.
		if custom_suggestions_params is None:
		  # Cria um DataFrame X usando as opções de configuração usadas para
			# treinar o modelo auxiliar, pois não foram passadas as opções de
			# configuração customizadas pelo usuário. X será criado usando a
			# função de agrupamento (groupby) do Pandas, que agrupa os dados
			# do DataFrame, com todas as variáveis de configuração usadas ao
			# treinar o modelo modelo auxiliar, usando o DataFrame armazenado
			# no campo dataset do objeto, sendo o cálculo da mediana da
			# variável alvo feito somente para o agrupamento poder ser feito, 
			# pois o que queremos é apenas o DataFrame com todas as
			# configurações usadas ao treinar o modelo auxiliar. Logo, depois
			# de resetar os índices para tornar eles colunas de X, pois são
			# os componentes de cada configuração, uma cópia é feita e X terá
			# o Dataframe final sem a coluna da variável alvo.
			X = (
					self.dataset.groupby(self.suggestion_names)[self.predicted_name]
					.median()
					.reset_index()
					.copy()
					.drop(columns=[self.predicted_name])
			)
		else:
		  # Cria um X usando as opções de configuração passadas como
			# parâmetro em custom_suggestions_params, que é um dicionário em
			# que a chave indica o componente da configuração e o valor
			# associado à chave é a lista com os possíveis valores para o
			# componente. Todas as combinações dos valores de todos os
			# componentes serão usadas para definir quais configurações serão
			# avaliadas e, consequentemente, o X que será usado para fazer a
			# predição da variável alvo usando o modelo auxiliar.
			# TODO: Ver a observção sobre as configurações customizadas do
			# usuário, que podem gerar combinações inválidas, e se seria
			# interessante fazer uma regra simples para evitar combinações
			# inválidas.

			# Cria uma lista com todas as combinações dos valores de todos os
			# componentes da configuração, que são passados como listas no
			# dicionário custom_suggestions_params, como cada componente da
			# configuração sendo uma chave do dicionário. A função product do
			# módulo itertools do Python é usada para gerar todas as possíveis
			# combinações dos valores de todos os componentes de uma
			# configuração. No final da função product, a lista será composta
			# por cada configuração, sendo cada configuração uma tupla com os
			# valores de cada componente dessa configuração.
			suggestions_combinations = list(
					itertools.product(*custom_suggestions_params.values())
			)

			# Depois de criada a lista de tuplas suggestions_cobinations, 
			# que contém todas as configurações, é criado um DataFrame X a
			# partir dessa lista, em que cada coluna do DataFrame será um
			# componente da configuração, e o nome de cada coluna será o nome
			# do componente da configuração, que é a chave do dicionário
			# custom_suggestions_params, pois já verificamos que todas as
			# chaves do dicionário são nomes de variáveis de configuração
			# usadas ao treinar o modelo auxiliar, e na mesma ordem dessas
			# variaveis no Dataframe armazenado no campo dataset do objeto.
			X = pd.DataFrame(suggestions_combinations, 
										   columns=custom_suggestions_params.keys())

		# Uma vez criado o X com todas as possíveis configurações, é
		# necessário adicionar as colunas com os parâmetros da aplicação
		# definidos pelo usuário. O loop for a seguir examina cada chave do
		# dicionário user_applicaion_params, que são os parâmetros da
		# aplicação definidos pelo usuário, e adiciona uma coluna ao
		# DataFrame X com o nome do parâmetro da aplicação e o valor
		# definido pelo usuário deste parâmetro.
		for param_name in user_applicaion_params.keys():
			X[param_name] = user_applicaion_params[param_name]

		# Reordena as colunas do DataFrame X para que elas fiquem na mesma
		# ordem das colunas do DataFrame X usado ao treinar o modelo
		# auxiliar.
		X = X[self.X.columns]	
   	
		# Se a verbosidade estiver habilitada, imprime o DataFrame X criado
		# com todas as possíveis configurações para a aplicação do usuário,
		# que serão usadas para fazer a predição da variável alvo do modelo
		# auxiliar.
		if verbose:
			print("➡️  X usado quando foi predito todos os valores da variável "
				    f"alvo {self.predicted_name} para todas as possíveis sugestões:")	
			print("\n", X.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

		# Faz a predição para todas as possíveis configurações, que estão no
		# DataFrame X criado anteriormente.
		y_pred = self.model.predict(X)

		# Se a verbosidade estiver habilitada, imprime o vetor com os
		# valores preditos da variável alvo do modelo auxiliar para todas as
		# possíveis configurações, que estão no DataFrame X criado
		# anteriormente.
		if verbose:
			print(f"➡️  y predito da variável alvo {self.predicted_name} para todas "
				    "as possíveis sugestões:")
			# Cria uma Series do Pandas a partir do vetor y_pred, para
			# facilitar a impressão do vetor com os valores preditos y_pred,
			y_pred_series = pd.Series(y_pred)	
			print("\n", y_pred_series.to_markdown(tablefmt="grid", floatfmt=".2f"), 
				    "\n", sep="")

		# Retorna uma tupla com o vetor y_pred com os valores preditos da
		# variável alvo do modelo auxiliar para todas as possíveis
		# configurações, e o DataFrame X com todas as possíveis
		# configurações usadas ao fazer as predições em y_pred.
		return (y_pred, X)
	
	def get_suggestion(self, user_applicaion_params, 
										 custom_suggestions_params=None, verbose=False):
		"""
		Função para retornar a melhor sugestão de configuração para os
		parâmetros da aplicação definidos pelo usuário do script de
		otimização e passados no dicionário user_application_params, 
		passando os parâmetros da aplicação definidos pelo usuário do script
		de otimização, e opcionalmente um dicionário com as configurações
		customizadas do usuário. Se o parâmetro opcional
		custom_suggestions_params com um dicionário com as configurações 
		customizadas do usuário não for passado, a melhor sugestão será 
		escolhida entre todas as configurações usadas ao treinar o modelo 
		auxiliar que preve a variável alvo usada para escolher esta melhor
		sugestão de configuração.	Junto com a configuração, também são
		retornados o DataFrame com todas as configurações consideradas, o 
		vetor com os valores preditos da variável alvo do modelo auxiliar
		para todas as configurações consideradas, o menor valor predito da
		variável alvo do modelo auxiliar, a posição do menor valor predito
		da variável alvo do modelo auxiliar, e, se o modelo para estimar o
		tempo de execução da melhor sugestão de configuração foi treinado, 
		ou seja, a variável alvo do tempo de execução foi definida, o tempo
		predito para a melhor sugestão de configuração. 

		Parâmetros:
			user_applicaion_params (dict): Dicionário com os parâmetros da 
																		 aplicação definidos pelo usuário do 
 																		 script de otimização, sendo as
																		 variáveis as usadas ao treinar o
																		 modelo. Estes valores serão fixos
																		 ao fazer a predição para o conjunto
																		 de possíveis configurações, pois
																		 estamostentando escolher a melhor
																		 sugestão de configuração para a
																		 execução da aplicação do usuário
																		 deifnida.
			custom_suggestions_params (dict): Parâmetro opcional que, se for
																				passado conterá, para cada
																				componente de uma possível
																				configuração, os valores para
																				esse componente. É um dicionário
																				em que a chave é o componente e
																				o valor assiciado à chave é a
																				lista com os possíveis valores
																				para o componente. As
																				configurações consideradas serão
																				as obtidas por todas as
																				possíveis combinações dos
																				valores dos componentes. Se o 
																				parâmetro não for passado, serão
																				usadas todas as configurações
																				usadas ao treinar o modelo
																				auxiliar para escolher a melhor
																				sugestão de configuração.
			verbose (bool): Habilita/desabilita as informações de verbosidade.                   

		Retorna:
			dict: Se não ocorrerem erros, retorna um dicionário com as 
						seguintes chaves: "Suggestion", sendo o valor da chave um 
						dicionário com os parâmetros da sugestão, em que cada chave
						é o nome de um componente (variável de configuração) da 
						melhor sugestão de configuração e o valor da chave o valor
						deste compomente; "Score" com a pontuação dada a melhor 
						sugestão de configuração, que no momento é o menor valor 
						predito da variável alvo do modelo auxiliar; "X" com o 
						DataFrame com todas as configurações consideradas; "y_pred"
						com o vetor com os valores preditos da variável alvo do
						modelo auxiliar para todas as configurações consideradas em
						X; "y_pred_minimum" com o menor valor predito da variável 
						alvo do modelo auxiliar; "y_pred_minimum_position" com a 
						posição do menor valor predito da variável alvo do modelo 
						auxiliar; e, se o modelo para estimar o tempo de execução da 
						melhor sugestão de configuração foi treinado, ou seja, a 
						variável alvo do tempo de execução foi definida, "Time" com
						o tempo predito para a melhor sugestão de configuração. Se
						algum erro ocorrer, gera uma exceção do Python.
		"""

		# Faz a predição dos valores para todas as configurações da base
		# usada para o treinamento e os parâmetros da aplicação passados,
		# usando a função auxiliar _predict_suggestions_data, que retorna
		# uma tupla com o vetor y_pred com os valores preditos da variável
		# alvo do modelo auxiliar para todas as configurações e o
		# DataFrame X com todas as configurações consideradas.
		y_pred, X = self._predict_suggestions_data(user_applicaion_params, 
																						   custom_suggestions_params, 
																							 verbose)
        
    # Descobre a posição do menor valor predito que indicará a posição 
		# da configuração predita, ou seja, a melhor sugestão de
		# configuração.
		y_pred_posmin = y_pred.argmin()

		# Também descobre o mebor valor predito da variável alvo do modelo
		# auxiliar, que será, no momento, também a pontuação da melhor
		# sugestão de configuração.
		y_pred_min = y_pred.min()

		# Se a verbosidade estiver habilitada, imprime o menor valor
		# predito da variável alvo do modelo auxiliar e a posição deste
		# menor valor no vetor com as predições y_pred.
		if verbose:
			print("y mínimo predito para os parâmetros da aplicação "
				    f"{user_applicaion_params}: {y_pred_min} está na posição "
						f"{y_pred_posmin} do vetor de predições!")
			
		# A sugestão será a configuração associada ao menor valor da
		# variável predita no DataFrame X, que é a configuração que está na
		# posição y_pred_posmin em X. Como estamos interessados apenas nos
		# parâmetros da sugestão, que são as variáveis de configuração e
		# somente na posição y_pred_posmin, o DataFrame X é filtrado para
		# obter apenas a linha da posição y_pred_posmin e as colunas com os
		# nomes das variáveis de configuração, que estão armazenadas no
		# campo suggestion_names do objeto. O resultado é convertido para um
		# dicionário, em que cada chave é o nome de um componente da
		# configuração e o valor da chave é o valor deste componente, que
		# será retornado como a melhor sugestão de configuração. Será este
		# o dicionário que será retornado como a melhor sugestão de 
		# configuração no campo "Suggestion" do dicionário retornado pela
		# função.
		y_suggestion = X.loc[y_pred_posmin,self.suggestion_names].to_dict()
		
    # Retorna a sugestão com o manor valor predito para a variável
		# predita pelo modelo.
		y_pred_s = pd.Series(y_pred)

		# 
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

	def get_suggestions(self, user_applications_params_df, 
											custom_suggestions_params=None, verbose=False):
		"""
		Função para retornar a melhor sugestão de configuração para os
		parâmetros da aplicação definidos pelo usuário do script de
		otimização, pa
		
		"""
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