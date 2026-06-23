import pandas as pd
import numpy as np
import sklearn.metrics as skmet
from sklearn.base import BaseEstimator

class BaselineByResource(BaseEstimator):
	def __init__(self, cols_sugestao, mode="max"):  # "max" ou "min"
		self.mode = mode
		self.cols_sugestao = cols_sugestao

	def fit(self, X, y):
		pass

	# Link do Google Genimi para a usca:
	# https://www.google.com/search?q=__sklearn_tags__&client=ubuntu-sn&hs=Qkd&sca_esv=16a54cb5159ee727&channel=fs&udm=50&fbs=ADc_l-acAb_3MMOAUx0zmbUpgBqRiigBgL2I_pgQa-94zvB054Dys3s2x_Qm_GJcU2DlSXgtwykOhjc8tZDD70ApjQy2F_7G9F-2oFDI3kA5ZHm8DWLK6AOUEc35WkjiPwiSq5cgw13vCSAugeyrjdaQyqjHSSjMVLQDnU7BTutdWxZ5D308B1qNQeQ-AQPQCTv-RsdK3xzR7Xp2Z4OEoO00PgJhWKMQsA&aep=1&ntc=1&sa=X&ved=2ahUKEwjV35zQnPyUAxWxl5UCHW-qJJcQ2J8OegQIFRAD&biw=1850&bih=968&dpr=1&mstk=AUtExfDNWvnJieF9Kl0tb0bSESavRUMFPw68UQlnOuPUPNIlPCzxelEBIGA0GOr1GQuCyvsiE07AJVLusIDT36xgpYwZXYljBj-NufiwlnKzC5hjogYNb9GjLv59JhZnWrmdaqbAUVK4RPgevplls-irEoI5F0nKyIGetSY&csuir=1&atvm=1
	def __sklearn_tags__(self):
		# Fetch default tags from the parent BaseEstimator
		tags = super().__sklearn_tags__()

		# Modify specific attributes safely using the new dataclass format
		tags.non_deterministic = False

		return tags

	def predict(self, X):
		R = np.ones(len(X), dtype=int)
		for col in self.cols_sugestao:
			R = R * X[col].to_numpy()
		R = -R if self.mode == "max" else R
		posmin = R.argmin()
		return R

	def score(self, X, y):
		y_pred = self.predict(X)
		return skmet.r2_score(y, y_pred)

	def get_params(self, deep=True):
		return {"mode": self.mode, "cols_sugestao": self.cols_sugestao}


class BaselineMostCommonConfig(BaseEstimator):
	"""
	Escolhe a configuração 'realista' (mais frequente) observada no oráculo.
	"""
	def __init__(self, cols_sugestao, cols_grupo, dados):
		self.dados = dados
		self.cols_sugestao = cols_sugestao
		self.cols_grupo = cols_grupo
		self.conf_oraculo = None

	# Link do Google Genimi para a usca:
	# https://www.google.com/search?q=__sklearn_tags__&client=ubuntu-sn&hs=Qkd&sca_esv=16a54cb5159ee727&channel=fs&udm=50&fbs=ADc_l-acAb_3MMOAUx0zmbUpgBqRiigBgL2I_pgQa-94zvB054Dys3s2x_Qm_GJcU2DlSXgtwykOhjc8tZDD70ApjQy2F_7G9F-2oFDI3kA5ZHm8DWLK6AOUEc35WkjiPwiSq5cgw13vCSAugeyrjdaQyqjHSSjMVLQDnU7BTutdWxZ5D308B1qNQeQ-AQPQCTv-RsdK3xzR7Xp2Z4OEoO00PgJhWKMQsA&aep=1&ntc=1&sa=X&ved=2ahUKEwjV35zQnPyUAxWxl5UCHW-qJJcQ2J8OegQIFRAD&biw=1850&bih=968&dpr=1&mstk=AUtExfDNWvnJieF9Kl0tb0bSESavRUMFPw68UQlnOuPUPNIlPCzxelEBIGA0GOr1GQuCyvsiE07AJVLusIDT36xgpYwZXYljBj-NufiwlnKzC5hjogYNb9GjLv59JhZnWrmdaqbAUVK4RPgevplls-irEoI5F0nKyIGetSY&csuir=1&atvm=1
	def __sklearn_tags__(self):
		# Fetch default tags from the parent BaseEstimator
		tags = super().__sklearn_tags__()

		# Modify specific attributes safely using the new dataclass format
		tags.non_deterministic = False

		return tags

	def fit(self, X=None, y=None):
		# Calcula a mediana dos EDPs para cada repetição definida pelas colunas cols_grupo (parâmetros da aplicação que definem o grupo)
		# e self.cols_sugestao (parâmetros não relacionados à apicação que definem o uso dos recursos).
		df_mean_EDP = self.dados.groupby(self.cols_grupo+self.cols_sugestao).EDP.median().reset_index()

		# Calcula a base com os EDPs do oráculo, considerando somente os menores EDPs.
		df_mean_EDP_oraculo = df_mean_EDP.groupby(self.cols_grupo).apply(lambda x: x[x.EDP == x.EDP.min()], include_groups=False)

		# Muda o dataframe com a contagem para cada sugestão e a média do EDP para cada sugestão.
		df_count_sugestoes = df_mean_EDP_oraculo.groupby(self.cols_sugestao).agg(Ocorrencias=('EDP', 'size'), **{'EDP mediano': ('EDP', 'mean')}).reset_index()

		# Adiciona uma coluna com o uso dos recursos (igual ao BaselineByResource).
		df_count_sugestoes['Custo'] = 1
		for col in self.cols_sugestao:
			df_count_sugestoes['Custo'] = df_count_sugestoes['Custo'] * df_count_sugestoes[col]

		# Determina o número de ocorrências da configuração mais comum do oráculo.
		max_ocorrencias = df_count_sugestoes['Ocorrencias'].max()

		# Considera somente as configurações com a maior ocorrência (suponto que exista somente uma)
		df_sugestoes_oraculo = df_count_sugestoes[df_count_sugestoes['Ocorrencias'] == max_ocorrencias]

		# Agora precisamos decidir o que fazer se tiver mais sugestões com o mesmo número máximo de ocorrências.
		# A primeira ideia seria usar a com o menor EDP médio e o custo, similar ao da classe BaselineByResource,
		# em caso de empate, ordenando df_sugestoes_oraculo, em ordem crescente, pela coluna EDP médio, e escolhendo a primeira coluna.
		df_sugestoes_oraculo = df_sugestoes_oraculo.sort_values(by=['EDP mediano','Custo'], ascending=True)

		# Outra ideia é usar um custo e o EDP médio, para isso precisamos de uma coluna auxiliar para o custo.
		#df_sugestoes_oraculo = df_sugestoes_oraculo.sort_values(by=['Custo','EDP médio da mediana'], ascending=True)

		# Como a primeira linha de df_sugestoes_oraculo tem a configuração escolhida, armazenamos ela.
		# Preciso descobrir o formato final. Ainda não sei qual o melhor formado (converti o dataframe de uma linha em um dict).
		self.conf_oraculo = df_sugestoes_oraculo.iloc[0].drop(labels=['Ocorrencias', 'EDP mediano', 'Custo']).to_dict()

		# Retorna a configuração do oráculo.
		return self.conf_oraculo, df_count_sugestoes

	def predict(self, X):
		R = np.zeros(len(X), dtype=int)
		for col in self.cols_sugestao:
			R = R + np.abs(X[col].to_numpy()-self.conf_oraculo[col])*2+1
		posmin = R.argmin()
		return R

	def score(self, X, y):
		y_pred = self.predict(X)
		return skmet.r2_score(y, y_pred)

	def get_params(self, deep=True):
		return {"cols_sugestao": self.cols_sugestao, "cols_grupo": self.cols_grupo, "dados": self.dados}
