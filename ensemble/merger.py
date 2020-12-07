import numpy as np
import pandas as pd
from scipy.stats import mode, rankdata

class MergePredictions:
	def __init__(self, ensemble_file, pred_cols):
		self.df = pd.read_csv(ensemble_file)
		self.pred_cols = pred_cols
		self.predictions = np.array(self.__get_predictions(self.df))
		self.targets = self.df.loc[:,"target"]
		self.ids = self.df.loc[:,"id"]

	def __get_predictions(self, df):
		predictions = []
		for col in self.pred_cols:
			preds = df[col].values
			preds = np.array([p.split(",") for p in preds], dtype="float32")
			predictions.append(preds)
		return predictions

	def __check_rank_modes(self, mode):
		modes = ["average", "min", "max", "dense", "ordinal"]
		if mode not in modes:
			raise Exception(f"mode send is not defined use any of {modes}")

	def mean_ensembling(self):
		return np.mean(self.predictions, axis=0)

	def max_ensembling(self):
		return np.max(self.predictions, axis=0)

	def weighted_average_ensembling(self, weights):
		preds = []
		for weight, prediction in zip(weights, self.predictions):
			for i in range(weight):
				preds.append(prediction)
		return np.mean(preds, axis=0)

	def rank_average_ensembling(self, mode="average"):
		self.__check_rank_modes(mode)
		predictions = rankdata(self.predictions, method=mode axis=2)
		return np.mean(predictions, axis=0)

	def weighted_rank_average_ensembling(self, weights, mode="average"):
		self.__check_rank_modes(mode)
		preds = []
		for weight, prediction in zip(weights, self.predictions):
			for i in range(weight):
				preds.append(prediction)
		predictions = rankdata(preds, method=mode, axis=2)
		return np.mean(predictions, axis=0)

	def voting_ensembling(self, voting="hard"):
		predictions = self.predictions
		if voting=="hard":
			predictions = np.argmax(predictions, axis=2)
			predictions, _ = mode(predictions, axis=0)
			return predictions
		elif voting=="soft":
			predictions = np.mean(predictions, axis=0)
			predictions = np.argmax(predictions, axis=2)
			return predictions
		else:
			raise Exception("voting type not defined choose from [hard, soft]")

	def weighted_voting_ensembling(self, weights, voting="hard"):
		preds = []
		for weight, prediction in zip(weights, self.predictions):
			for i in range(weight):
				preds.append(prediction)
		if voting=="hard":
			predictions = np.argmax(preds, axis=2)
			predictions, _ = mode(predictions, axis=0)
			return predictions
		elif voting=="soft":
			predictions = np.mean(preds, axis=0)
			predictions = np.argmax(predictions, axis=2)
			return predictions
		else:
			raise Exception("voting type not defined choose from [hard, soft]")