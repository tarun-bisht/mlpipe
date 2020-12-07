import tensorflow.keras.metrics as m

class Metrics:
	def __init__(self, metric):
		self.metrics = self.__get_metric(metric)

	def __get_metric(self, metric):
		if metric=="auc":
			return m.AUC()
		elif metric=="accuracy":
			return m.Accuracy()
		elif metric=="binary_accuracy":
			return m.BinaryAccuracy()
		elif metric=="categorical_accuracy":
			return m.CategoricalAccuracy()
		elif metric=="binary_crossentropy":
			return m.BinaryCrossentropy()
		elif metric=="categorical_crossentropy":
			return m.CategoricalCrossentropy()
		elif metric=="sparse_categorical_crossentropy":
			return m.SparseCategoricalCrossentropy()
		elif metric=="kl_divergence":
			return m.KLDivergence()
		elif metric=="poisson":
			return m.Poission()
		elif metric=="mse":
			return m.MeanSquaredError()
		elif metric=="rmse":
			return m.RootMeanSquaredError()
		elif metric=="mae":
			return m.MeanAbsoluteError()
		elif metric=="mean_absolute_percentage_error":
			return m.MeanAbsolutePercentageError()
		elif metric=="mean_squared_logarithm_error":
			return m.MeanSquaredLogarithmError()
		elif metric=="cosine_similarity":
			return m.CosineSimilarity()
		elif metric=="log_cosh_error":
			return m.LogCoshError()
		elif metric=="precision":
			return m.Precision()
		elif metric=="recall":
			return m.Recall()
		elif metric=="true_positive":
			return m.TruePositives()
		elif metric=="true_negative":
			return m.TrueNegatives()
		elif metric=="false_positive":
			return m.FalsePositives()
		elif metric=="false_negative":
			return m.FalseNegatives()
		else:
			raise Exception("specified metric not defined")

	def score(self, target, predictions):
		self.metrics.reset_states()
		self.metrics.update_state(target, predictions)
		return self.metrics.result().numpy()

class TopKAccuracyMetrics:
	def __init__(self, metric, k):
		self.metrics = self.__get_metric(metric)
		self.k = k

	def __get_metric(self, metric):
		if metric=="categorical_accuracy":
			return m.TopKCategoricalAccuracy(self.k)
		elif metric=="sparse_categorical_accuracy":
			return m.SparseTopKCategoricalAccuracy(self.k)
		else:
			raise Exception("specified metric not defined")

	def score(self, target, predictions):
		self.metrics.reset_states()
		self.metrics.update_state(target, predictions)
		return self.metrics.result().numpy()