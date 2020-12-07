import tensorflow as tf
import numpy as np
import pandas as pd

class Blending:
	def __init__(self, ensemble_file, pred_cols):
		self.df = pd.read_csv(ensemble_file)
		self.pred_cols = pred_cols
		self.predictions = np.array(self.__get_predictions(self.df))
		self.targets = self.df.loc[:,"target"]
		self.ids = self.df.loc[:,"id"]
		self.model = None

	def __get_predictions(self, df):
		predictions = []
		for col in self.pred_cols:
			preds = df[col].values
			preds = np.array([p.split(",") for p in preds], dtype="float32")
			predictions.append(preds)
		return predictions

	def define_neural_network(self, dense_units, output_activation, loss_function,
		activations="relu", metrics=["accuracy"], optimizer="adam"):
		output_units = self.targets.nunique()
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Input(shape=len(self.predictions)))
		if type(activations)==str:
			for units in dense_units:
				model.add(tf.keras.layers.Dense(units=units,activations=activations))
		elif type(activations)==list or type(activations)==tuple:
			for units, activation in zip(dense_units, activations):
				model.add(tf.keras.layers.Dense(units=units,activation=activation))
		else:
			raise Exception("activation should be string, tuple or list")

		model.add(tf.keras.layers.Dense(units=output_units, activation=output_activation))
		model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
		return model

	def fit_nn(self, model, epochs, batch_size=32, callbacks=None, verbose=1):
		if self.model is not None:
			raise Warning("This will override previous blending model")
		self.model = model
		self.model.fit(x=self.predictions, y=self.targets.values, 
			batch_size=batch_size, epochs=epochs, callbacks=callbacks,
			verbose=verbose)

	def fit(self, model):
		if self.model is not None:
			raise Warning("This will override previous blending model")
		self.model = model
		self.model.fit(X=self.predictions, y=self.targets.values)

	def predict(self, X):
		return self.model.predict(X=X)






		








