import pandas as pd
import os
import secrets

def create_ensemble_format(models_dict, X, Y=None, id_col=None, path="data/ensemble"):
	os.makedirs(path)
	df = pd.DataFrame()
	if id_col is not None:
		if len(X) == len(id_col):
			df["id"] = id_col
		else:
			raise Exception("number of samples should be equal to id provided in id_col")
	else:
		num = len(X)
		df["id"] = range(1, num+1)
	for name, model in models.items():
		predictions = model.predict(X)
		df[f"{name}"] = [",".join([str(p) for p in pred]) for pred in predictions]
	if Y is not None:
		if len(Y) == len(X):
			df["target"] = [",".join([str(t) for t in y]) for y in Y]
	df.to_csv(os.path.join(path,f"ensemble_{secrets.token_hex(4)}"))



