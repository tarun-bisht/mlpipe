import pandas as pd
import sklearn.model_selection as ms

class CrossValidation:
	def __init__(self, df, shuffle,random_state=11):
		self.df = df
		self.random_state = random_state
		if shuffle is True:
			self.df = df.sample(frac=1,
				random_state=self.random_state).reset_index(drop=True)

	def hold_out_split(self,percent,stratify=None):
		if stratify is not None:
			y = self.df[stratify]
			train,val = ms.train_test_split(self.df, test_size=percent/100,
				stratify=y, random_state=self.random_state)
			return train,val
		size = len(self.df) - int(len(self.df)*(percent/100))
		train = self.df.iloc[:size,:]
		val = self.df.iloc[size:,:]
		return train,val

	def kfold_split(self, splits, stratify=None):
		if stratify is not None:
			kf = ms.StratifiedKFold(n_splits=splits)
			y = self.df[stratify]
			for train, val in kf.split(X=self.df,y=y):
				t = self.df.iloc[train,:]
				v = self.df.iloc[val, :]
				yield t,v
		else:
			kf = ms.KFold(n_splits=splits)
			for train, val in kf.split(X=self.df):
				t = self.df.iloc[train,:]
				v = self.df.iloc[val, :]
				yield t,v