import pandas as pd
from .dataset import TFImageDataset
from .dataset import CrossValidation
from .dataset.utils import df_from_image_dirs
from .encodings.categorical import EncodeCategories

class ImageDatasetGenerator:
	def __init__(self, image_size, batch_size, repeat, shuffle=True,
		validation_split=0,
		num_folds = 0,
		preprocessing_functions=None,
		random_state = 11,
		verbose = 0):
	
		self.image_size = image_size
		self.batch_size = batch_size
		self.repeat = repeat
		self.shuffle = shuffle
		self.preprocessing_functions = preprocessing_functions
		self.validation_split = validation_split
		self.folds = num_folds
		self.verbose = verbose
		self.random_state = random_state
		self.encoder = None
		if self.validation_split and self.folds:
			raise Exception("Cannot use validation split and kfolds at same time, consider using anyone of those")

	def __get_target_col_after_encoding(self, df, target_col, encoding):
		col_name = target_col
		if encoding=="onehot":
			col_name = f"{Y_col}_ohe_"
		elif encoding=="binary":
			col_name = f"{Y_col}_bin_"
		cols=[]
		for col in df.columns:
			if col.startswith(col_name):
				cols.append(col)
		return cols

	def from_directories(self, directory, encoding="onehot",
		image_format="jpg", channels=3):

		df = df_from_image_dirs(directory,
			image_format=image_format,
			relative_path=True,
			verbose=self.verbose)

		cross_val = CrossValidation(df,
			shuffle=self.shuffle,
			random_state=self.random_state)

		data_obj = TFImageDataset(batch_size=self.batch_size,
			repeat=self.repeat,
			image_size=self.image_size,
			channels=channels,
			preprocessing_functions= self.preprocessing_functions)

		if self.validation_split:
			train, val = cross_val.hold_out_split(self.validation_split,
				stratify="classes")

			full_dataset = pd.concat([train, val]).reset_index(drop=True)
			enc = EncodeCategories(full_dataset, ["classes"], encoding)
			enc.fit()
			if self.encoder is not None:
				raise Warning("This will reset prevoius encoded classes in encoder. Create a new ImageDatasetGenerator object")
			self.encoder = enc
			train = enc.transform(train)
			val = enc.transform(val)
			del(full_dataset)
			train_dataset = data_obj.create_dataset(X=train.iloc[:,0].values, 
				Y=train.iloc[:,1:].values)
			val_dataset = data_obj.create_dataset(X=val.iloc[:,0].values, 
				Y=val.iloc[:,1:].values)
			return train_dataset, val_dataset

		elif self.folds:
			for train, val in cross_val.kfold_split(self.folds,
				stratify="classes"):
				full_dataset = pd.concat([train, val]).reset_index(drop=True)
				enc = EncodeCategories(full_dataset, ["classes"], encoding)
				enc.fit()
				if self.encoder is not None:
					raise Warning("This will reset previous encoded classes in encoder. Create a new ImageDatasetGenerator object")
				self.encoder = enc
				train = enc.transform(train)
				val = enc.transform(val)
				del(full_dataset)

				train_dataset = data_obj.create_dataset(X=train.iloc[:,0].values, 
					Y=train.iloc[:,1:].values)
				val_dataset = data_obj.create_dataset(X=val.iloc[:,0].values, 
					Y=val.iloc[:,1:].values)
				yield train_dataset, val_dataset
		else:
			enc = EncodeCategories(df, ["classes"], encoding)
			df = enc.fit_transform()
			if self.encoder is not None:
				raise Warning("This will reset prevoius encoded classes in encoder. Create a new ImageDatasetGenerator object")
			self.encoder = enc
			train_dataset = data_obj.create_dataset(X=df.iloc[:,0].values, 
				Y=df.iloc[:,1:].values)
			return train_dataset

	def from_dataframe(self, dataframe, X_col, Y_col, directory=None,
		encoding="onehot", image_format="jpg", channels=3):

		cross_val = CrossValidation(df,
			shuffle=self.shuffle,
			random_state=self.random_state)

		data_obj = TFImageDataset(batch_size=self.batch_size,
			repeat=self.repeat,
			image_size=self.image_size,
			channels=channels,
			preprocessing_functions= self.preprocessing_functions)

		if self.validation_split:
			train, val = cross_val.hold_out_split(self.validation_split,
				stratify=Y_col)

			full_dataset = pd.concat([train, val]).reset_index(drop=True)
			enc = EncodeCategories(full_dataset, [Y_col], encoding)
			enc.fit()
			if self.encoder is not None:
				raise Warning("This will reset prevoius encoded classes in encoder. Create a new ImageDatasetGenerator object")
			self.encoder = enc
			train = enc.transform(train)
			val = enc.transform(val)
			
			target_col = self.__get_target_col_after_encoding(full_dataset, 
				Y_col, 
				encoding)

			del(full_dataset)

			train_dataset = data_obj.create_dataset(X=train.loc[:,X_col].values, 
				Y=train[target_col].values, directory=directory)
			val_dataset = data_obj.create_dataset(X=val.loc[:,X_col].values, 
				Y=val[target_col].values, directory=directory)
			return train_dataset, val_dataset

		elif self.folds:
			for train, val in cross_val.kfold_split(self.folds,
				stratify=True):
				full_dataset = pd.concat([train, val]).reset_index(drop=True)
				enc = EncodeCategories(full_dataset, [Y_col], encoding)
				enc.fit()
				if self.encoder is not None:
					raise Warning("This will reset prevoius encoded classes in encoder. Create a new ImageDatasetGenerator object")
				self.encoder = enc
				train = enc.transform(train)
				val = enc.transform(val)

				target_col = self.__get_target_col_after_encoding(full_dataset, 
					Y_col, 
					encoding)

				del(full_dataset)

				train_dataset = data_obj.create_dataset(X=train.loc[:,X_col].values, 
					Y=train[target_col].values, directory=directory)
				val_dataset = data_obj.create_dataset(X=val.loc[:,X_col].values, 
					Y=val[target_col].values, directory=directory)
				yield train_dataset, val_dataset
		else:
			enc = EncodeCategories(df, [Y_col], encoding)
			df = enc.fit_transform()
			if self.encoder is not None:
				raise Warning("This will reset prevoius encoded classes in encoder. Create a new ImageDatasetGenerator object")
			self.encoder = enc

			target_col = self.__get_target_col_after_encoding(df, 
					Y_col, 
					encoding)

			train_dataset = data_obj.create_dataset(X=df.loc[:,X_col].values, 
				Y=df[target_col].values)
			return train_dataset

		def get_class_indices(self):
			return self.encoder.categories_

