import sklearn.preprocessing as prep

class EncodeCategories:
	def __init__(self, df, encode_cols, encoding_type, 
		handle_na=False, na_placeholder="NaN"):
		self.df = df
		self.encode_cols = encode_cols
		self.encoding_type = encoding_type
		self.handle_na = handle_na

		self.label_encoders = {}
		self.binary_encoders = {}
		self.one_hot_encoder = None
		self.na_placeholder = na_placeholder

		if self.handle_na:
			self.df = self.__handle_missing_category(self.df, 
				placeholder=na_placeholder)
	
	def __handle_missing_category(self, df, placeholder="NaN"):
		for cat in self.encode_cols:
			df.loc[:, cat] = df.loc[:, cat].astype(str).fillna(placeholder)
		return df

	def __label_encoder_fit(self,df, cat):
		le = prep.LabelEncoder()
		le.fit(self.df[cat].values)
		self.label_encoders[cat] = le

	def __label_encoder_transform(self,df, cat):
		return self.label_encoders[cat].transform(df[cat].values)

	def __binary_encoder_fit(self,df, cat):
		lbl = prep.LabelBinarizer()
		lbl.fit(self.df[cat].values)
		self.binary_encoders[cat] = lbl

	def __binary_encoder_transform(self,df, cat):
		return self.binary_encoders[cat].transform(df[cat].values)

	def __one_hot_fit(self,df, sparse=False):
		ohe = prep.OneHotEncoder(sparse=sparse)
		ohe.fit(self.df[self.encode_cols].values)
		self.one_hot_encoder = ohe

	def __one_hot_transform(self,df, cat):
		return self.one_hot_encoder.transform(df[cat].values)

	def __label_encoder(self, df, fit=True):
		for cat in self.encode_cols:
			if fit:
				self.__label_encoder_fit(df,cat)
			df.loc[:,cat] = self.__label_encoder_transform(df,cat)
		return df

	def __binary_encoder(self, df, fit=True):
		for cat in self.encode_cols:
			if fit:
				self.__binary_encoder_fit(df,cat)
			val = self.__binary_encoder_transform(df, cat)
			df = df.drop(cat, axis=1)
			for i in range(val.shape[1]):
				new_col_name = f"{cat}_bin_{i}"
				df[new_col_name] = val[:, i]
		return df

	def __one_hot_encoder(self, df, sparse=False, fit=True):
		if fit:
			self.__one_hot_fit(df, sparse)
		val = self.__one_hot_transform(df, self.encode_cols)
		for cat in self.encode_cols:
			df = df.drop(cat, axis=1)
			for i in range(val.shape[1]):
				new_col_name = f"{cat}_ohe_{i}"
				df[new_col_name] = val[:, i]
		return df

	def fit(self):
		if self.encoding_type == "label":
			for cat in self.encode_cols:
				self.__label_encoder_fit(self.df,cat)
		elif self.encoding_type == "binary":
			for cat in self.encode_cols:
				self.__binary_encoder_fit(self.df,cat)
		elif self.encoding_type == "onehot":
			self.__one_hot_fit(self.df, False)
		elif self.encoding_type == "onehot_sparse":
			self.__one_hot_fit(self.df, True)
		else:
			raise Exception("specified encoding type not defined")

	def fit_transform(self):
		df = self.df.copy(deep=True)
		if self.encoding_type == "label":
			return self.__label_encoder(df)
		elif self.encoding_type == "binary":
			return self.__binary_encoder(df)
		elif self.encoding_type == "onehot":
			return self.__one_hot_encoder(df)
		elif self.encoding_type == "onehot_sparse":
			return self.__one_hot_encoder(df, True)
		else:
			raise Exception("specified encoding type not defined")

	def transform(self,dataframe):
		if self.handle_na:
			dataframe = self.__handle_missing_category(dataframe, 
				placeholder=self.na_placeholder)
		df = self.df.copy(deep=True)
		if self.encoding_type == "label":
			return self.__label_encoder(df, fit=False)
		elif self.encoding_type == "binary":
			return self.__binary_encoder(df, fit=False)
		elif self.encoding_type == "onehot":
			return self.__one_hot_encoder(df, sparse=False, fit=False)
		elif self.encoding_type == "onehot_sparse":
			return self.__one_hot_encoder(df, sparse=True, fit=False)
		else:
			raise Exception("specified encoding type not defined")

	def inverse_transform(self,data, col):
		if self.encoding_type == "label":
			return self.label_encoders[col].inverse_transform(data)
		elif self.encoding_type == "binary":
			return self.binary_encoders[col].inverse_transform(data)
		elif self.encoding_type == "onehot":
			return self.one_hot_encoder[col].inverse_transform(data)
		elif self.encoding_type == "onehot_sparse":
			return self.one_hot_encoder[col].inverse_transform(data)
		else:
			raise Exception("specified encoding type not defined")

	def get_categories(self):
		if self.encoding_type == "label":
			categories=[]
			for col in self.encode_cols:
				categories.append(self.label_encoders[col].categories_)
			return categories
		elif self.encoding_type == "binary":
			categories=[]
			for col in self.encode_cols:
				categories.append(self.binary_encoders[col].categories_)
			return categories
		elif self.encoding_type == "onehot":
			return self.one_hot_encoder.categories_
		elif self.encoding_type == "onehot_sparse":
			return self.one_hot_encoder.categories_
		else:
			raise Exception("specified encoding type not defined")