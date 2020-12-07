import tensorflow as tf
import os

def __decode_image(img,channels,image_format):
	if image_format=="jpg" or image_format=="jpeg":
		return tf.io.decode_jpeg(img, channels=channels)
	elif image_format=="png":
		return tf.io.decode_png(img, channels=channels)
	elif image_format=="bmp":
		return tf.io.decode_bmp(img, channels=channels)
	else:
		raise Exception(f"Decoding of {image_format} is not defined")

def load_tf_image(image_path, image_format="jpeg", dim=None, 
	preprocessing_function=None, channels=3, dtype="float32"):
	img = tf.io.read_file(image_path)
	img = __decode_image(img,channels=channels, image_format=image_format)
	if dim is not None:
		img = tf.image.resize(img,dim)
	if preprocessing_function:
		img = preprocessing_function(img)
	img = tf.image.convert_image_dtype(img, dtype)
	return img

class TFSimpleDataset:
	def __init__(self,batch_size, repeat,
		drop_remainder_in_batch=False, 
		num_parallel_calls=tf.data.experimental.AUTOTUNE,
		buffer_size=tf.data.experimental.AUTOTUNE):
		self.batch_size = batch_size
		self.drop_remainder = drop_remainder_in_batch
		self.num_parallel_calls = num_parallel_calls
		self.buffer_size = buffer_size
		self.repeat = repeat

	def create_dataset(self, X, Y=None):
		datasetX = tf.data.Dataset.from_tensor_slices(X)
		if Y is not None :
			datasetY = tf.data.Dataset.from_tensor_slices(Y)
			dataset = tf.data.Dataset.zip((datasetX,datasetY))
		else:
			dataset = datasetX
		dataset = dataset.batch(self.batch_size, 
			drop_remainder=self.drop_remainder)
		if self.repeat:
			dataset = dataset.repeat()
		dataset = dataset.prefetch(buffer_size=self.buffer_size)
		return dataset

class TFImageDataset:
	def __init__(self,batch_size, repeat, image_size, image_format="jpg",
		channels=3, preprocessing_functions=None, drop_remainder_in_batch=False, 
		num_parallel_calls=tf.data.experimental.AUTOTUNE,
		buffer_size=tf.data.experimental.AUTOTUNE):
		self.batch_size = batch_size
		self.image_size = image_size
		self.image_format = image_format
		self.channels = channels
		self.preprocess_fn = preprocessing_functions
		self.drop_remainder = drop_remainder_in_batch
		self.num_parallel_calls = num_parallel_calls
		self.buffer_size = buffer_size
		self.repeat = repeat

	def __get_path_of_image(self, image_folder, image_name, 
		format="jpg", apply_format=False):
		if apply_format:
			return os.path.join(image_folder,f"{image_name}.{format}")
		return os.path.join(image_folder,image_name)

	def create_dataset(self, X, Y=None, directory=None, ext_applied=True):
		if directory:
			X = [self.__get_path_of_image(directory, str(x),
					format=self.image_format, 
					apply_format=ext_applied) 
					for x in X
				]
		datasetX = tf.data.Dataset.from_tensor_slices(X).map(
				lambda path: load_tf_image(path,
					image_format=self.image_format,
					dim=self.image_size,
					channels=self.channels),
				num_parallel_calls=self.num_parallel_calls
		)
		if self.preprocess_fn is not None:
			for edit in self.preprocess_fn:
				datasetX = tf.data.Dataset.from_tensor_slices(datasetX).map(
					edit,
					num_parallel_calls=self.num_parallel_calls
				)
		if Y is not None:
			datasetY = tf.data.Dataset.from_tensor_slices(Y)
			dataset = tf.data.Dataset.zip((datasetX,datasetY))
		else:
			dataset = datasetX
		dataset = dataset.batch(self.batch_size, 
			drop_remainder=self.drop_remainder)
		if self.repeat:
			dataset = dataset.repeat()
		dataset = dataset.prefetch(buffer_size=self.buffer_size)
		return dataset
		