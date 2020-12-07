import tensorflow as tf

class Padding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), padding_type="REFLECT", **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.padding_type = padding_type
    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, 
        	padding_height], [padding_width, padding_width], [0,0] ], 
        	self.padding_type)

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
    def call(self,inputs):
        batch, rows, cols, channels = [i for i in inputs.get_shape()]
        mu, var = tf.nn.moments(inputs, [1,2], keepdims=True)
        shift = tf.Variable(tf.zeros([channels]))
        scale = tf.Variable(tf.ones([channels]))
        epsilon = 1e-3
        normalized = (inputs-mu)/tf.sqrt(var + epsilon)
        return scale * normalized + shift

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1,
        padding=None, normalization=None, activation=None,
        **kwargs):
        super(ConvLayer,self).__init__(**kwargs)
        if padding:
            self.padding=Padding2D(padding=[(k-1)//2 for k in kernel_size], 
                padding_type=padding)
        else:
            self.padding = None
        self.conv2d=tf.keras.layers.Conv2D(filters,kernel_size,strides)
        self.activation=activation
        self.normalization=normalization
    def call(self,inputs):
        if self.padding:
            x=self.padding(inputs)
        else:
            x = inputs
        x=self.conv2d(x)
        if self.normalization:
            x=self.normalization(x)
        if self.activation:
            x=self.activation(x)
        return x

class ConvResidualLayer(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides=1,skip=2
        padding=None,normalization=None, activation=None,
        **kwargs):
        super(ConvResidualLayer,self).__init__(**kwargs)
        self.convs=[]
        for s in range(skip):
            self.convs.append(ConvLayer(filters,kernel_size,strides,
            padding=padding,normalization=normalization,
            activation=activation))
        self.add=tf.keras.layers.Add()
    def call(self,inputs):
        residual=inputs
        x=inputs
        for conv in self.convs:
            x=conv(x)
        x=self.add([x,residual])
        return x

class UpsampleLayer(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides=1,upsample=2,
        padding=None,normalization=None, activation=None,
        **kwargs):
        super(UpsampleLayer,self).__init__(**kwargs)
        self.upsample=tf.keras.layers.UpSampling2D(size=upsample)
        self.conv2d=ConvLayer(filters,kernel_size,strides,
            padding=padding,normalization=normalization,
            activation=activation)
    def call(self,inputs):
        x=self.upsample(inputs)
        x=self.conv2d(x)
        return x