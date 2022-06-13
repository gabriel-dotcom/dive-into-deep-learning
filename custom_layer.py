""" Layers specifically for handling images, text, looping over sequential data, and performing dynamic programming """
# Layer without Parameters
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)

layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))

# we can incorporate our layer as a component in constructing more complex model
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()]) # Sequential groups a linear stack of layers into a model

""" As an extra sanity check, we can send random data through the network and check that the mean is in fact 0. 
    Because we are dealing with floating point numbers, we may still see a very small nonzero number due to quantization. 
"""
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)

# Layers with Parameters
""" Now that we know how to define simple layers, let us move on to defining layers with parameters that can be adjusted through training. 
    We can use built-in functions to create parameters, which provide some basic housekeeping functionality. In particular, they govern access, 
    initialization, sharing, saving, and loading model parameters. This way, among other benefits, we will not need to write custom serialization 
    routines for every custom layer 

    Now let us implement our own version of the fully-connected layer. Recall that this layer requires two parameters, one to represent the weight and 
    the other for the bias. In this implementation, we bake in the ReLU activation as a default. This layer requires to input arguments: in_units and 
    units, which denote the number of inputs and outputs, respectively.
"""
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(
            name='weight', 
            shape=[X_shape[-1], self.units], 
            initializer=tf.random_normal_initializer()
        )

        self.bias = self.add_weight(
            name='bias',
            shape=[self.units],
            initializer=tf.zeros_initializer()
        )

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)

# next instantiate the MyDense class and access its model parameters
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()

# we can directly carry out forward propagation calculations using custom layers
dense(tf.random.uniform((2, 5)))

# we can also construct models using custom layers. Once we have that we can it just like the built-in fully-connect layer
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))