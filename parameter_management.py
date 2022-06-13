""" - Accessing parameters for debugging, diagnostics, and visualizations.
    - Parameter initialization.
    - Sharing parameters across different model components. 
"""

import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])

X = tf.random.uniform((2,4))
net(X)

# Parameter Access
""" When a model is defined via the Sequential class, we can first access any layer by indexing into the model as though it were a list 
Each layer’s parameters are conveniently located in its attribute. 
"""

print(net.layers[2].weights)

""" [<tf.Variable 'dense_1/kernel:0' shape=(4, 1) dtype=float32, numpy=
    array([[ 0.6178552 ],
       [-0.0967508 ],
       [ 0.20512176],
       [ 0.94068813]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>] 

    The output tells us a few important things. First, this fully-connected layer contains two parameters, corresponding to that layer’s
     weights and biases, respectively. Both are stored as single precision floats (float32). Note that the names of the parameters allow us to 
     uniquely identify each layer’s parameters, even in a network containing hundreds of layers.
"""

# Targeted Parameters
""" The following code extracts the bias from the second neural network layer, which returns a parameter class instance, and further accesses that parameter’s value """

print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))

# All parameters at Once
""" When we need to perform operations on all parameters, accessing them one-by-one can grow tedious. The situation can grow especially unwieldy when we work with more 
    complex blocks (e.g., nested blocks), since we would need to recurse through the entire tree to extract each sub-block’s parameters. 
    Below we demonstrate accessing the parameters of the first fully-connected layer vs. accessing all layers. 
"""

print(net.layers[1].weights)
print(net.get_weights())

""" [<tf.Variable 'dense/kernel:0' shape=(4, 4) dtype=float32, numpy=
array([[-0.79512775,  0.83131605, -0.46970606, -0.386196  ],
       [-0.23707926,  0.58012336, -0.4286804 ,  0.80075485],
       [-0.16268367,  0.3944226 , -0.3487624 , -0.7027458 ],
       [-0.56425023, -0.02344972,  0.5696172 , -0.53518283]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
[array([[-0.79512775,  0.83131605, -0.46970606, -0.386196  ],
       [-0.23707926,  0.58012336, -0.4286804 ,  0.80075485],
       [-0.16268367,  0.3944226 , -0.3487624 , -0.7027458 ],
       [-0.56425023, -0.02344972,  0.5696172 , -0.53518283]],
      dtype=float32), array([0., 0., 0., 0.], dtype=float32), array([[ 0.6178552 ],
       [-0.0967508 ],
       [ 0.20512176],
       [ 0.94068813]], dtype=float32), array([0.], dtype=float32)] 

This provides us with another way of accessing the parameters of the network as follows. """

net.get_weights()[1]

# array([0., 0., 0., 0.], dtype=float32)