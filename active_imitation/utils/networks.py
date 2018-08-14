import tensorflow as tf
from active_imitation.utils import ConcreteDropout

def denseNet(inputs, layer_sizes, dropout, dropout_flag, reuse=None, reg_weight=1e-7, name=""):
    """
    Create a typical Deep MLP
    Inputs:
        inputs - tensorflow inputs to feed to the first layer
        layers_sizes - a list of tensorflow layer sizes
        reuse - should tf variables be able to be reused
    Outputs: a tensorflow network of stacked dense layers
    """
    for i, size in enumerate(layer_sizes):
        # Use relu activation
        inputs = tf.layers.dense(inputs=inputs,
                                units=size,
                                activation=tf.nn.relu,
                                reuse=reuse,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_weight),
                                name=name + '_' + str(i))
        inputs = tf.layers.dropout(inputs, rate=dropout, training=dropout_flag)
        
    return inputs
    

def concreteNet(inputs, layer_sizes, wd, dd,  name=""):
        """
        Concrete Dropout Layers
        Inputs:
            inputs - tensorflow inputs to feed to the first layer
            layers_sizes - a list of tensorflow layer sizes
            reuse - should tf variables be able to be reused
        Outputs: a tensorflow network of stacked dense layers
        """
        for i, size in enumerate(layer_sizes):
            # Use relu activation
            inputs = ConcreteDropout(tf.layers.Dense(size, 
                                                    activation=tf.nn.relu,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name=name + '_' + str(i),
                                                    ), weight_regularizer=wd, 
                                                    dropout_regularizer=dd, 
                                                    trainable=True)(inputs, training=True)
        return inputs

    
