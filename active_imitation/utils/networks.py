import tensorflow as tf
from active_imitation.utils import ConcreteDropout

def denseNet(inputs, layer_sizes, dropout, dropout_flag, reuse=None, name=""):
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
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
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
            inputs = ConcreteDropout(tf.layers.Dense(size, activation=tf.nn.relu, name=name + '_' + str(i)), 
                                    weight_regularizer=wd, dropout_regularizer=dd, 
                                    trainable=True)(inputs, training=True)
        return inputs
        
        
# if __name__ == "__main__":
#     # import ipdb; ipdb.set_trace()
#     x = tf.placeholder(tf.float32, [None,1])
#     net_output = denseNet(x, [8, 16, 32, 1], reuse=False, name='main')
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     out = sess.run(net_output, feed_dict={x:[[12.]]})
#     print(out)
    
