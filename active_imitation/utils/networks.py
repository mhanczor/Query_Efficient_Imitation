import tensorflow as tf

def denseNet(inputs, layer_sizes, reuse=None, flatten=False, name=""):
    """
    Create a typical MLP
    Inputs:
        inputs - tensorflow inputs to feed to the first layer
        layers_sizes - a list of tensorflow layer sizes
        reuse - should tf variables be able to be reused
    Outputs: a tensorflow network of stacked dense layers
    """
    for i, size in enumerate(layer_sizes):
        # Use relu activation for all but the last layer,
        activation = tf.nn.relu if i < len(layer_sizes) - 1 else None
        inputs = tf.layers.dense(inputs=inputs,
                                units=size,
                                reuse=reuse,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name=name + '_' + str(i))
        if activation:
            inputs = activation(inputs)
    if flatten:
        assert layers_sizes[-1] == 1
        inputs = tf.reshape(inputs, [-1])
    return inputs

def scaledOutput(inputs, scaling=1):
    """
    Compress the output within [-1, 1] using a tanh nonlinearity and then scale
    """
    outputs = scaling * tf.tanh(inputs)
    return outputs

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    x = tf.placeholder(tf.float32, [None,1])
    net_output = denseNet(x, [8, 16, 32, 1], reuse=False, name='main')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run(net_output, feed_dict={x:[[12.]]})
    print(out)