import tensorflow as tf
# https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
# Padding==Same:
# H_out = H_in * stride
# Padding==Valid
# H_out = (H_in-1) * stride + ksize

def initialize_generator_weights(nf = 32, nz=100):
    # Store layers weight & bias
    # A random value generator to initialize weights.
    random_normal = tf.initializers.RandomNormal()
    # Network parameters.
    # nf = number of filters for 1st residual convolutional layer.
    # nz = random noise dimension
    
    weights_g = {
        # FC Layer: nz inputs, nf*64 units output
        'fc1': tf.Variable(random_normal([nz, nf*32])),
        # FC Layer: nf*64 inputs, 2*2*nf*32 units output
        'fc2': tf.Variable(random_normal([nf*32, 2*2*nf*16])),

        # Residual Transpose Convolutional Layer 1: 4x4 conv, nf*16 input, nf*8 output filters 
        'wu1': tf.Variable(random_normal([4, 4, nf*8, nf*16])),#upsample block has a conv of kernel=4x4 
        'wr1': tf.Variable(random_normal([3, 3, nf*8, nf*8])),#residual block has convs of kernel=3x3 and input channels = output channels
        'wi1': tf.Variable(random_normal([1, 1, nf*8, nf*8])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Transpose Convolutional Layer 2: 3x3 conv, nf*8 input, nf*4 output filters 
        'wu2': tf.Variable(random_normal([3, 3, nf*4, nf*8])),#upsample block has convs of kernel=3x3
        'wr2': tf.Variable(random_normal([3, 3, nf*4, nf*4])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi2': tf.Variable(random_normal([1, 1, nf*4, nf*4])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Transpose Convolutional Layer 3: 3x3 conv, nf*4 input, nf*2 output filters 
        'wu3': tf.Variable(random_normal([3, 3, nf*2, nf*4])),#upsample block has convs of kernel=3x3
        'wr3': tf.Variable(random_normal([3, 3, nf*2, nf*2])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi3': tf.Variable(random_normal([1, 1, nf*2, nf*2])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Transpose Convolutional Layer 4: 3x3 conv, nf*2 input, nf output filters 
        'wu4': tf.Variable(random_normal([3, 3, nf, nf*2])),#upsample block has convs of kernel=3x3
        'wr4': tf.Variable(random_normal([3, 3, nf, nf])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi4': tf.Variable(random_normal([1, 1, nf, nf])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Transpose Convolutional Layer 5: 4x4 conv, nf input, 1 output filters 
        'wu5': tf.Variable(random_normal([4, 4, 1, nf])),#upsample block has convs of kernel=3x3
        'wr5': tf.Variable(random_normal([3, 3, 1, 1])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi5': tf.Variable(random_normal([1, 1, 1, 1])),#identity map has a conv of kernel=1x1 and input channels = output channels
    }
    biases_g = {
        'fc1': tf.Variable(tf.zeros([nf*32])),
        'fc2': tf.Variable(tf.zeros([2*2*nf*16])),
    }
    return weights_g, biases_g

# Create some wrappers for simplicity.
def residual_block(x, W, strides=1):
    # conv2d wrapper, with bias and relu activation.
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, keepdims=False)
    x = tf.nn.relu(tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-9))
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, keepdims=False)
    x = tf.nn.relu(tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-9)) 
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')
    return x
def identity_map(x, W,strides=1):
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, keepdims=False)
    x = tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-9)
    return x
def upsample(x, W, padding=1, strides=2):
    output_shape = strides*(x.shape[1]-1)+W.shape[0]-2*padding
    return tf.nn.conv2d_transpose(input=x,  filters=W, output_shape=tf.convert_to_tensor([x.shape[0],output_shape,output_shape,W.shape[2]]), strides=[1, strides, strides, 1], padding='SAME')   


def residual_transpose_convolution_block(x, Wu, Wr, Wi, padding=0):
        x = upsample(x, Wu, padding)
        y = residual_block(x, Wr) + identity_map(x, Wi)
        return y
    
# Create Generator
def generator(x, weights_g, biases_g, nz=100, nf=32):
    # Input shape: [-1, nz]. 
    # Fully connected layer, Output shape: [-1, nf*32].
    out = tf.add(tf.matmul(x, weights_g['fc1']), biases_g['fc1'])
    # Fully connected layer, Output shape: [-1, 2*2*nf*16].
    out = tf.add(tf.matmul(out, weights_g['fc2']), biases_g['fc2'])
    
    out = tf.reshape(out, [-1, 2, 2, nf*16])
    
    # Residual Transpose Convolution Layer. Output shape: [-1, 4, 4, nf*8].
    out = residual_transpose_convolution_block(out, weights_g['wu1'], weights_g['wr1'],  weights_g['wi1'], padding=1)
    
    # Residual Transpose Convolution Layer. Output shape: [-1, 7, 7, nf*4].
    out = residual_transpose_convolution_block(out, weights_g['wu2'], weights_g['wr2'],  weights_g['wi2'], padding=1)
    
    # Residual Transpose Convolution Layer. Output shape: [-1, 13, 13, nf*2].
    out = residual_transpose_convolution_block(out, weights_g['wu3'], weights_g['wr3'],  weights_g['wi3'], padding=1)
    
    # Residual Transpose Convolution Layer. Output shape: [-1, 25, 25, nf].
    out = residual_transpose_convolution_block(out, weights_g['wu4'], weights_g['wr4'],  weights_g['wi4'], padding=1)
    
    # Residual Transpose Convolution Layer. Output shape: [-1, 50, 50, 1].
    out = residual_transpose_convolution_block(out, weights_g['wu5'], weights_g['wr5'],  weights_g['wi5'], padding=1)

    # tanh so the image pixels range between -1 and 1
    return tf.math.tanh(out)
     
def generator_loss(fake_output):
    mse = tf.compat.v1.losses.mean_squared_error(labels=tf.ones_like(fake_output), predictions=fake_output, weights=1.0, )
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output))
    loss = mse # cross_entropy 
    return -tf.reduce_mean(loss)