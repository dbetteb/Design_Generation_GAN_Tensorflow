import tensorflow as tf
# Store layers weight & bias

def initialize_discriminator_weights(nf = 32, nc=1):
    # A random value generator to initialize weights.
    random_normal = tf.initializers.RandomNormal()
    # Network parameters.
    # nf = number of filters for 1st residual convolutional layer.

    weights_d = {
        # Residual Convolutional Layer 1: 4x4 conv, nc input, nf filters (Designs have nc channels; depends on the input: design only --> nc=1 or design with constraints --> nc>1).
        'wd1': tf.Variable(random_normal([4, 4, nc, nf])),#downsample block has a conv of kernel=4x4 
        'wr1': tf.Variable(random_normal([3, 3, nf, nf])),#residual block has convs of kernel=3x3 and input channels = output channels
        'wi1': tf.Variable(random_normal([1, 1, nf, nf])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 2: 3x3 conv, nf input, nf*2 filters 
        'wd2': tf.Variable(random_normal([3, 3, nf, nf*2])),#downsample block has convs of kernel=3x3
        'wr2': tf.Variable(random_normal([3, 3, nf*2, nf*2])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi2': tf.Variable(random_normal([1, 1, nf*2, nf*2])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 3: 3x3 conv, nf*2 input, nf*4 filters 
        'wd3': tf.Variable(random_normal([3, 3, nf*2, nf*4])),#downsample block has convs of kernel=3x3
        'wr3': tf.Variable(random_normal([3, 3, nf*4, nf*4])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi3': tf.Variable(random_normal([1, 1, nf*4, nf*4])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 4: 3x3 conv, nf*4 input, nf*8 filters 
        'wd4': tf.Variable(random_normal([3, 3, nf*4, nf*8])),#downsample block has convs of kernel=3x3
        'wr4': tf.Variable(random_normal([3, 3, nf*8, nf*8])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi4': tf.Variable(random_normal([1, 1, nf*8, nf*8])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 5: 3x3 conv, nf*8 input, nf*16 filters 
        'wd5': tf.Variable(random_normal([3, 3, nf*8, nf*16])),#downsample block has convs of kernel=3x3
        'wr5': tf.Variable(random_normal([3, 3, nf*16, nf*16])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi5': tf.Variable(random_normal([1, 1, nf*16, nf*16])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 6: 4x4 conv, nf*16 input, nf*32 filters 
        'wd6': tf.Variable(random_normal([4, 4, nf*16, nf*32])),#downsample block has convs of kernel=4x4
        'wr6': tf.Variable(random_normal([3, 3, nf*32, nf*32])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi6': tf.Variable(random_normal([1, 1, nf*32, nf*32])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # Residual Convolutional Layer 7: 4x4 conv, nf*32 input, nf*64 filters 
        'wd7': tf.Variable(random_normal([4, 4, nf*32, nf*64])),#downsample block has convs of kernel=4x4
        'wr7': tf.Variable(random_normal([3, 3, nf*64, nf*64])),#residual block has a conv of kernel=3x3 and input channels = output channels
        'wi7': tf.Variable(random_normal([1, 1, nf*64, nf*64])),#identity map has a conv of kernel=1x1 and input channels = output channels

        # FC Out Layer: nf*64 inputs, 1 unit (probability of a design being fake or real)
        # fc1_units = nf*64 # number of neurons for 1st fully-connected layer (last layer of the discriminator).

        'fc': tf.Variable(random_normal([nf*64, 1]))
    }
    biases_d = {
        'fc': tf.Variable(tf.zeros([1])),
    }
    return weights_d, biases_d

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
#     gamma = tf.sqrt(variance)
    x = tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-9)
    return x
def downsample(x, W, strides=2):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')    

def residual_convolution_block(x, Wd, Wr, Wi):
        x = downsample(x, Wd)
        y = residual_block(x, Wr) + identity_map(x, Wi)
        return y
    
# Create Discriminator
def discriminator(x, weights_d, biases_d, nc=1):
    # Input shape: [-1, 100, 100, nc]. A batch of 101x101xnc images. designs + constraints--> multi-channel image--> nc>1, only designs--> nc=1
    x = tf.reshape(x, [-1, 100, 100, nc])
    # Residual Convolution Layer. Output shape: [-1, 50, 50, nf].
    out = residual_convolution_block(x, weights_d['wd1'], weights_d['wr1'],  weights_d['wi1'])
    
    # Residual Convolution Layer. Output shape: [-1, 25, 25, nf*2].
    out = residual_convolution_block(out, weights_d['wd2'], weights_d['wr2'],  weights_d['wi2'])
    
    # Residual Convolution Layer. Output shape: [-1, 13, 13, nf*4].
    out = residual_convolution_block(out, weights_d['wd3'], weights_d['wr3'],  weights_d['wi3'])
    
    # Residual Convolution Layer. Output shape: [-1, 7, 7, nf*8].
    out = residual_convolution_block(out, weights_d['wd4'], weights_d['wr4'],  weights_d['wi4'])
    
    # Residual Convolution Layer. Output shape: [-1, 4, 4, nf*16].
    out = residual_convolution_block(out, weights_d['wd5'], weights_d['wr5'],  weights_d['wi5'])
    
    # Residual Convolution Layer. Output shape: [-1, 2, 2, nf*32].
    out = residual_convolution_block(out, weights_d['wd6'], weights_d['wr6'],  weights_d['wi6'])
    
    # Residual Convolution Layer. Output shape: [-1, 1, 1, nf*64].
    out = residual_convolution_block(out, weights_d['wd7'], weights_d['wr7'],  weights_d['wi7'])
    

    # Reshape output to fit fully connected layer input, Output shape: [-1, nf*64].
    out = tf.reshape(out, [-1, weights_d['fc'].get_shape().as_list()[0]])
    out = tf.nn.dropout(out, rate=0.35)
    # Fully connected layer, Output shape: [-1, 1].
    out = tf.add(tf.matmul(out, weights_d['fc']), biases_d['fc'])
    # Sigmoid because the value of probability is always between 0 and 1
    return tf.sigmoid(out)
     
def discriminator_loss(real_output, fake_output):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    total_loss = tf.reduce_sum(real_loss+fake_loss)
    return total_loss