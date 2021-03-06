import tensorflow as tf
# Store layers weight & bias

def initialize_discriminator_weights(nf = 32, nc=1):
    # A random value generator to initialize weights.
    random_normal = tf.initializers.RandomNormal()
    # Network parameters.
    # nf = number of filters for 1st residual convolutional layer.

    weights_d = {
#         
        # Residual Convolutional Layer 2: 3x3 conv, nc input, nf filters 
        'wd2': tf.Variable(random_normal([3, 3, nc, nf])),#downsample block has convs of kernel=3x3
        
        # Residual Convolutional Layer 3: 3x3 conv, nf input, nf*2 filters 
        'wd3': tf.Variable(random_normal([3, 3, nf, nf*2])),#downsample block has convs of kernel=3x3
        
        # Residual Convolutional Layer 4: 3x3 conv, nf*2 input, nf*4 filters 
        'wd4': tf.Variable(random_normal([3, 3, nf*2, nf*4])),#downsample block has convs of kernel=3x3
        
        # Residual Convolutional Layer 5: 3x3 conv, nf*4 input, nf*8 filters 
        'wd5': tf.Variable(random_normal([3, 3, nf*4, nf*8])),#downsample block has convs of kernel=3x3
        
        # Residual Convolutional Layer 6: 4x4 conv, nf*8 input, nf*16 filters 
        'wd6': tf.Variable(random_normal([4, 4, nf*8, nf*16])),#downsample block has convs of kernel=4x4
        
        # Residual Convolutional Layer 7: 4x4 conv, nf*16 input, nf*32 filters 
        'wd7': tf.Variable(random_normal([4, 4, nf*16, nf*32])),#downsample block has convs of kernel=4x4
       
        # FC Out Layer: nf*64 inputs, 1 unit (probability of a design being fake or real)
        # fc1_units = nf*64 # number of neurons for 1st fully-connected layer (last layer of the discriminator).

        'fc': tf.Variable(random_normal([nf*32, 1]))
    }
    biases_d = {
        'fc': tf.Variable(tf.zeros([1])),
    }
    return weights_d, biases_d

def downsample_2(x, W, strides=2):
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')    
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, keepdims=False)
    x = tf.nn.relu(tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-9))
    return x

# Create Discriminator
def discriminator(x, weights_d, biases_d, nc=1):
    # Input shape: [-1, 50, 50, nc]. A batch of 101x101xnc images. designs + constraints--> multi-channel image--> nc>1, only designs--> nc=1
    x = tf.reshape(x, [-1, 50, 50, nc])
    
    # Residual Convolution Layer. Output shape: [-1, 25, 25, nf].
    out = downsample_2(x, weights_d['wd2'])
    
    # Residual Convolution Layer. Output shape: [-1, 13, 13, nf*2].
    out = downsample_2(out, weights_d['wd3'])
    
    # Residual Convolution Layer. Output shape: [-1, 7, 7, nf*4].
    out = downsample_2(out, weights_d['wd4'])
    
    # Residual Convolution Layer. Output shape: [-1, 4, 4, nf*8].
    out = downsample_2(out, weights_d['wd5'])
    
    # Residual Convolution Layer. Output shape: [-1, 2, 2, nf*16].
    out = downsample_2(out, weights_d['wd6'])
    
    # Residual Convolution Layer. Output shape: [-1, 1, 1, nf*32].
    out = downsample_2(out, weights_d['wd7'])
    

    # Reshape output to fit fully connected layer input, Output shape: [-1, nf*32].
    out = tf.reshape(out, [-1, weights_d['fc'].get_shape().as_list()[0]])
    out = tf.nn.dropout(out, rate=0.35)
    # Fully connected layer, Output shape: [-1, 1].
    out = tf.add(tf.matmul(out, weights_d['fc']), biases_d['fc'])
    # Sigmoid because the value of probability is always between 0 and 1
    return tf.sigmoid(out)

def discriminator_loss(real_output, fake_output):
    #real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    real_loss = tf.compat.v1.losses.mean_squared_error(predictions=real_output, labels=tf.ones_like(real_output))
 
    #fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    fake_loss = tf.compat.v1.losses.mean_squared_error(predictions=fake_output, labels=tf.zeros_like(fake_output))
    
    total_loss = tf.reduce_mean(real_loss) - tf.reduce_mean(fake_loss) #real_loss + fake_loss
    return total_loss