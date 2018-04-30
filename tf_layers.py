import tensorflow as tf

# TODO: conv3d, max_pool3d, avg_pool3d, dropout

def _variable_on_cpu(name, shape, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable on CPU
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple): shape of the variable
            `initializer` (tf function): initializer of the variable
            `id` (int): CPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
    """
    with tf.device('/cpu:{}'.format(id)):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_on_gpu(name, shape, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable on GPU
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple of int): shape of the variable
            `initializer` (tf function): initializer of the variable
            `id` (int): GPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
        Notes:
            - possible `initializer` may be:
                tf.constant_initializer(value)
                tf.contrib.layers.xavier_initializer()
                tf.truncated_normal_initializer(stddev=stddev)
    """
    with tf.device('/gpu:{}'.format(id)):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

_get_variable = _variable_on_gpu

def _variable_with_weight_decay(name, shape, wd, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable with weight decay on GPU 
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple of int): shape of the variable
            `wd` (float): coefficient of weight decay
            `initializer` (tf function instance): initializer of the variable
            `id` (int): GPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
        Notes:
            - weight decay loss is added to collection 'losses' with name 'weight_loss'
    """
    var = _get_variable(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def batch_norm(scope, inputs, is_training, bn_decay, data_format='NCHW'):
    """ tensorflow layer: Batch normalization
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): input tensor
            `is_training` (bool): training or testing
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
    """
    outputs = tf.contrib.layers.batch_norm(inputs, center=True, scale=True,
                                           is_training=is_training, decay=bn_decay,
                                           updates_collections=None, scope=scope, 
                                           data_format=data_format)
    return outputs

def conv1d(scope, inputs, num_output_channels, kernel_size, stride=1, padding='SAME',
           data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
           activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 1D convolution with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 3D input tensor of shape BxLxC or BxCxL
            `num_output_channels` (int): number of output channels
            `kernel_size` (int): kernel size of 1D convolution
            `stride` (int): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_shape = [kernel_size, C, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 1D convolution
        outputs = tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def conv2d(scope, inputs, num_output_channels, kernel_size, stride=[1,1], padding='SAME',
           data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
           activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 2D convolution with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
    """
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, C, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 2D convolution
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel, strides=[1, stride_h, stride_w, 1], padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def conv2d_transpose(scope, inputs, num_output_channels, kernel_size, stride=[1,1], padding='SAME',
                     data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
                     activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 2D convolution transpose with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    # from slim.convolution2d_transpose
    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
        dim_size *= stride_size
        if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
        return dim_size
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, num_output_channels, C] # in/out channel reverse comparing to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 2D convolution transpose
        stride_h, stride_w = stride
        outH = get_deconv_dim(H, stride_h, kernel_h, padding)
        outW = get_deconv_dim(W, stride_w, kernel_w, padding)
        if data_format == 'NHWC':
            output_shape = [B, outH, outW, num_output_channels]
        else:
            output_shape = [B, num_output_channels, outH, outW]
        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         strides=[1, stride_h, stride_w, 1], 
                                         padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def fully_connected(scope, inputs, num_outputs, initializer=tf.contrib.layers.xavier_initializer(), 
                    weight_decay=None, activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: fully connected with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 2D input tensor of shape BxHxWxC or BxN
            `num_outputs` (int): number of output dimensions
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
    """
    with tf.variable_scope(scope):
        # specify weights and biases
        B, N = inputs.get_shape().as_list()
        weight = _variable_with_weight_decay('weights',
                                             shape=[N, num_outputs],
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        # perform fully connected layer
        outputs = tf.nn.bias_add(tf.matmul(inputs, weight), biases)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def max_pool2d(scope, inputs, kernel_size, stride=[2,2], padding='VALID', data_format='NHWC'):
    """ tensorflow layer: 2D max pooling
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
    """
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name='max-pool2d',
                                 data_format=data_format)
    return outputs

def avg_pool2d(scope, inputs, kernel_size, stride=[2,2], padding='VALID', data_format='NHWC'):
    """ tensorflow layer: 2D average pooling
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name='avg-pool2d',
                                 data_format=data_format)
    return outputs

def get_lr_expdecay(step, base_lr, decay_steps, decay_rate, end_lr, staircase=False):
    """ get learning rate decay tf operation
        Args:
            `step` (tensor): global step
            `base_lr` (float): base learning rate
            `decay_steps` (int): decay steps
            `decay_rate` (float): decay rate
            `end_lr` (float): end learning rate
            `staircase` (bool): params of exponential decay
        Returns:
            `lr` (tensor): learning rate op
        Notes:
            1. lr = max(base_lr * decay_rate^(step / decay_steps), end_lr)
    """
    lr = tf.train.exponential_decay(
            base_lr,
            step,
            decay_steps,
            decay_rate,
            staircase=staircase)
    lr = tf.maximum(lr, end_lr) # clipping to end learning rate
    return lr
