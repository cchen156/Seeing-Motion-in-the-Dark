import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers import layer_norm
import pdb

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv_filter = tf.cast(deconv_filter,x1.dtype)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def Unet(input, dims=32, demosaic=False):
    conv1=slim.conv2d(input,dims,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm, scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv5_2')

    net = conv5
    for i in range(16):
        temp = net
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu,normalizer_fn=layer_norm, scope='g_res%d_conv1'%i)
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_res%d_conv2'%i)
        net = net + temp  
      
    net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_res')
    conv5 = net + conv5

    up6 =  upsample_and_concat( conv5, conv4, dims*8, dims*16  )
    conv6=slim.conv2d(up6,  dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, dims*4, dims*8  )
    conv7=slim.conv2d(up7,  dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, dims*2, dims*4 )
    conv8=slim.conv2d(up8,  dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, dims, dims*2 )
    conv9=slim.conv2d(up9,  dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv9_2')
    
    if demosaic:
        conv10 = slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        conv10 = tf.depth_to_space(conv10,2) 
    else:
        conv10 = slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    
    return conv10

