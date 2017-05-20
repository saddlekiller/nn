#!/usr/bin/env python

# coding=utf-8
class param(object):
    
    def weights(name,shape,initialize_mode=tf.random_normal_initializer,
                reg=tf.contrib.layers.l2_regularizer,reg_const=0.0):
        weights=tf.get_variable(
            name=name,
            shape=shape,
            initializer=initialize_mode(stddev=1./float(shape[-1]**0.5)),
            regularizer=reg(reg_const)
        )
        return weights
        
    def biases(name,shape,regularisation_const=0.0,
                reg=tf.contrib.layers.l2_regularizer,reg_const=0.0):
        biase=tf.get_variable(
            name=name,
            initializer=tf.zeros(shape),
            regularizer=reg(reg_const)
        )
        return biase
    
    def kernels(name,shape,initialize_mode=tf.random_normal_initializer,
                reg=tf.contrib.layers.l2_regularizer,reg_const=0.0):
        kernels=tf.get_variable(
            name=name,
            shape=shape,
            initializer=initialize_mode(stddev=1./float(shape[-1]**0.5)),
            regularizer=reg(reg_const)
        )
        return kernels

class layer(object):
    
    def __init__(self,name,inputs):
        raise NotImplementedError
    
    def initialize(self):
        raise NotImplementedError
    
    def set_weights(self,weights,biases):
        raise NotImplementedError
        
    def get_outputs(self):
        raise NotImplementedError
        
    def get_info(self):
        print(self.name)
        
class affine_layer(layer):
    
    def __init__(self,name,inputs,weights_shape=[]):
        self.name=name
        self.inputs=inputs
        self.initialize(weights_shape)
    
    def initialize(self,weights_shape):
        self.weights=param.weights(name=self.name+'_w',shape=weights_shape)
        self.biases=param.biases(name=self.name+'_b',shape=weights_shape[-1])
        
    def set_weights(self,weights,biases):
        self.weights=weights
        self.biases=biases
        
    def get_outputs(self,active=tf.nn.sigmoid):
        self.outputs=active(tf.matmul(self.inputs,self.weights)+self.biases)
        return self.outputs
    
class convolution_layer(layer):
    
    def __init__(self,name,inputs,kernel_shape=[]):
        self.name=name
        self.inputs=inputs
        self.initialize(kernel_shape)
        
    def initialize(self,kernel_shape):
        self.kernels=param.kernels(name=self.name+'_k',shape=kernel_shape)
        self.biases=param.biases(name=self.name+'_b',shape=kernel_shape[-1])
    
    def set_weights(self,kernels,biases):
        self.kernels=kernels
        self.biases=biases
        
    def get_outputs(self,active=tf.nn.sigmoid,padding='VALID',strides=[1,1,1,1]):
        self.outputs=active(tf.nn.conv2d(self.inputs,self.kernels,padding=padding,strides=strides)+self.biases)
        return self.outputs
    
class maxpooling_layer(layer):
    
    def __init__(self,name,inputs):
        self.name=name
        self.inputs=inputs
        
    def initialize(self):
        raise NotImplementedError
    
    def set_weights(self):
        raise NotImplementedError
        
    def get_outputs(self,active=tf.identity,padding='VALID',strides=[1,1,1,1],ksize=[1,2,2,1]):
        self.outputs=active(tf.nn.max_pool(self.inputs,padding=padding,strides=strides,ksize=ksize))
        return self.outputs

class reshape_layer(layer):
    
    def __init__(self,name,inputs):
        self.name=name
        self.inputs=inputs
        
    def initialize(self):
        raise NotImplementedError
    
    def set_weights(self):
        raise NotImplementedError
        
    def get_outputs(self,shape):
        self.outputs=tf.reshape(self.inputs,shape)
        return self.outputs

