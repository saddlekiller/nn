import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.plotpy as plt

graph=tf.Graph()
with graph.as_default():
    
    placeholder_input=tf.placeholder(tf.float32,[None,32,32,3],'holder_input')
    placeholder_target=tf.placeholder(tf.float32,[None,10],'holder_target')
    
    conv1_layer=convolution_layer(inputs=placeholder_input,kernel_shape=[5,5,3,6],name='conv1_layer')
    conv1_out=conv1_layer.get_outputs(active=tf.nn.relu,padding='VALID')
    
    max1_layer=maxpooling_layer(inputs=conv1_out,name='max1_layer')
    max1_out=max1_layer.get_outputs(padding='VALID',strides=[1,2,2,1])
    
    conv2_layer=convolution_layer(inputs=max1_out,kernel_shape=[5,5,6,16],name='conv2_layer')
    conv2_out=conv2_layer.get_outputs(active=tf.nn.relu,padding='VALID')
    
    max2_layer=maxpooling_layer(inputs=conv2_out,name='max2_layer')
    max2_out=max2_layer.get_outputs(padding='VALID',strides=[1,2,2,1])
    
    conv3_layer=convolution_layer(inputs=max2_out,kernel_shape=[5,5,16,64],name='conv3_layer')
    conv3_out=conv3_layer.get_outputs(active=tf.nn.relu,padding='VALID')
    
    reshape1_layer=reshape_layer(inputs=conv3_out,name='reshape1_layer')
    reshape1_out=reshape1_layer.get_outputs([-1,64])
    
    affine1_layer=affine_layer(inputs=reshape1_out,weights_shape=[64,128],name='affine1_layer')
    affine1_out=affine1_layer.get_outputs(tf.nn.relu)
        
    affine2_layer=affine_layer(inputs=affine1_out,weights_shape=[128,256],name='affine2_layer')
    affine2_out=affine2_layer.get_outputs(tf.nn.relu)
        
    affine3_layer=affine_layer(inputs=affine2_out,weights_shape=[256,10],name='affine3_layer')
    affine3_out=affine3_layer.get_outputs(tf.nn.softmax)
    
    print('################ STRUCTURE ################')
    print(conv1_out)
    print(max1_out)
    print(conv2_out)
    print(max2_out)
    print(conv3_out)
    print(reshape1_out)
    print(affine1_out)
    print(affine2_out)
    print(affine3_out)
    print('###########################################')
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=affine3_out,labels=placeholder_target))
    train_step=tf.train.AdamOptimizer().minimize(loss)
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(affine3_out,1),tf.argmax(placeholder_target,1)),tf.float32))
    iteration=200
    interval=10
    storage={'index':[],'acc':[],'error':[],'conv1':[],'conv2':[],'conv3':[],'kernel1':[],'kernel2':[],'kernel3':[]}
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):
            errors=0.
            accs=0.
            index=0
            for input_batch,target_batch in zip(train_data['inputs'],train_data['targets']):
#                 if index==2:
#                     break
                feed={placeholder_input:input_batch,placeholder_target:target_batch}
                _,e,a=sess.run([train_step,loss,acc],feed_dict=feed)
                errors+=e
                accs+=a
                index+=1
            errors/=index
            accs/=index
            if i%interval==0:
                print('epoch # {0:02d}: training error = {1:.4f} training accuracy = {2:.4f}'.format(i+1,errors,accs))
                c1,c2,c3,k1,k2,k3=sess.run([conv1_layer.outputs,conv2_layer.outputs,conv3_layer.outputs,
                                            conv1_layer.kernels,conv2_layer.kernels,conv3_layer.kernels
                                           ],feed_dict={placeholder_input:train_data['inputs'][0],
                                                        placeholder_target:train_data['targets'][0]})

                storage['index'].append(i)
                storage['acc'].append(a)
                storage['error'].append(e)
                storage['conv1'].append(c1)
                storage['conv2'].append(c2)
                storage['conv3'].append(c3)
                storage['kernel1'].append(k1)
                storage['kernel2'].append(k2)
                storage['kernel3'].append(k3)
                valid_err=0.
                valid_acc=0.
                valid_index=0
                for input_batch,target_batch in zip(valid_data['inputs'],valid_data['targets']):
#                     if valid_index==2:
#                         break
                    feed={placeholder_input:input_batch,placeholder_target:target_batch}
                    e,a=sess.run([loss,acc],feed_dict=feed)
                    valid_err+=e
                    valid_acc+=a
                    valid_index+=1
                valid_err/=valid_index
                valid_acc/=valid_index
                print('--------------------------------------------------------------------------------------------')
                print('|epoch # {0:02d}: validation error = {1:.4f} validation accuracy = {2:.4f}|'.format(i+1,valid_err,valid_acc))
                print('--------------------------------------------------------------------------------------------')
    np.save('results',storage)

    
