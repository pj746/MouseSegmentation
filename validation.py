import sys, getopt, os, re, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./MiceSegmentation/')
# import inspect
# import resource
import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import resnet_v2
# from tensorflow.contrib.slim.nets import resnet_utils
# from tensorflow.contrib.slim.nets import inception
# from tensorflow.contrib.slim.nets import vgg
# from Network.readers import means, scale, atan2
# import scipy.ndimage.morphology as morph
import numpy as np
import cv2
# import time
# from datetime import datetime
import Network.datasets as datasets
from Network.transformer import *
import matplotlib.pyplot as plt

# Setting the default training parameters
# training set size:250 validation set size:47 'const_learn_rate': whether decay the learning rate
path = r'./MiceSegmentation/'
arg_dict = {'batch_size': 1,'network_to_restore':path+r'data/model.ckpt-456000',
			'log_dir': path+r'data/',	'train_list': path+r'data/training/xaa', 'valid_list': path+r'data/training/xab',
			'dataset_folder': path+r'data/', 'input_size': 480}

#######################
#Network
#######################
# Segmentation Only Network (no angle prediction)
def construct_segsoft_v5(images, is_training):
	batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	# Normalize the image inputs (map_fn used to do a "per batch" calculation)
	norm_imgs = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
	kern_size = [5,5]
	filter_size = 8
	# Run the segmentation net without pooling
	with tf.variable_scope('SegmentEncoder'):
		with slim.arg_scope([slim.conv2d],
							activation_fn=tf.nn.relu,
							padding='SAME',
							weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
							weights_regularizer=slim.l2_regularizer(0.0005),
							normalizer_fn=slim.batch_norm,
							normalizer_params=batch_norm_params):
			c1 = slim.conv2d(norm_imgs, filter_size, kern_size)
			p1 = slim.max_pool2d(c1, [2,2], scope='pool1') #240x240
			c2 = slim.conv2d(p1, filter_size*2, kern_size)
			p2 = slim.max_pool2d(c2, [2,2], scope='pool2') #120x120
			c3 = slim.conv2d(p2, filter_size*4, kern_size)
			p3 = slim.max_pool2d(c3, [2,2], scope='pool3') #60x60
			c4 = slim.conv2d(p3, filter_size*8, kern_size)
			p4 = slim.max_pool2d(c4, [2,2], scope='pool4') # 30x30
			c5 = slim.conv2d(p4, filter_size*16, kern_size)
			p5 = slim.max_pool2d(c5, [2,2], scope='pool5') # 15x15
			c6 = slim.conv2d(p5, filter_size*32, kern_size)
			p6 = slim.max_pool2d(c6, [3,3], stride=3, scope='pool6') # 5x5
			c7 = slim.conv2d(p6, filter_size*64, kern_size)
	with tf.variable_scope('SegmentDecoder'):
		upscale = 2 # Undo the pools once at a time
		mynet = slim.conv2d_transpose(c7, filter_size*32, kern_size, stride=[3, 3], activation_fn=None)
		mynet = tf.add(mynet, c6)
		mynet = slim.conv2d_transpose(mynet, filter_size*16, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c5)
		mynet = slim.conv2d_transpose(mynet, filter_size*8, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c4)
		mynet = slim.conv2d_transpose(mynet, filter_size*4, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c3)
		mynet = slim.conv2d_transpose(mynet, filter_size*2, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c2)
		mynet = slim.conv2d_transpose(mynet, filter_size, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c1)
		seg = slim.conv2d(mynet, 2, [1,1], scope='seg')
	return seg

def read_image(filename, input_size):
	image_contents = tf.read_file(filename)
	image = tf.image.decode_png(image_contents, channels=1)
	image = tf.image.resize_images(image, [input_size, input_size])
	return image

def get_validation_img(dataset, input_size, valid_num=10, show=False, save=True): # This one is used after training finished and starting to save the output image.
	#read the validation image
	inputs = dataset.valid_images
	inputs2 = dataset.valid_seg
	input_queue = tf.train.slice_input_producer([inputs,inputs2], shuffle=False)
	example_list = [read_image(input_queue[0], input_size),read_image(input_queue[1], input_size)]
	
	sess = tf.Session()
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch = construct_segsoft_v5(image_placeholder, is_training) # Output of the network. Shape:(N,480,480,2)
		seg_eval_batch = tf.nn.softmax(network_eval_batch)[:,:,:,0] # Only grab the "Mouse"
	with tf.variable_scope('Input_Decoding'):
		image_batch, label_batch = tf.train.batch(example_list,batch_size = 1)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])
	##########################################
	#Add one more parameter 'Show_or_save' for this function. Store the following code as another function later: if using server: save, otherwise: view
	if valid_num != 0:
		for i in range(valid_num):
			input_batch, expect_batch = sess.run([image_batch,label_batch])
			output = sess.run(seg_eval_batch, feed_dict={image_placeholder: input_batch, is_training: False})
			input_name = dataset.valid_list[i]
			input_img = np.squeeze(input_batch) # Wipe out redundant axis
			expect_img = np.squeeze(expect_batch)
			output_img = np.squeeze(output)
			#Compare the images.
			if show:
				plt.figure()
				plt.title(input_name)
				plt.subplot(1,3,1)
				plt.cla()
				frame = plt.gca()
				frame.axes.get_yaxis().set_visible(False)
				frame.axes.get_xaxis().set_visible(False)
				plt.imshow(input_img)

				plt.subplot(1,3,2)
				plt.title('expected output')  #image of label
				plt.cla()
				frame = plt.gca()
				frame.axes.get_yaxis().set_visible(False)
				frame.axes.get_xaxis().set_visible(False)
				plt.imshow(expect_img,cmap=plt.cm.gray)

				plt.subplot(1,3,3)
				plt.title('output') #image of network's output
				plt.cla()
				frame = plt.gca()
				frame.axes.get_yaxis().set_visible(False)
				frame.axes.get_xaxis().set_visible(False)
				plt.imshow(output_img,cmap=plt.cm.gray)
				plt.pause(0.3)
				plt.show()
			# Save the outputs.
			if save:
				input_name = dataset.valid_list[i]
				cv2.imwrite(arg_dict['log_dir']+r'/Result/input'+input_name, input_img)
				cv2.imwrite(arg_dict['log_dir']+r'/Result/label'+input_name, expect_img)
				cv2.imwrite(arg_dict['log_dir']+r'/Result/output'+input_name, output_img*255)

# Prep the dataset
if 'dataset_folder' in arg_dict.keys() and arg_dict['dataset_folder'] is not None:
	arg_dict['dataset'] = datasets.TrackingDataset(arg_dict['train_list'], arg_dict['valid_list'], arg_dict['dataset_folder'])
elif 'train_list' in arg_dict.keys():
	arg_dict['dataset'] = datasets.TrackingDataset(arg_dict['train_list'], arg_dict['valid_list'], '.')

# Start training.

with tf.device('/cpu:0'):
	# trainSegSoftNetwork(arg_dict)
	get_validation_img(arg_dict['dataset'], arg_dict['input_size'],show=True, valid_num=10, save=True)