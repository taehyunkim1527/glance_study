import tensorflow as tf
import pickle
import numpy as np
import cifar
import model
import time
import math
import sys
import os
from datetime import datetime

########################### FLAGS ######################################################

tf.app.flags.DEFINE_string('model_type', '', 'Model type. Supported : res, wrn')
tf.app.flags.DEFINE_string('data', '', 'Name of dataset. Supported: cifar-10, cifar-100')
tf.app.flags.DEFINE_integer('num_layer', 0, 'The number of layers the model uses')
tf.app.flags.DEFINE_integer('wfactor', 1, 'Resnet width factor (for wrn)')
tf.app.flags.DEFINE_string('name', '', 'id of the model')
tf.app.flags.DEFINE_integer('gpu', 0, 'Which GPU to use')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch')
tf.app.flags.DEFINE_boolean('test', '', 'Run test if True else run train')

FLAGS = tf.app.flags.FLAGS

#set GPU to use
os.environ["CUDA_VISIBLE_DEVICE"] = str(FLAGS.gpu)

########################### MAIN ########################################################

def main(argv = None):

	model_name = generate_name()
	print('Model Name: %s' %model_name)

### log directory
	log_dir = '/tmp/log_ensemble'

### prepare for checkpoint
	ckpt_dir = './ckpt'
	ckpt_path = ckpt_dir + '/' + model_name + '.ckpt'
	meta_path = ckpt_dir + '/' + model_name + '.meta'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

### download dataset
	data_dir = './data'
	cifar.maybe_download_and_extract(data_dir, FLAGS.data)

### build model
	model.build(FLAGS.model_type, data_dir, FLAGS.data)

### create a local session to run training
	config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.45
	with tf.Session(config=config) as sess:
### log the graph data
		writer = tf.train.SummaryWriter(log_dir, sess.graph)  ### Only for graph. if i wanna summary some data, have to use the func 'writer.add_summary)
### ckpt saver
		saver = tf.train.Saver(max_to_keep=30)
### start the queue runners
		coord = tf.train.Coordinator
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
