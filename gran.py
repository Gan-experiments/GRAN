import tensorflow as tf
import numpy as np
import random
import datetime
from utils import*
import os 

batch_size=64
data_shape=32
class batchnorm():
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)
def lrelu(x, leak=0.2):
	return tf.maximum(x,x*leak)

def conv(x,num_filters,kernel=5,stride=[1,2,2,1],name="conv",padding='SAME'):
	with tf.variable_scope(name):
		w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))

		b=tf.get_variable('b',shape=[num_filters],
		    initializer=tf.constant_initializer(0.0))
		con=tf.nn.conv2d(x, w, strides=stride, padding=padding)
		return tf.reshape(tf.nn.bias_add(con, b),con.shape)

def fcn(x,num_neurons,name="fcn"):#(without batchnorm )
	with tf.variable_scope(name):

		w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))

		b=tf.get_variable('b',shape=[num_neurons],
		    initializer=tf.constant_initializer(0.0))
		return tf.matmul(x,w)+b

def deconv(x,output_shape,kernel=5,stride=[1,2,2,1],name="deconv"):
	with tf.variable_scope(name):
		num_filters=output_shape[-1]
		w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))
		b=tf.get_variable('b',shape=[num_filters],
		    initializer=tf.constant_initializer(0.0))
		decon=tf.nn.conv2d_transpose(x, w, strides=stride,output_shape=output_shape)
		return tf.reshape(tf.nn.bias_add(decon, b),decon.shape)

def load_mnist():
	data_dir = os.path.join("./data-1", "mnist")

	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(np.float)

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i,y[i]] = 1.0

	return (X/255.),y_vec

def decoder(x,h_ct):
	with tf.variable_scope("forward"):
		fbn0		=	batchnorm(name="g_f_bn0")
		H0  		=	tf.nn.tanh(fbn0(fcn(x,784,"g_H0")))
		#print "hct shape ------------------------",h_ct.get_shape().as_list()
		R0		=	tf.concat((H0,h_ct),axis=1)
		fbn1		=	batchnorm(name="g_f_bn1")
		H1		=	tf.nn.relu(fbn1(fcn(R0,32*7*7*8,"g_H1")))
		H1		=	tf.reshape(H1,[batch_size,7,7,32*8])
		H2		=	tf.nn.relu(deconv(H1,[batch_size,14,14,32*4],kernel=5,name="g_H2"))
		H3		=	deconv(H2,[batch_size,28,28,1],kernel=5,name="g_H3")
		return H3

def encoder(ct):
	with tf.variable_scope("backward"):
		bbn0		=	batchnorm(name="g_b_bn0")
		H0		=	tf.nn.relu(bbn0(conv(ct,32*8,name="g_b_H0")))
		bbn1		=	batchnorm(name="g_b_bn1")
		H1		=	tf.nn.relu(bbn1(conv(ct,32*4,name="g_b_H1")))
		H1		=	tf.reshape(H1,[batch_size,-1])
		#bbn2		=	batchnorm(name="g_b_bn2")
		return tf.nn.tanh(fcn(H1,28*28,name="g_b_H2"))

def generator(z):
	h_ct            =	tf.constant(0.0,dtype=tf.float32,shape=[batch_size,784])
#	h_ct		=	h_in
        zt	=	tf.nn.tanh(fcn(z,h_ct.get_shape().as_list()[-1],name="g_zt"))
	with tf.variable_scope("generator") as scope:
		ct=tf.zeros([batch_size,28,28,1])
		for i in range(5):
			#print "shape *****************************************",h_ct.get_shape().as_list()
			
			if i>=1:
				scope.reuse_variables()
                        
			del_ct 	=	decoder(zt,h_ct)
			ct	=	ct+del_ct
			h_ct	=	encoder(del_ct)
		return tf.nn.sigmoid(tf.reshape(ct,[batch_size,28,28,1]))


def discriminator(x,reuse=False):
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()
		conx		=	tf.reshape(x,[batch_size,28,28,1])
		dbn0		=	batchnorm(name="d_bn0")
		H0		=	lrelu(dbn0(conv(conx,8*64,name="d_h0_conv")))
		dbn1		=	batchnorm(name="d_bn1")
		H1		=	lrelu(dbn1(conv(H0,4*64,name="d_h1_conv")))
		H1 		=	tf.reshape(H1,[batch_size,-1])
		H2 		=	lrelu(fcn(H1,1024,name="d_h2_dense"))
		H3		=	fcn(H2,1,name="d_h3_dense")
		return H3

with tf.device('/gpu:0'):
	k=0
	img  		=   tf.placeholder(tf.float32, [batch_size, 28,28,1], name='images')
#	h_in		=   tf.placeholder(tf.float32, [batch_size, 28*28], name='h_in')
	y      		=   tf.placeholder(tf.float32, [batch_size, 10], name='labels')
	X,Y 		= 	load_mnist()
	z      		=   tf.placeholder(tf.float32, [batch_size, 100], name='latent')
	fake_img 	=	generator(z)
	real_source	=	discriminator(img)
	fake_source 	=	discriminator(fake_img,reuse=True)
	d_loss          =	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_source,labels= tf.ones_like(real_source)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels= tf.zeros_like(fake_source)))

	g_loss          =	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels= tf.ones_like(fake_source)))


	#d_loss 		=	 0.5*tf.reduce_mean(tf.square(real_source-1))+ 0.5*tf.reduce_mean(tf.square(fake_source))
	#g_loss		=	tf.reduce_mean(tf.square(fake_source-1))
	t_vars		= 	tf.trainable_variables()
	d_vars      =   [var for var in t_vars if 'd_' in var.name]
	g_vars      =   [var for var in t_vars if 'g_' in var.name]
#	for i in range(len(g_vars)):
#		print g_vars[i].name
	d_opt       =   tf.train.AdamOptimizer(learning_rate=0.0002).minimize(d_loss,var_list=d_vars)
	g_opt       =   tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss,var_list=g_vars)
	init   		= 	tf.global_variables_initializer()
	config 		= 	tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	with tf.Session(config=config) as sess:
		sess.run(init)
		for epoch in range(1000):
		#	h_ct_in	= np.zeros(shape=(batch_size,784))
			for batch_i in range(X.shape[0]//batch_size):
				batch_x = X[(batch_i)*batch_size :(batch_i+1)*batch_size ]
				batch_y = Y[(batch_i)*batch_size :(batch_i+1)*batch_size ]
				z_=np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)
				for n in range(2):
					sess.run(d_opt,feed_dict={z:z_,img:batch_x.reshape([batch_size,28,28,1]),y:batch_y})
				sess.run(g_opt,feed_dict={z:z_,y:batch_y})
				D_loss = sess.run(d_loss,feed_dict={z:z_,img:batch_x.reshape([batch_size,28,28,1]),y:batch_y})
				G_loss = sess.run(g_loss,feed_dict={z:z_,img:batch_x.reshape([batch_size,28,28,1]),y:batch_y})
				print "d loss after batch ",k," is ",D_loss
				print "g loss after batch ",k," is ",G_loss
				k+=1
				if k % 100 ==0:
                    
					sample = sess.run(fake_img,feed_dict={z:z_})

					save_images(sample, image_manifold_size(sample.shape[0]),
                              './{}/{:02d}.png'.format("gran_out_cross", k))


                    

























