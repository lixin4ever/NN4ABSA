import theano
import theano.tensor as T
from theano.tensor.signal import pool, conv
from utils import *
from nn_utils import *


class NN:
	"""
	fully-connected neural networks
	"""
	def __init__(self, n_in, n_out, name, use_bias=False, activation='softmax'):
		"""
		"""
		W_shape = (n_in, n_out)
		self.W = glorot_uniform_share(shape=W_shape, name='W_%s' % name)
		self.b = theano.shared(np.array(np.zeros(n_out), dtype='float32'), name='b_%s' % name)
		self.activation = activation
		self.use_bias = use_bias
		self.params = [self.W]
		if self.use_bias:
			self.params.append(self.b)

	def forward(self, x, is_sequence=False):
		"""
		x: shape: (bs, n_in)
		W: shape: (n_in, n_out)
		"""
		if is_sequence:
			W_seq = T.tile(self.W, (x.shape[0], 1, 1))
			Wx = T.batched_dot(x, W_seq).reshape((x.shape[0] * x.shape[1], self.W.shape[1]))
		else:
			Wx = T.dot(x, self.W)
		if self.use_bias:
			Wx = Wx + self.b  # N to 1xN
		if self.activation == 'tanh':
			out = T.tanh(Wx)
		elif self.activation == 'softmax':
			out = T.nnet.softmax(Wx)
		else:
			raise Exception("Invalid activations!")
		if is_sequence:
			return out.reshape((x.shape[0], x.shape[1], self.W.shape[1]))
		else:
			return out

class LSTM:

	def __init__(self, win_size, dim_w, dim_h, bs, name):
		"""
		"""
		self.W, self.U, self.b = lstm_init(component='%s_lstm' % name, n_in=win_size * dim_w, n_out=dim_h)

		self.params = [self.W, self.U, self.b]

		self.n_in = win_size * dim_w

		self.n_out = dim_h

		self.bs = bs

	def forward(self, x):
		"""
		perform scan over the input sequence
		"""
		h_t0 = theano.shared(np.array(np.zeros((self.bs, self.n_out)), dtype='float32'))
		c_t0 = theano.shared(np.array(np.zeros((self.bs, self.n_out)), dtype='float32'))

		# H: h[1:T], C: c[1:T]
		[H, C], _ = theano.scan(fn=self.recurrence, sequences=[x.dimshuffle(1, 0, 2)],
			outputs_info=[h_t0, c_t0])

		# return the hidden states, shape: (bs, max_len, dim_h)
		return H.dimshuffle(1, 0, 2)

	def recurrence(self, x, h_tm1, c_tm1):
		"""
		recurrence function of lstm
		"""

		Wx = T.dot(x, self.W)
		Uh = T.dot(h_tm1, self.U)
		SUM = Wx + Uh + self.b

		i_t = T.nnet.hard_sigmoid(SUM[:, :self.n_out])
		f_t = T.nnet.hard_sigmoid(SUM[:, self.n_out:2*self.n_out])
		c_hat = T.tanh(SUM[:, 2*self.n_out:3*self.n_out])
		c_t = i_t * c_hat + f_t * c_tm1
		o_t = T.nnet.hard_sigmoid(SUM[:, 3*self.n_out:])
		h_t = o_t * T.tanh(c_t)
		return h_t, c_t

class CNN:

	def __init__(self, input_shape, filter_shape, activation='relu', pooling_mode=None):
		"""
		input_shape: (bs, 1, sent_length, dim_w)
		filter_shape: (n_filter, 1, kernel_height, kernel_width)
		"""
		assert input_shape[3] == filter_shape[3]

		self.kernel_size = filter_shape[2]
		
		# dimension of feature map
		self.poolsize = input_shape[2] - self.kernel_size

		self.W = glorot_uniform_share(shape=filter_shape, name='W_conv_%s' % self.kernel_size, 
			poolsize=self.poolsize, activation=activation)

		b_values = np.array(np.zeros(filter_shape[0]), dtype='float32')

		self.b = theano.shared(value=b_values, name='b_conv_%s' % (self.kernel_size))

		self.params = [self.W, self.b]

		self.input_shape = input_shape

		self.filter_shape = filter_shape

		self.activation = activation
		
		self.pooling_mode = pooling_mode

	def forward(self, x):
		"""
		perform the convolution over the input x
		"""
		# conv: shape: ()
		conv_feat = T.nnet.conv2d(input=x, filters=self.W, input_shape=self.input_shape, 
			filter_shape=self.filter_shape, 
			border_mode='valid')

		if self.activation == 'relu':
			conv_out = T.nnet.relu(conv_feat + self.b.dimshuffle('x', 0, 'x', 'x'))
		elif self.activation == 'tanh':
			conv_out = T.tanh(conv_feat + self.b.dimshuffle('x', 0, 'x', 'x'))
		else:
			raise Exception("Invalid activation function!!")
		if not self.pooling_mode or self.pooling_mode == 'lstm-att':
			return conv_out.reshape((self.input_shape[0], self.filter_shape[0], self.poolsize))
		elif self.pooling_mode == 'max':
			return pool.pool_2d(input=conv_out, ws=(self.poolsize, 1), ignore_border=True)
		else:
			raise Exception("Invalid pooling mode!!")