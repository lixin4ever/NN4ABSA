import numpy as np
import theano.tensor as T
import theano
import random
from collections import OrderedDict


def lstm_init(component, n_in, n_out, n_extra=0, initializer='glorot_uniform'):
    """
    initialize a LSTM unit
    notation style reference blog: http://deeplearning.net/tutorial/lstm.html
    """
    # order [i, f, c, o]
    if initializer == 'glorot_uniform':
        # non-recurrent weight matrices
        W_values = np.concatenate([glorot_uniform(shape=(n_in, n_out)),
                                   glorot_uniform(shape=(n_in, n_out)),
                                   glorot_uniform(shape=(n_in, n_out)),
                                   glorot_uniform(shape=(n_in, n_out))], axis=1)
        # recurrent weight matrices
        U_values = np.concatenate([orthogonal(shape=(n_out+n_extra, n_out)),
                                   orthogonal(shape=(n_out+n_extra, n_out)),
                                   orthogonal(shape=(n_out+n_extra, n_out)),
                                   orthogonal(shape=(n_out+n_extra, n_out))], axis=1)
        #V_values = glorot_uniform(shape=(n_out, n_out))
        # forget gate bias is set to 1.0
        b_values = np.concatenate([np.zeros(n_out, dtype='float32'),
                                   #np.ones(n_out, dtype='float32'), # forget gate bias
                                   np.zeros(n_out, dtype='float32'),
                                   np.zeros(n_out, dtype='float32'),
                                   np.zeros(n_out, dtype='float32')])

    W = theano.shared(value=W_values, name='W_%s' % component)
    U = theano.shared(value=U_values, name='U_%s' % component)
    b = theano.shared(value=b_values, name='b_%s' % component)
    #V = theano.shared(value=V_values, name='V_%s' % component)
    return W, U, b

def get_fan(shape, poolsize=None):
	# note: for convolution shape: (n_filters, filter_height, filter_width)
	# for hidden layer, shape: (n_in, n_out)
	if len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]
	else:
		fan_in = np.prod(shape[1:])
		fan_out = shape[0] * np.prod(shape[2:]) / float(poolsize)
	return fan_in, fan_out

def normal(shape, mean=0, stddev=0.01):
	"""
	sampling from normal distribution center at 0 with stddev 0.01
	"""
	values = np.random.normal(loc=mean, scale=stddev, size=shape)
	return np.array(values, dtype='float32')

def orthogonal(shape):
    """
    equivalent to orthogonal_init but return numpy array 
    """
    if len(shape) == 1:
        values = np.zeros_like(shape)
    else:
        a = np.random.normal(loc=0.0, scale=1.0, size=shape)
        # reconstruction based on reduced SVD
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        q = q.reshape(shape)
        values = q
    return values.astype("float32")

def normal_share(shape, name, mean=0, stddev=0.01):
	"""
	sampling from normal distribution center at 0 with stddev 0.01
	"""
	values = normal(shape=shape, mean=mean, stddev=stddev)
	return theano.shared(value=values, name=name)

def glorot_uniform(shape, activation='relu', poolsize=None):
	"""
	# glorot uniform initialization, return a shared tensor
	"""
	#if activation != 'relu':
	if False:
		fan_in, fan_out = get_fan(shape, poolsize)
		scale = np.sqrt(6.0 / (fan_in + fan_out))
		values = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype='float32')
	else:
		#print("Not glorot")
		# this kind of initialization is adopted in Kim Yoon's work.
		values = np.array(np.random.uniform(low=-0.1, high=0.1, size=shape), dtype='float32')
	return values

def glorot_uniform_share(shape, name, activation='relu', poolsize=None):
	values = glorot_uniform(shape, activation, poolsize)
	return theano.shared(values, name=name)

def kaiming_uniform(shape, activation='relu', poolsize=None):
	fan_in, fan_out = get_fan(shape, poolsize)
	scale = np.sqrt(6.0 / fan_in)
	values = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype='float32')
	return values

def kaiming_uniform_share(shape, name, activation='relu', poolsize=None):
	values = kaiming_uniform(shape, activation, poolsize)
	return theano.shared(values, name=name)

def adam(cost, params, max_norm=3.0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
	"""
	adam optimizer, default learning rate is 0.001
	"""
	
	grads = T.grad(cost, params)
	t_prev = theano.shared(value=np.float32(0.0))
	updates = OrderedDict()

	t = t_prev + 1
	one = T.constant(1, dtype='float32')

	a_t = lr * T.sqrt(one - beta2**t) / (one - beta1**t)

	for p, g in zip(params, grads):
		value = p.get_value(borrow=True)
		m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), 
								broadcastable=p.broadcastable)
		v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                            	broadcastable=p.broadcastable)

		m_t = beta1 * m_prev + (one-beta1) * g
		v_t = beta2 * v_prev + (one-beta2) * g ** 2

		step = a_t * m_t / (T.sqrt(v_t) + epsilon)

		stepped_p = p - step
		updates[m_prev] = m_t
		updates[v_prev] = v_t
		# add max norm constraint on the W_hidden
		#if p.name == 'W_hidden':
		#	col_norms = T.sqrt(T.sum(T.sqr(stepped_p), axis=0))
		#	desired_norms = T.clip(col_norms, 0, max_norm)
		#	scale = desired_norms / (1e-7 + col_norms)
		#	updates[p] = stepped_p * scale
		#else:
		#	updates[p] = stepped_p
		updates[p] = stepped_p
	updates[t_prev] = t
	return updates

def adadelta(cost, params, lr=0.1, rho=0.9, epsilon=1e-6):
	"""
	adadelta optimizer for neural network
	"""
	grads = T.grad(cost, params)
	updates = OrderedDict()

	one = T.constant(1)

	for p, g in zip(params, grads):
		value = p.get_value(borrow=True)
		# accu: accumulate update magnitudes
		accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			broadcastable=p.broadcastable)
		# delta_accu: accumulate update magnitudes (recursively)
		delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			broadcastable=p.broadcastable)

		# update accu
		accu_new = rho * accu + (one - rho) * g ** 2
		updates[accu] = accu_new

		# compute parameter update, using the old delta_accu
		update = (g * T.sqrt(delta_accu + epsilon) / T.sqrt(accu_new + epsilon))
		updates[p] = p - lr * update

		# update delta_accu
		delta_accu_new = rho * delta_accu + (one - rho) * g ** 2
		updates[delta_accu] = delta_accu_new
	return updates

def hinge_loss(pred_dist, true_label, delta=1.0):
	"""
	Reference: On the algorithmic implementation of multiclass kernel-based vector machines
	pred_dist: 2D tensor, the output probabilities of softmax layer
	true_label: 1D tensor, the gold standard sentiment label
	delta: hinge loss margin

	"""
	n_cls = pred_dist.shape[1]
	true_dist = T.extra_ops.to_one_hot(true_label, n_cls)
	correct = pred_dist[true_dist.nonzero()]
	errors = T.reshape(pred_dist[(1-true_dist).nonzero()], (-1, n_cls-1))
	error = T.max(errors, axis=1)
	#error = errors[:, n_cls-2]  # fix the last class as negative sample: (postive, neutral), (negative, neutral), (neutral, negative)
	return T.nnet.relu(error - correct + delta)

def dos_santos_loss(pred_dist, true_label, p_margin=2.5, n_margin=0.5, factor=2):
	"""
	margin-based loss function
	Reference: 
	Santos C N, Xiang B, Zhou B. Classifying relations by ranking with convolutional neural networks[J]. 
	arXiv preprint arXiv:1504.06580, 2015.
	"""
	n_cls = pred_dist.shape[1]
	true_dist = T.extra_ops.to_one_hot(true_label, n_cls)
	p_score = pred_dist[true_dist.nonzero()]
	n_scores = T.reshape(pred_dist[(1-true_dist).nonzero()], (-1, n_cls-1))
	n_score = T.max(n_scores, axis=1)
	#n_score = pred_dist[:, 2]
	loss = T.log(1.0 + T.exp(factor * (p_margin - p_score))) + T.log(1.0 + T.exp(factor * (n_margin + n_score)))
	return loss

def wang_loss(pred_dist, true_label):
	"""
	Margin-based pairwise loss
	Reference: Wang L, Cao Z, de Melo G, et al. Relation Classification via Multi-Level Attention CNNs[C].
	ACL (1). 2016.
	"""
	n_cls = pred_dist.shape[1]
	true_dist = T.extra_ops.to_one_hot(true_label, n_cls)
	p_score = pred_dist[true_dist.nonzero()]
	n_scores = T.reshape(pred_dist[(1-true_dist).nonzero()], (-1, n_cls-1))
	n_score = T.max(n_scores, axis=1)
	loss = p_score + (1.0 - n_score)
	return loss

def shared(data_list, dtype='float32'):
	"""
	convert numpy arrays to theano shared variables
	"""
    tensor_list = []
    for data in data_list:
        t = theano.shared(value=np.asarray(data, dtype=dtype), borrow=True)
        tensor_list.append(t)
    return tensor_list