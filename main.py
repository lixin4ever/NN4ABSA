from model import *
import argparse
from utils import *
from evals import *
import lasagne
import theano
import theano.tensor as T
import numpy as np

class NN4ABSA:
	# neural network based models for aspect-based sentiment analysis
	def __init__(self, args):
		"""
		"""
		self.ds_name = args.ds_name
		self.bs = args.bs
		self.ctx_mode = args.ctx_mode
		self.use_val = args.use_val
		self.dim_w = args.dim_w
		self.dim_h = args.dim_w
		self.n_filter = args.n_filter
		self.ctx_feat_mode = args.ctx_feat_mode
		self.target_feat_mode = args.target_feat_mode
		self.dropout_rate = args.dropout_rate
		self.pooling_mode = args.pooling_mode
		self.dim_y = 3

		dataset = build_dataset_nn(dataset_name=self.ds_name, batch_size=self.bs, use_val=self.use_val, ctx_mode=self.ctx_mode)

		# determine convolutional kernels
		if args.ds_name == '14semeval_rest':
			self.kernels = [3, 4, 5]
			self.n_epoch = 200
		elif args.ds_name == '14semeval_laptop':
			self.kernels = [3]
			self.n_epoch = 200
		elif args.ds_name == 'Twitter':
			self.kernels = [3, 4, 5]
			self.n_epoch = 100
		else:
			raise Exception("Invalid dataset name!!")

		self.train_x, self.train_x_t, self.train_y = dataset[:3]
		self.val_x, self.val_x_t, self.val_y = dataset[3:6]
		self.test_x, self.test_x_t, self.test_y = dataset[6:9]

		vocab, embedding_weights = dataset[9:11]
		self.n_train, self.n_test = dataset[11:13]

		self.max_len = self.train_x.shape[1]
		print("max length:", self.max_len)
		self.max_len_t = self.train_x_t.shape[1]

		self.n_train_batches = self.train_x.shape[0] // self.bs
		self.n_val_batches = self.val_x.shape[0] // self.bs
		self.n_test_batches = self.test_x.shape[0] // self.bs

		self.WDEmb = theano.shared(value=embedding_weights, name='W_embedding')

		self.is_conv = True
		# padded input for cnn
		vectors = [transform(data=[self.train_x, self.val_x, self.test_x], 
			embedding_weights=embedding_weights, kernel_size=k, win_size=1, is_conv=self.is_conv) for k in self.kernels]

		self.train_x = [data[0] for data in vectors]
		self.val_x = [data[1] for data in vectors]
		self.test_x = [data[2] for data in vectors]

		while len(self.train_x) < 3:
			self.train_x.append(self.train_x[0])
			self.val_x.append(self.val_x[0])
			self.test_x.append(self.test_x[0])	

		print(self.train_x[0].shape)
		print(self.train_x[1].shape)
		print(self.train_x[2].shape)

		self.train_x = shared(data_list=self.train_x, dtype='float32')
		self.val_x = shared(data_list=self.val_x, dtype='float32')
		self.test_x = shared(data_list=self.test_x, dtype='float32')

		self.train_x_t, self.val_x_t, self.test_x_t = transform(data=[self.train_x_t, self.val_x_t, self.test_x_t], 
			embedding_weights=embedding_weights)
		self.train_x_t, self.val_x_t, self.test_x_t = shared(data_list=[self.train_x_t, self.val_x_t, self.test_x_t], dtype='float32')
		self.train_y, self.val_y, self.test_y = shared(data_list=[self.train_y, self.val_y, self.test_y], dtype='int32')

		# symbolic input of the computational graph
		# input for multi-channel cnn
		self.x1, self.x2, self.x3 = [T.tensor3(name='input_x%s' % (i+1)) for i in range(3)]
		self.xt = T.tensor3(name='input_target')
		self.y = T.ivector(name='y_sentiment')

		self.build_graph()

		self.make_func()			

	def build_graph(self):
		"""
		build the computational graph
		"""

		self.layers = []

		if self.ctx_feat_mode == 'cnn':
			for k in self.kernels:
				input_shape = (self.bs, 1, self.max_len + k - 1, self.dim_w)
				filter_shape = (self.n_filter, 1, k, self.dim_w)
				self.layers.append(CNN(input_shape=input_shape, filter_shape=filter_shape, pooling_mode=self.pooling_mode))
			conv_inputs = [self.x1, self.x2, self.x2]
			conv_outputs = []
			for i in range(len(self.kernels)):
				conv_outputs.append(self.layers[i].forward(x=conv_inputs[i].dimshuffle(0, 'x', 1, 2)).flatten(2))
			ctx_feat = T.concatenate(conv_outputs, axis=1)
		elif self.ctx_feat_mode == 'lstm':
			self.layers.append(LSTM(win_size=1, dim_w=self.dim_w, dim_h=self.dim_h, bs=self.bs, name='lstm_ctx'))
			ctx_feat = self.layers[-1].forward(x=self.x1).mean(axis=1)
		else:
			raise Exception("Invalid context feature mode!!")

		if self.target_feat_mode == 'weight_sum':
			one_sum_prob = np.random.uniform(low=0, high=1, size=(1, self.max_len_t))
			self.W_target = theano.shared(value=np.array(one_sum_prob, dtype='float32'), name='W_target')
			self.W_target_norm = T.nnet.softmax(self.W_target).reshape((self.max_len_t,))
			target_feat = T.dot(self.W_target_norm, self.xt)
		elif self.target_feat_mode == 'mean':
			target_feat = self.xt.mean(axis=1)
		else:
			raise Exception("Invalid target feature mode!!")

		feat = T.concatenate([ctx_feat, target_feat], axis=1)
		srng = T.shared_randomstreams.RandomStreams(np.random.randint(99999))
		mask = srng.binomial(n=1, p=1-self.dropout_rate, size=feat.shape)
		feat_dropout = feat * mask

		n_in = 0
		if self.ctx_feat_mode == 'cnn':
			n_in += len(self.kernels) * self.n_filter
		elif self.ctx_feat_mode == 'lstm':
			n_in += self.dim_h
		else:
			raise Exception("Invalid context feature mode")
		if self.target_feat_mode == 'weight_sum':
			n_in += self.dim_w
		else:
			raise Exception("Invalid target feature mode")
		self.layers.append(NN(n_in=n_in, n_out=self.dim_y, name='fc_layer', use_bias=True))

		p_y_x = self.layers[-1].forward(x=feat * (1 - self.dropout_rate))
		p_y_x_dropout = self.layers[-1].forward(x=feat_dropout)

		self.pred = T.argmax(p_y_x, axis=1)
		self.pred_dropout = T.argmax(p_y_x_dropout, axis=1)

		self.loss = T.nnet.categorical_crossentropy(coding_dist=p_y_x_dropout, true_dist=self.y).mean()

		self.params = []
		for layer in self.layers:
			self.params.extend(layer.params)
		if self.target_feat_mode == 'weight_sum':
			self.params.append(self.W_target)
		print(self.params)

	def make_func(self):
		"""
		make theano functions
		"""
		updates = lasagne.updates.adam(loss_or_grads=self.loss, params=self.params)
		
		index = T.iscalar()

		train_outputs= [self.pred_dropout, self.y, self.loss]
		val_outputs = [self.pred, self.y, self.loss]
		test_outputs = [self.pred, self.y, self.loss]

		self.train_func = theano.function(inputs=[index], outputs=train_outputs, updates=updates,
			givens={
				self.x1: self.train_x[0][index*self.bs:(index+1)*self.bs],
				self.x2: self.train_x[1][index*self.bs:(index+1)*self.bs],
				self.x3: self.train_x[2][index*self.bs:(index+1)*self.bs],
				self.xt: self.train_x_t[index*self.bs:(index+1)*self.bs],
				self.y: self.train_y[index*self.bs:(index+1)*self.bs], 
			})

		self.val_func = theano.function(inputs=[index], outputs=val_outputs,
			givens={
				self.x1: self.val_x[0][index*self.bs:(index+1)*self.bs],
				self.x2: self.val_x[1][index*self.bs:(index+1)*self.bs],
				self.x3: self.val_x[2][index*self.bs:(index+1)*self.bs],
				self.xt: self.val_x_t[index*self.bs:(index+1)*self.bs],
				self.y: self.val_y[index*self.bs:(index+1)*self.bs] 
			})

		self.test_func = theano.function(inputs=[index], outputs=test_outputs,
			givens={
				self.x1: self.test_x[0][index*self.bs:(index+1)*self.bs],
				self.x2: self.test_x[1][index*self.bs:(index+1)*self.bs],
				self.x3: self.test_x[2][index*self.bs:(index+1)*self.bs],
				self.xt: self.test_x_t[index*self.bs:(index+1)*self.bs],
				self.y: self.test_y[index*self.bs:(index+1)*self.bs]
			})

	def run(self):
		"""
		training and evaluation
		"""
		best_val_metric, best_test_metric = -9999.0, -9999.0
		best_val_loss, best_test_loss = 9999.0, 9999.0
		best_epoch = 0
		best_value = (0.0, 0.0, 'TEMP')
		for i in range(self.n_epoch):
			print('Epoch %s/%s' % (i+1, self.n_epoch))
			train_losses, val_losses, test_losses = [], [], []
			train_y_pred, val_y_pred, test_y_pred = [], [], []
			train_y_gold, val_y_gold, test_y_gold = [], [], []
			for j in np.random.permutation(range(self.n_train_batches)):
				y_pred, y_gold, losses = self.train_func(j)
				train_y_gold.extend(np.array(y_gold, dtype='int32'))
				train_y_pred.extend(np.array(y_pred, dtype='int32'))
				train_losses.append(losses)
			train_acc, train_f1, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
			
			log = 'train loss: %.4f, train acc: %.4f, train f1: %.4f, ' % (sum(train_losses), train_acc, train_f1)
			if self.use_val:
				for j in range(self.n_val_batches):
					y_pred, y_gold, losses = self.val_func(j)
					val_losses.append(losses)
					val_y_gold.extend(np.array(y_gold, dtype='int32'))
					val_y_pred.extend(np.array(y_pred, dtype='int32'))
				val_acc, val_f1, _ = evaluate(pred=val_y_pred, gold=val_y_gold)
				val_loss = sum(val_losses)
				log += 'val loss: %.4f, val acc: %.4f, val f1: %.4f' % (val_loss, val_acc, val_f1)
				print(log)
				# use validation accuracy to select model
				#if val_acc > best_val_metric:
				#	best_val_metric = val_acc
				#	best_epoch = i + 1
				for j in range(self.n_test_batches):
					y_pred, y_gold, losses = self.test_func(j)
					test_losses.append(losses)
					test_y_gold.extend(np.array(y_gold, dtype='int32'))
					test_y_pred.extend(np.array(y_pred, dtype='int32'))
				# when calculate the test acc, do not involve the padding dataset
				test_acc, test_f1, res_string = evaluate(pred=test_y_pred[:self.n_test], gold=test_y_gold[:self.n_test])
				if test_acc > best_test_metric:
					best_test_metric = test_acc
					best_epoch = i + 1
					print("Current best results: acc: %.4f, f1: %.4f" % (test_acc, test_f1))
					print(res_string)
					best_value = (test_acc, test_f1, res_string)
			else:
				for j in range(self.n_test_batches):
					y_pred, y_gold, losses = self.test_func(j)
					test_losses.append(losses)
					test_y_gold.extend(np.array(y_gold, dtype='int32'))
					test_y_pred.extend(np.array(y_pred, dtype='int32'))
				# when calculate the test acc, do not involve the padding dataset
				test_acc, test_f1, res_string = evaluate(pred=test_y_pred[:self.n_test], gold=test_y_gold[:self.n_test])
				log += 'test loss: %.4f, test acc: %.4f, test f1: %.4f' % (sum(test_losses), test_acc, test_f1)
				print(log)
				if test_acc < best_test_metric:
					best_test_metric = test_acc
					best_epoch = i + 1
					best_value = (test_acc, test_f1, res_string)
		print('In epoch %s:' % best_epoch)
		print('\nBest acc: %.4f, best f1: %.4f\n%s' % (best_value[0], best_value[1], best_value[2]))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='NN Models for ABSA')

	parser.add_argument("-ds_name", type=str, default="Twitter", help="dataset name")
	parser.add_argument("-n_filter", type=int, default=100, help="number of convolutional filters")
	parser.add_argument("-ctx_mode", type=str, default="full", help="context mode for each sentence")
	parser.add_argument("-bs", type=int, default=64, help="batch size")
	parser.add_argument("-model_type", type=str, default="static", help="model type")
	parser.add_argument("-use_val", type=int, default=1, help="model name")
	parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings")
	parser.add_argument("-dropout_rate", type=float, default=0.8, help="dropout rate")
	parser.add_argument("-dim_h", type=int, default=20, help="dimension of hidden state")
	parser.add_argument("-dim_h_target", type=int, default=100, help="dimension of hidden state for target lstm")
	parser.add_argument("-pooling_mode", type=str, default="max", help="pooling mode in the convolution layer")
	parser.add_argument("-ctx_feat_mode", type=str, default="cnn", help="method that learn the context features")
	parser.add_argument("-target_feat_mode", type=str, default="weight_sum", help="method that learn the target features")
	parser.add_argument("-use_tgt_specific_feat", type=int, default=0, help="use auxilliary task or not")
	parser.add_argument("-use_l2", type=int, default=0, help="use L2 regularization")
	parser.add_argument("-l2", type=float, default=1e-5, help="coefficient of L2 regularizer")

	args = parser.parse_args()

	print(args)

	model = NN4ABSA(args)

	model.run()