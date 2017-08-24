# -*- coding: utf-8 -*-
import numpy as np
import random
import string
import pickle
import os
from nltk import ngrams


def load_twitter_data(file_path):
	"""
	load Twitter data
	"""
	count = 0
	ctx, target, y = [], [], []
	vocab = {}
	idx = 1 # index of word start from 1
	with open(file_path) as fp:
		for line in fp:
			if count % 3 == 0:
				# filter stop words, numbers and punctuations
				# ctx.append([ele for ele in line.strip().lower().split(' ') if ele not in stopword_list and not ele.isdigit()])
				cws = line.strip().lower().split(' ')
				cws = preprocess(words=cws)
				#print(cws)
				ctx.append(cws)
				for w in cws:
					if w not in vocab and w != '$t$':
						vocab[w] = idx
						idx += 1
                       
			elif count % 3 == 1:
				tws = line.strip().lower().split(' ')
				target.append(tws)
				for w in tws:
					if w not in vocab:
						vocab[w] = idx
						idx += 1

			elif count % 3 == 2:
				label_text = line.strip()
				# negative: 0, positive: 1, neutral: 2
				if label_text == '-1':
					y.append(0)
				elif label_text == '1':
					y.append(1)
				elif label_text == '0':
					y.append(2)
			count += 1
	return ctx, target, np.array(y, dtype='int32')

def load_semeval_data(file_path):
	"""
	load data from SemEval dataset
	"""
	ctx, target, y = [], [], []
	vocab = {}
	idx = 1 # index start from 1
	print("Load data from %s..." % file_path)
	class_count = np.zeros(3)
	neu_idxs = []
	sample_idx = 0
	with open(file_path) as fp:
		for line in fp:
			eles = line.strip().split()
			cws = [] # context words
			tws = [] # target words
			lable = 0
			for w in eles:
				if '/p' in w or '/n' in w or '/0' in w:
					if '$t$' not in cws:
						cws.append('$t$')
					if '/p' in w:
						# positive
						word, _ = w.split('/p')
						label = 1
					elif '/n' in w:
						# negative
						word, _ = w.split('/n')
						label = 0
					elif '/0' in w:
						# neutral
						word, _ = w.split('/0')
						label = 2
					tws.append(word)
					if word not in vocab:
						vocab[word] = idx
						idx += 1
				else:
					cws.append(w)
					if w not in vocab:
						vocab[w] = idx
						idx += 1
			class_count[label] += 1
			ctx.append(cws)
			target.append(tws)
			y.append(label)
			if label == 2:
				neu_idxs.append(sample_idx)
			sample_idx += 1
	balanced_neu_count = int(class_count[1] * 0.7)
	extra_neu_count = int(balanced_neu_count - class_count[2])
	extra_ctx, extra_target, extra_y = [], [], []
	if 'train' in file_path and 'news_comments' not in file_path:
		# perform data augmentation on the training set
		#print("length of neu_idxs:", len(neu_idxs))
		#print("extra_count:", extra_neu_count)
		for idx in np.random.choice(neu_idxs, extra_neu_count):
			extra_ctx.append(ctx[idx])
			extra_target.append(target[idx])
			extra_y.append(y[idx])
		print("Augmented %s neural samples in the training set!!" % extra_neu_count)
		ctx.extend(extra_ctx)
		target.extend(extra_target)
		y.extend(extra_y)
	return ctx, target, np.array(y, dtype='int32')

def load_word_embeddings(emb_file='../OTE/embeddings/glove_twitter_200d.txt', splitter=' '):
	"""
	load pre-trained word embeddings
	"""
	word_embeddings = {}
	count = 0
	print("load word vectors....")
	with open(emb_file) as fp:
		for line in fp:
			count += 1
			eles = line.strip().split(splitter)
			w = eles[0]
			word_embeddings[w] = eles[1:]
	print("load %s word vectors" % count)
	return word_embeddings

def load_lexicon(file_path='./lexicon/sent_lexicon_full.txt'):
	"""
	load sentiment lexicon
	"""
	lexicon = {}
	count = 0
	n_lines = 0
	with open(file_path, encoding='utf-8', errors='ignore') as fp:
		for line in fp:
			words = line.strip().split()
			n_lines += 1
			for w in words:
				if w not in lexicon:
					lexicon[w] = count
					count += 1
	return lexicon

def preprocess(words):
	"""
	process twitter sentence by filtering numbers, pronouns, punctuations
	"""

	intab = ''.join([ele for ele in string.punctuation if ele != '$'])
	#intab += '0123456789'
	outtab = '\t' * len(intab)
	trantab = str.maketrans(intab, outtab)
	sent = ' '.join(words)
	sent_norm = sent.translate(trantab).replace('\t', '')
	return sent_norm.split()

def get_vocab(sents, targets):
	"""
	build vocab
	"""
	vocab = {}
	idx = 1 # word index starts from 1
	for ws in sents:
		for w in ws:
			if w not in vocab and w != '$t$':
				vocab[w] = idx
				idx += 1
	for ws in targets:
		for w in ws:
			if w not in vocab and w != '$t$':
				vocab[w] = idx
				idx += 1
	return vocab

def max_pooling(feats, dim_w):
	feats = np.array(feats, dtype='float32')
	if feats.shape[0] == 0:
		feats = np.zeros((1, dim_w), dtype='float32')
	return np.amax(feats, axis=0)

def min_pooling(feats, dim_w):
	feats = np.array(feats, dtype='float32')
	if feats.shape[0] == 0:
		feats = np.zeros((1, dim_w), dtype='float32')
	return np.amin(feats, axis=0)

def mean_pooling(feats, dim_w):
	feats = np.array(feats, dtype='float32')
	if feats.shape[0] == 0:
		feats = np.zeros((1, dim_w), dtype='float32')
	return np.sum(feats, axis=0) / feats.shape[0]

def hybrid_pooling(feats, dim_w):
	# max pooling
	max_feat = max_pooling(feats, dim_w)
	# mean pooling
	mean_feat = mean_pooling(feats, dim_w)
	# min pooling
	min_feat = min_pooling(feats, dim_w)
	return np.concatenate([max_feat, mean_feat, min_feat])

def word2vec(w2vs, word_seqs, kernel_size=1, win_size=1, is_conv=False):
	"""
	w2vs: word vectors
	win_size is for the window-based input (odd number)
	kernel_size is for the padding in the convolution operation
	"""
	new_data = []
	dim_w = len(w2vs[0])
	assert (kernel_size > 1 and is_conv) or (kernel_size == 1 and (not is_conv))
	n_padded = kernel_size - 1
	if n_padded % 2 == 0:
		n_padded_pre = n_padded_post = n_padded // 2
	else:
		n_padded_pre = (n_padded + 1) // 2
		n_padded_post = n_padded - n_padded_pre
	pad_vecs_pre = np.array(np.zeros((n_padded_pre, win_size * dim_w)), dtype='float32')
	pad_vecs_post = np.array(np.zeros((n_padded_post, win_size * dim_w)), dtype='float32')
	for words in word_seqs:
		if win_size == 1:
			#new_data.append(w2vs[words])
			vecs = w2vs[words]
			if is_conv:
				vecs = np.concatenate([pad_vecs_pre, vecs, pad_vecs_post])
		else:
			tmp = []
			sent_len = len(words)
			# padding for generating the window-based ngrams
			padded_word = [0 for _ in range(win_size // 2)]
			tmp.extend(padded_word) # prefix
			tmp.extend(words)
			tmp.extend(padded_word)  # postfix
			new_words = np.array(list(ngrams(tmp, win_size)), dtype='int32').flatten()
			vecs = w2vs[new_words].reshape((sent_len, win_size * dim_w))
			if is_conv:
				vecs = np.concatenate([pad_vecs_pre, vecs, pad_vecs_post])
		#if conv and win_size > 1:
		#	#assert kernel_size is not None
		#	# pad the original sentence to ensure that the output shape of conv is equal to that of input
		#	pad_vecs = np.array([np.zeros(win_size * dim_w) for _ in range(int(win_size / 2))], dtype='float32')
		#	vecs = np.concatenate((pad_vecs, vecs, pad_vecs), axis=0)
		new_data.append(vecs) 
	return np.array(new_data, dtype='float32')

def transform(data, embedding_weights, kernel_size=1, win_size=1, is_conv=False):
	return [word2vec(w2vs=embedding_weights, word_seqs=seqs, kernel_size=kernel_size, win_size=win_size, is_conv=is_conv) for seqs in data]

def padding(seqs, max_len, symbol=0):
	padded_seqs = []
	for seq in seqs:
		tmp = [ele for ele in seq]
		while len(tmp) < max_len:
			tmp.append(symbol)
		padded_seqs.append(tmp)
	return np.array(padded_seqs, dtype='int32')

def process_sent(train_sent, test_sent, train_target, test_target, vocab, ds_name, lexicon):
	"""
	split sent as left context and right context (only sentimental words are kept)
	"""
	max_len_l, max_len_r = -1, -1
	# left context and right context
	train_lctx, train_rctx = [], []
	# left context and right context
	test_lctx, test_rctx = [], []
	# number of training samples
	n_train = len(train_sent)
	count = 0

	for sent in train_sent + test_sent:
		t_pos = sent.index('$t$')
		l_ctx, r_ctx = [], []
		for i in range(len(sent)):
			w = sent[i]
			if i < t_pos and w != '$t$' and True:
				l_ctx.append(vocab[w])
			elif i > t_pos and w != '$t$' and True:
				r_ctx.append(vocab[w])
		if count < n_train:
			tws = [vocab[w] for w in train_target[count]]
		else:
			tws = [vocab[w] for w in test_target[count-n_train]]
		# add target words in the contexts
		#if ds_name == 'Twitter' or ds_name == '14semeval_rest':
		l_ctx.extend(tws)
		r_ctx = tws + r_ctx
		if len(l_ctx) > max_len_l:
			max_len_l = len(l_ctx)
		if len(r_ctx) > max_len_r:
			max_len_r = len(r_ctx)
		if count < n_train:
			train_lctx.append(l_ctx)
			train_rctx.append(r_ctx)
		else:
			test_lctx.append(l_ctx)
			test_rctx.append(r_ctx)
		count += 1

	train_x_l = padding(seqs=train_lctx, max_len=max_len_l)
	train_x_r = padding(seqs=train_rctx, max_len=max_len_r)

	test_x_l = padding(seqs=test_lctx, max_len=max_len_l)
	test_x_r = padding(seqs=test_rctx, max_len=max_len_r)
	
	return train_x_l, train_x_r, test_x_l, test_x_r

def process_sent2(train_sent, test_sent, train_target, test_target, vocab, ds_name, lexicon=None):
	"""
	"""
	max_len = -1
	n_train = len(train_sent)
	count = 0
	train_x, test_x = [], []
	train_wc, test_wc = [], []
	train_tpos, test_tpos = [], []
	train_y_opi, test_y_opi = [], []
	for sent in train_sent + test_sent:
		temp, temp_opi = [], []
		tpos = 0
		find_target = False
		if count < n_train:
			tws = [w for w in train_target[count]]
		else:
			tws = [w for w in test_target[count-n_train]]
		if lexicon:
			tws_opi = [int(w in lexicon) for w in tws]
		else:
			tws_opi = [0 for w in tws]
		for w in sent:
			if w == '$t$':
				#temp.append(0)
				#temp_opi.append(0)
				temp.extend([vocab[w] for w in tws])
				temp_opi.extend(tws_opi)
				#pass
				find_target = True
			else:
				temp.append(vocab[w])
				if lexicon and w in lexicon:
					temp_opi.append(1)
				else:
					temp_opi.append(0)
			if not find_target:
				tpos += 1
		if count < n_train:
			train_x.append([wid for wid in temp])
			train_wc.append(len(sent))
			train_tpos.append(tpos)
			train_y_opi.append(temp_opi)
		else:
			test_x.append([wid for wid in temp])
			test_wc.append(len(sent))
			test_tpos.append(tpos)
			test_y_opi.append(temp_opi)
		if len(temp) > max_len:
			max_len = len(temp)
		count += 1
	print("max length: %s" % max_len)
	train_x = padding(seqs=train_x, max_len=max_len)
	test_x = padding(seqs=test_x, max_len=max_len)
	train_y_opi = padding(seqs=train_y_opi, max_len=max_len)
	test_y_opi = padding(seqs=test_y_opi, max_len=max_len)
	return train_x, test_x, train_wc, test_wc, train_tpos, test_tpos, train_y_opi, test_y_opi

def process_target(train_target, test_target, vocab):
	"""
	"""
	n_train = len(train_target)
	count = 0
	max_len_target = -1
	train_x_t, test_x_t = [], []
	train_wc_t, test_wc_t = [], []
	for target in train_target + test_target:
		tws = []
		for w in target:
			tws.append(vocab[w])
		if max_len_target < len(tws):
			max_len_target = len(tws)
		if count < n_train:
			train_x_t.append(tws)
			train_wc_t.append(len(target))
		else:
			test_x_t.append(tws)
			test_wc_t.append(len(target))
		count += 1
	train_x_t = padding(seqs=train_x_t, max_len=max_len_target)
	test_x_t = padding(seqs=test_x_t, max_len=max_len_target)
	return train_x_t, test_x_t, train_wc_t, test_wc_t

def pad_dataset(data_tuples, batch_size, mode='train'):
	"""
	pad and shuffle the dataset
	data_tupes = [(sent_1, target_1, label_1), ..., (sent_n, target_n, label_n)]
	"""
	# copy of data tuples
	shuffled_data = [ele for ele in data_tuples]
	if mode == 'train':
		# shuffle the training set and validation set
		random.shuffle(shuffled_data)
	n_sample = len(data_tuples)
	n_extra = batch_size - n_sample % batch_size
	shuffled_data.extend(shuffled_data[:n_extra])
	shuffled_components = [list(ele) for ele in list(zip(*shuffled_data))]
	#print(shuffled_components)
	return shuffled_components

def split_train(train_data, train_label, batch_size):
	"""
	split original training set as training set and validation set
	note: the ratio of train/val/test is kept same with the orignal training set
	"""
	n_samples = int(len(train_data[0]))
	split_rate = 0.9 # 75% data is used for training and the rest for testing
	total_count = np.zeros(3, dtype='int32')
	for l in train_label:
		total_count[l] += 1
	class_count = np.zeros(3, dtype='int32')

	# number of data field (including the label)
	n_fields = len(train_data) + 1

	train_data_new = [[] for _ in range(n_fields)]
	val_data_new = [[] for _ in range(n_fields)]

	for i in range(n_samples):
		y = train_label[i]
		data = [ele[i] for ele in train_data]
		if class_count[y] < int(total_count[y] * split_rate):
			for j in range(len(train_data)):
				train_data_new[j].append(data[j])
			train_data_new[-1].append(y)
		else:
			for j in range(len(train_data)):
				val_data_new[j].append(data[j])
			val_data_new[-1].append(y)
		class_count[y] += 1
	return train_data_new, val_data_new

def to_nparray(data_list, dtype='float32'):
	return [np.array(data, dtype=dtype) for data in data_list]

def copy(data_list):
	"""
	return copies of numpy arrays
	"""
	return [np.array(arr.copy(), dtype='int32') for arr in data_list]

def init_ndarray_zero(shape_list, dtype='float32'):
	return [np.array(np.zeros(shape), dtype=dtype) for shape in shape_list]

def calculate_pos_weight(word_count, pos, max_len):
	"""
	calculate position weight score for the pooling component
	"""
	assert len(word_count) == len(pos)
	weights = []
	for i in range(len(word_count)):
		wc = word_count[i]
		# target position
		tpos = pos[i]
		tmp = []
		for j in range(max_len):
			if j >= wc or j == tpos:
				tmp.append(0.0)
			else:
				dist = (wc - abs(j - tpos)) / float(wc)
				tmp.append(dist)
		weights.append(tmp)
	return np.array(weights, dtype='float32')

def build_dataset_nn(dataset_name, batch_size, use_val=True, lexicon=None, ctx_mode='full'):
    """
    build dataset for neural network-based model
    """
    train_file = './dataset/%s/train.txt' % dataset_name
    test_file = './dataset/%s/test.txt' % dataset_name

    print("Load dataset and pre-trained word vectors...")
    if dataset_name == 'Twitter':
        train_sent, train_t, train_y = load_twitter_data(file_path=train_file)
        test_sent, test_t, test_y = load_twitter_data(file_path=test_file)
        # use default Twitter embeddings
        emb_file = '../OTE/embeddings/glove_42B_300d.txt'
        #w2vs = load_word_embeddings(emb_file=emb_file)
        if not os.path.exists('./embeddings/%s' % dataset_name):
        	w2vs = load_word_embeddings(emb_file=emb_file)
    else:
        train_sent, train_t, train_y = load_semeval_data(file_path=train_file)
        test_sent, test_t, test_y = load_semeval_data(file_path=test_file)
        # use domain-specific word embeddings
        if "laptop" in dataset_name:
        	emb_file = '../OTE/embeddings/glove_42B_300d.txt'
        	#emb_file = '../amazon_full/vectors/amazon_laptop_vec_200_5.txt'
        elif "rest" in dataset_name:
        	#emb_file = '../yelp/yelp_vec_200_2_win5_sent.txt'
        	emb_file = '../OTE/embeddings/glove_42B_300d.txt'
        elif 'news' in dataset_name:
        	emb_file = './embeddings/news_comments.txt'
        if not os.path.exists('./embeddings/%s' % dataset_name):
        	w2vs = load_word_embeddings(emb_file=emb_file)
    assert len(train_t) == len(train_sent) == len(train_y)

    vocab = get_vocab(sents=train_sent+test_sent, targets=train_t+test_t)
    #vocab['$t$'] = len(vocab)

    #print("save vocab of %s..." % dataset_name)

    #pickle.dump(vocab, open('./embeddings/%s.voc' % dataset_name, 'wb'))

    train_x_t, test_x_t, train_wc_t, test_wc_t = process_target(train_target=train_t, test_target=test_t, vocab=vocab)

    # fetch embeddings for the words in the dataset
    cc = 0
    if not os.path.exists('./embeddings/%s' % dataset_name):
    	dim_w = len(w2vs['the'])
    	embeddings = np.zeros((len(vocab) + 1, dim_w), dtype='float32')
    	for w in vocab:
    		wid = vocab[w]
    		if w in w2vs:
    			embeddings[wid] = np.array([float(ele) for ele in w2vs[w]], dtype='float32')
    			cc += 1
    		#else:
    		#	embeddings[wid] = np.array(np.random.uniform(-0.1, 0.1, dim_w), dtype='float32')
    	# dump the embedding
    	pickle.dump(embeddings, open('./embeddings/%s' % dataset_name, 'wb'))
    else:
    	embeddings = pickle.load(open('./embeddings/%s' % dataset_name, 'rb'))
    	dim_w = len(embeddings[0])
    for i in range(embeddings.shape[0]):
    	if not np.count_nonzero(embeddings[i]):
    		# for oov words, randomly initialize the word embeddings
    		embeddings[i] = np.array(np.random.uniform(-0.25, 0.25, dim_w), dtype='float32')

    if ctx_mode == 'separate':
    	train_x_l, train_x_r, test_x_l, test_x_r = process_sent(train_sent=train_sent, 
    	test_sent=test_sent, train_target=train_t, 
    	test_target=test_t, vocab=vocab, ds_name=dataset_name, 
    	lexicon=lexicon)
    	n_train = len(train_x_r)

    	n_test = len(test_x_l)

    	train_tuples = list(zip(train_x_l, train_x_r, train_x_t, train_y))
    	test_tuples = list(zip(test_x_l, test_x_r, test_x_t, test_y))

	    # pad and shuffle the training set
    	train_x_l, train_x_r, train_x_t, train_y = pad_dataset(data_tuples=train_tuples, batch_size=batch_size)

	    # pad the testing set to perform batch prediction
    	test_x_l, test_x_r, test_x_t, test_y = pad_dataset(data_tuples=test_tuples, batch_size=batch_size, mode='test')

    	if use_val:
    		train, val = split_train(train_data=[train_x_l, train_x_r, train_x_t], train_label=train_y, batch_size=batch_size)
    		train_x_l, train_x_r, train_x_t, train_y = train
    		val_x_l, val_x_r, val_x_t, val_y = val

    	train_x_l = np.array(train_x_l, dtype='int32')
    	train_x_r = np.array(train_x_r, dtype='int32')
    	train_x_t = np.array(train_x_t, dtype='int32')
    	train_y = np.array(train_y, dtype='int32')

    	if use_val:
    		val_x_l = np.array(val_x_l, dtype='int32')
    		val_x_r = np.array(val_x_r, dtype='int32')
    		val_x_t = np.array(val_x_t, dtype='int32')
    		val_y = np.array(val_y, dtype='int32')
    	else:
    		val_x_l = np.array(np.zeros((batch_size, train_x_l.shape[1])), dtype='int32')
    		val_x_r = np.array(np.zeros((batch_size, train_x_r.shape[1])), dtype='int32')
    		val_x_t = np.array(np.zeros((batch_size, train_x_t.shape[1])), dtype='int32')
    		val_y = np.array(np.zeros(batch_size), dtype='int32')

    	test_x_l = np.array(test_x_l, dtype='int32')
    	test_x_r = np.array(test_x_r, dtype='int32')
    	test_x_t = np.array(test_x_t, dtype='int32')
    	test_y = np.array(test_y, dtype='int32')
    #print(test_x_t.shape)

    	return train_x_l, train_x_r, train_x_t, train_y, \
    	val_x_l, val_x_r, val_x_t, val_y, \
    	test_x_l, test_x_r, test_x_t, test_y, \
    	vocab, embeddings, n_train, n_test

    elif ctx_mode == 'full':
    	train_x, test_x, train_wc, test_wc, train_tpos, test_tpos, train_y_opi, test_y_opi = process_sent2(train_sent=train_sent, 
    		test_sent=test_sent, train_target=train_t, test_target=test_t,
    		vocab=vocab, ds_name=dataset_name, lexicon=lexicon)
    	n_train = len(train_x)
    	n_test = len(test_x)

    	train_tuples = list(zip(train_x, train_x_t, train_y, train_wc, train_tpos, train_y_opi, train_wc_t))
    	test_tuples = list(zip(test_x, test_x_t, test_y, test_wc, test_tpos, test_y_opi, test_wc_t))
    	train_x, train_x_t, train_y, train_wc, train_tpos, train_y_opi, train_wc_t = pad_dataset(data_tuples=train_tuples, batch_size=batch_size)
    	test_x, test_x_t, test_y, test_wc, test_tpos, test_y_opi, test_wc_t = pad_dataset(data_tuples=test_tuples, batch_size=batch_size, mode='test')

    	if use_val:
    		train, val = split_train(train_data=[train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_wc_t], train_label=train_y, batch_size=batch_size)
    		train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_wc_t, train_y = train
    		val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_wc_t, val_y = val
    		
    		train_tuples = list(zip(train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_wc_t, train_y))
    		val_tuples = list(zip(val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_wc_t, val_y))

    		train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_wc_t, train_y = pad_dataset(data_tuples=train_tuples, batch_size=batch_size, mode='train')
    		val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_wc_t, val_y = pad_dataset(data_tuples=val_tuples, batch_size=batch_size, mode='val') 

    	train_data_list = [train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_y, train_wc_t]
    	train_x, train_x_t, train_wc, train_tpos, train_y_opi, train_y, train_wc_t = to_nparray(data_list=train_data_list, dtype='int32')

    	if use_val:
    		val_data_list = [val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_y, val_wc_t]
    		val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_y, val_wc_t = to_nparray(data_list=val_data_list, dtype='int32')
    	else:
    		# build the fake validation set
    		shape_list = [(batch_size, train_x.shape[1]), (batch_size, train_x_t.shape[1]), \
    		(batch_size,), (batch_size,), (batch_size, train_y_opi.shape[1]), (batch_size,), (batch_size,)]
    		val_x, val_x_t, val_wc, val_tpos, val_y_opi, val_y, val_wc_t = init_ndarray_zero(shape_list=shape_list, dtype='int32')

    	test_data_list = [test_x, test_x_t, test_wc, test_tpos, test_y_opi, test_y, test_wc_t]
    	test_x, test_x_t, test_wc, test_tpos, test_y_opi, test_y, test_wc_t = to_nparray(data_list=test_data_list, dtype='int32')

    	return train_x, train_x_t, train_y, \
    	val_x, val_x_t, val_y, \
    	test_x, test_x_t, test_y, \
    	vocab, embeddings, n_train, n_test, \
    	train_wc, train_tpos, \
    	val_wc, val_tpos, \
    	test_wc, test_tpos, \
    	train_y_opi, val_y_opi, test_y_opi, \
    	train_wc_t, val_wc_t, test_wc_t