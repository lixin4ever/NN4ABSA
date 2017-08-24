# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import f1_score

def evaluate(pred, gold):
	"""
	evaluate accuracy and macro-F1 of ABSA task
	"""
	pred_count = np.zeros(3, dtype='int32')
	gold_count = np.zeros(3, dtype='int32')
	hit_count = np.zeros(3, dtype='int32')
	
	# number of testing documents
	n_test = len(gold)
	for i in range(n_test):
		y_p = int(pred[i])
		y_g = gold[i]
		pred_count[y_p] += 1
		gold_count[y_g] += 1
		if y_p == y_g:
			hit_count[y_p] += 1
	
	# number of true predictions
	total_hit = sum(hit_count)
	# accuracy
	acc = float(total_hit) / n_test
	# negative: 0, positive: 1, neutral: 2
	r_neg = float(hit_count[0]) / gold_count[0]
	p_neg = float(hit_count[0]) / pred_count[0]
	r_pos = float(hit_count[1]) / gold_count[1]
	p_pos = float(hit_count[1]) / pred_count[1]
	r_neu = float(hit_count[2]) / gold_count[2]
	p_neu = float(hit_count[2]) / pred_count[2]
	# formula for calculation of macro-f1
	# r = (r_neg + r_pos + r_neu) / 3.0
	# p = (p_neg + p_pos + p_neu) / 3.0
	#macro_f = 2 * p * r / (p + r)
	macro_f = f1_score(y_true=gold, y_pred=pred, labels=[0, 1, 2], average='macro')
	result_string = ''
	result_string = '%sneg: recall: %s/%s, precision: %s/%s \n' % (result_string, 
		hit_count[0], gold_count[0], hit_count[0], pred_count[0])
	result_string = '%spos: recall: %s/%s, precision: %s/%s \n' % (result_string, 
		hit_count[1], gold_count[1], hit_count[1], pred_count[1])
	result_string = '%sneu: recall: %s/%s, precision: %s/%s \n' % (result_string, 
		hit_count[2], gold_count[2], hit_count[2], pred_count[2])
	return acc, macro_f, result_string