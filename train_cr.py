import argparse
from datasets import load_dataset
import nltk
import numpy as np
import random
import re
import time
import torch

from model_cr import *

def _parse_args():
	parser = argparse.ArgumentParser(description='train.py')
	parser.add_argument('--load_model_path', type=str, default=None, help='path to load model from')
	parser.add_argument('--save_model_path', type=str, default='model3.pt', help='path to save model to')
	parser.add_argument('--n_ex', type=int, default=100, help='number of examples')
	parser.add_argument('--dev_ratio', type=int, default=0.01, help='ratio of dev to total examples')
	return parser.parse_args()

def read_data(n_ex, dev_ratio):
	nltk.download('punkt')
	# dataset = load_dataset('wikipedia', name='20220301.simple', split = f'train[:{n_ex}]')
	dataset = load_dataset('wikipedia', name='20220301.simple', split='train')
	data = []
	n_sents = 0
	f1 = 0
	for ex_i in range(len(dataset)):
		if len(data) == n_ex: break
		text = dataset[ex_i]['text']
		paras = text.split('\n\n')
		para_sents, labels = [], []
		for para in paras:
			if '\n' in para: continue
			para_sent = nltk.tokenize.sent_tokenize(para)
			if len(para_sent) <= 1: continue
			para_sents.append(para_sent)
			labels.append([1] + [0]*(len(para_sent)-2) + [2])
		sents = [sent for para in para_sents for sent in para]
		labels = torch.LongTensor([b for a in labels for b in a])
		if len(sents) >= 9 and len(para_sents) >= 3:
			data.append(Document(sents, labels))
			n_sents += len(sents)
	print('read %i docs, %i sents' % (len(data), n_sents))

	n_ex = len(data)
	data = np.array(data)
	perm = np.arange(n_ex)
	np.random.shuffle(perm)
	n_dev_ex = int(n_ex * dev_ratio)
	train_data = data[perm[n_dev_ex:]]
	dev_data = data[perm[:n_dev_ex]]
	return train_data, dev_data

if __name__ == '__main__':
	start = time.time()

	args = _parse_args()
	print(args)

	random.seed(17)
	np.random.seed(17)
	torch.random.manual_seed(17)

	model = train_model(args, *read_data(args.n_ex, args.dev_ratio))

	if args.save_model_path is not None:
		torch.save(model.state_dict(), args.save_model_path)
	print('done. clock %f' % (time.time() - start))
