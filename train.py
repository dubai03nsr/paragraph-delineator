import argparse
from datasets import load_dataset
import nltk
import numpy as np
import random
import re
import time
import torch

from model import *

def _parse_args():
	parser = argparse.ArgumentParser(description='train.py')
	parser.add_argument('--load_model_path', type=str, default=None, help='path to load model from')
	parser.add_argument('--save_model_path', type=str, default='model.pt', help='path to save model to')
	parser.add_argument('--n_train_ex', type=int, default=2, help='number of training examples')
	parser.add_argument('--n_dev_ex', type=int, default=1, help='number of devset examples')
	return parser.parse_args()

def read_data(n_ex, train: bool):
	# dataset = load_dataset('wikipedia', name='20220301.simple', split = f'train[:{n_ex}]' if train else f'train[{-n_ex-1}:-1]')
	dataset = load_dataset('wikipedia', name='20220301.simple', split = f'train[:{n_ex}]' if train else f'train[5000:{5000+n_ex}]')
	data = [None] * n_ex
	for ex_i in range(n_ex):
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
		print(len(sents), 'sentences')
		data[ex_i] = Document(sents, labels)

		"""
		sents = np.array(nltk.tokenize.sent_tokenize(text))
		para_end, para_beg = np.empty(len(sents), dtype=bool), np.empty(len(sents), dtype=bool)
		
		text_i, sent_i = 0, 0
		for sent_i in range(len(sents)):
			assert(text[text_i:text_i+len(sents[sent_i])] == sents[sent_i])
			text_i += len(sents[sent_i])
			assert(text_i <= len(text))
			n_spaces = re.match('\\s*', text[text_i:text_i+5]).span(0)[1]
			assert(n_spaces < 5)
			assert(text_i == len(text) or (text[text_i] == '\n') == (text[text_i+1] == '\n')), f'{text_i} {text[text_i:text_i+20]}\n{repr(text)}'
			para_end[sent_i] = text_i == len(text) or text[text_i] == '\n'
			para_beg[sent_i + 1] = para_end[sent_i] # overflow makes para_beg[0] = True
			text_i += n_spaces
		assert(text_i == len(text))
		
		labels = np.array([(1 if para_beg[i] else 0) + (2 if para_end[i] else 0) for i in range(len(sents))]) # 1 = beg; 2 = end; 3 = both; 0 = neither
		mask = labels != 3 # remove single-sentence paragraphs
		sents, labels = sents[mask], labels[mask]
		data[ex_i] = Document(sents, labels)
		"""
	return data

if __name__ == '__main__':
	args = _parse_args()
	print(args)

	random.seed(17)
	torch.random.manual_seed(17)

	nltk.download('punkt')
	train_data = read_data(args.n_train_ex, train=True)
	dev_data = read_data(args.n_dev_ex, train=False)

	model = train_model(args, train_data, dev_data)

	torch.save(model.state_dict(), args.save_model_path)
