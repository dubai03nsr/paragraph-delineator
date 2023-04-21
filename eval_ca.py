import argparse
from datasets import load_dataset
import math
import nltk
import numpy as np
import random
import re
import time
import torch

from model_ca import *

def _parse_args():
	parser = argparse.ArgumentParser(description='train.py')
	parser.add_argument('--load_model_path', type=str, default='model4.pt', help='path to load model from')
	parser.add_argument('--n_ex', type=int, default=100, help='number of examples')
	parser.add_argument('--dev_ratio', type=int, default=0.01, help='ratio of dev to total examples')
	return parser.parse_args()

def read_data(n_ex, dev_ratio):
	nltk.download('punkt')
	dataset = load_dataset('wikipedia', name='20220301.simple', split='train')
	data = []
	n_sents = 0
	n_dev_ex = int(n_ex * dev_ratio)
	n_test_ex = n_dev_ex
	for ex_i in range(len(dataset)):
		if len(data) == n_ex + n_test_ex: break
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

	data = np.array(data)
	perm = np.arange(n_ex)
	np.random.shuffle(perm)
	train_data = data[perm[n_dev_ex:]]
	dev_data = data[perm[:n_dev_ex]]
	test_data = data[-n_test_ex:]
	return train_data, dev_data, test_data

class Evaler():
	def __init__(self, args):
		self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
		self.sent_embr = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
		self.d_model = self.sent_embr.embeddings.word_embeddings.embedding_dim
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		print('device', self.device)
		self.sent_embr_model = SentenceEmbedder(device=self.device, tokenizer=self.tokenizer, sent_embr=self.sent_embr).to(self.device)
		self.model = Transformer(device=self.device, d_model=self.d_model, nhead=4, dim_feedforward=2048, num_layers=3, pred_cutoff=0.2).to(self.device)
		self.model.load_state_dict(torch.load(args.load_model_path))
		self.model.eval()
		self.coref_scorer = CorefScorer()

	def eval(self, data):
		sent_embs = [self.sent_embr_model(data[i].sents) for i in range(len(data))]
		labels = torch.LongTensor([label for j in range(len(data)) for label in data[j].eval_labels])
		x = nn.utils.rnn.pad_sequence([sent_embs[j] for j in range(len(data))])
		max_sents_len = max([len(data[i].sents) for i in range(len(data))])
		coref_scores = torch.zeros(len(data), max_sents_len, max_sents_len).to(self.device)
		for i in range(len(dev_data)):
			pad_len = max_sents_len - len(data[i].sents)
			coref_scores[i] = nn.functional.pad(self.coref_scorer.get_scores(data[i].sents), (0, pad_len, 0, pad_len))
		for i in range(len(data[0].sents)):
			for j in range(len(data[0].sents)):
				print(coref_scores[0, i, j].item(), end=' ')
			print()

		log_probs = torch.transpose(self.model.forward(x, coref_scores).cpu(), 0, 1)
		tight_log_probs = torch.zeros(len(labels), 3, dtype=torch.float32)
		tlp_i = 0
		for i in range(len(data)):
			n = len(data[i].labels)
			tight_log_probs[tlp_i:tlp_i+n] = log_probs[i][:n]
			tlp_i += n
		assert(tlp_i == len(labels))
		preds = torch.argmax(tight_log_probs, 1)
		preds[0] = 1
		preds[-1] = 2
		preds[preds == 2] = 0
		metric = torchmetrics.classification.BinaryF1Score()
		score = metric(preds, labels)
		print(score)

		for i in range(len(data[0].sents)):
			if data[0].labels[i] == 1:
				print('<IND>', end='')
			print(data[0].sents[i])
		print()
		for i in range(len(data[0].sents)):
			if preds[i] == 1:
				print('<IND>', end='')
			print(data[0].sents[i])


if __name__ == '__main__':
	start = time.time()

	args = _parse_args()
	print(args)

	random.seed(17)
	np.random.seed(17)
	torch.random.manual_seed(17)

	_, dev_data, test_data = read_data(args.n_ex, args.dev_ratio)
	evaler = Evaler(args)
	evaler.eval(dev_data)
	evaler.eval(test_data)
