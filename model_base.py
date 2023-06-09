import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoModel, AutoTokenizer

class Document(object):
	def __init__(self, sents, labels):
		assert(len(sents) == len(labels))
		self.sents = sents
		self.labels = labels
		# eval_labels: 1 if indent, else 0
		self.eval_labels = torch.clone(labels)
		self.eval_labels[labels == 2] = 0

class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model, dtype=torch.float32)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.dropout(x + self.pe[:x.size(0)])

class Transformer(nn.Module):
	def __init__(self, device, d_model, nhead, dim_feedforward, num_layers, window_size, pred_cutoff):
		super().__init__()

		self.device = device
		# self.tokenizer = tokenizer
		# self.sent_embr = sent_embr
		self.d_model = d_model
		self.window_size = window_size
		self.pred_cutoff = pred_cutoff

		self.pos_embeddings = PositionalEncoding(d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dtype=torch.float32)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.decoder = nn.Linear(d_model, 3, dtype=torch.float32)
		self.log_softmax = nn.LogSoftmax(dim=-1)
	
	def forward(self, X):
		rep = self.pos_embeddings(X)
		# ninfs = torch.ones(rep.shape[0], rep.shape[0], dtype=torch.float64) * float('-inf')
		# mask = torch.add(torch.triu(ninfs, diagonal=self.window_size), torch.tril(ninfs, diagonal=-self.window_size)).to(self.device)
		# rep = self.transformer_encoder(rep, mask=mask)
		rep = self.transformer_encoder(rep)
		rep = self.decoder(rep)
		rep = self.log_softmax(rep)
		return rep

	"""
	def predict(self, doc):
		inputs = self.tokenizer(doc, padding=True, truncation=True, return_tensors='pt').to(self.device)
		with torch.no_grad():
			X = self.sent_embr(**inputs).pooler_output
		probs = torch.exp(torch.squeeze(self.forward(torch.unsqueeze(X, 1)), 1).cpu())

		n = probs.shape[0]
		pred = torch.zeros(n)
		pred[0] = 1
		for i in range(1, n-1):
			pred[i] = 1 if probs[i-1][2] * probs[i][1] > self.pred_cutoff else 0
		pred[n-1] = 0
		return pred
	"""

class SentenceEmbedder(nn.Module):
	def __init__(self, device, tokenizer, sent_embr):
		super().__init__()

		self.device = device
		self.tokenizer = tokenizer
		self.sent_embr = sent_embr
	
	def forward(self, X):
		inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
		with torch.no_grad():
			res = self.sent_embr(**inputs).pooler_output.to(self.device)
		return res

def train_model(args, train_data, dev_data):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device', device)

	tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
	sent_embr = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
	sent_embr_model = SentenceEmbedder(device=device, tokenizer=tokenizer, sent_embr=sent_embr).to(device)
	d_model = sent_embr.embeddings.word_embeddings.embedding_dim

	start_time = time.time()
	sent_embs = [sent_embr_model(train_data[i].sents) for i in range(len(train_data))]
	dev_sent_embs = [sent_embr_model(dev_data[i].sents) for i in range(len(dev_data))]
	del sent_embr_model
	dev_labels = torch.LongTensor([label for j in range(len(dev_data)) for label in dev_data[j].eval_labels])
	dev_x = nn.utils.rnn.pad_sequence([dev_sent_embs[j] for j in range(len(dev_data))])
	print('embedded sentences', time.time() - start_time)

	model = Transformer(device=device, d_model=d_model, nhead=4, dim_feedforward=2048, num_layers=3, window_size=5, pred_cutoff=0.2).to(device)
	if args.load_model_path is not None:
		model.load_state_dict(torch.load(args.load_model_path))
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	loss_fn = nn.NLLLoss()
	
	ex_idxs = list(range(len(train_data)))
	n_epochs = 50
	batch_size = 64
	for ep_i in range(n_epochs):
		random.shuffle(ex_idxs)
		epoch_loss = 0
		score = 0
		for i in range(0, len(ex_idxs), batch_size):
			cbatch_size = min(batch_size, len(ex_idxs) - i)
			batch_x = nn.utils.rnn.pad_sequence([sent_embs[ex_idxs[i+j]] for j in range(cbatch_size)])
			# assert(batch_x.device == device)
			log_probs = torch.transpose(model.forward(batch_x).cpu(), 0, 1)
			labels = torch.LongTensor([label for j in range(cbatch_size) for label in train_data[ex_idxs[i+j]].labels])
			tight_log_probs = torch.zeros(labels.shape[0], 3, dtype=torch.float32)
			tlp_i = 0
			for j in range(cbatch_size):
				n = len(train_data[ex_idxs[i+j]].labels)
				tight_log_probs[tlp_i:tlp_i+n] = log_probs[j][:n]
				tlp_i += n
			# assert(tlp_i == labels.shape[0])
			loss = loss_fn(tight_log_probs, labels)

			"""
			preds = torch.argmax(tight_log_probs, 1)
			preds[preds == 2] = 0
			metric = torchmetrics.classification.BinaryF1Score()
			eval_labels = torch.LongTensor([label for j in range(cbatch_size) for label in train_data[ex_idxs[i+j]].eval_labels])
			score = metric(preds, eval_labels)
			# print('f1', metric(preds, labels))
			# print(labels)
			"""

			epoch_loss += loss.item()
			model.zero_grad()
			loss.backward()
			optimizer.step()

			# print('batch %i, loss %f' % (i, loss.item()))

		"""
		model.eval()
		metric = torchmetrics.classification.BinaryF1Score()
		f1_score = 0
		# for ex in dev_data:
		for ex in train_data:
			pred = model.predict(ex.sents)
			f1_score += metric(pred, ex.eval_labels)
		# print('epoch %i, clock %f, loss %f, f1 %f' % (ep_i, time.time() - start_time, epoch_loss, f1_score / len(dev_data)))
		print('epoch %i, clock %f, loss %f, f1 %f' % (ep_i, time.time() - start_time, epoch_loss, f1_score / len(train_data)))
		model.train()
		"""
		model.eval()
		log_probs = torch.transpose(model.forward(dev_x).cpu(), 0, 1)
		tight_log_probs = torch.zeros(len(dev_labels), 3, dtype=torch.float32)
		tlp_i = 0
		for i in range(len(dev_data)):
			n = len(dev_data[i].labels)
			tight_log_probs[tlp_i:tlp_i+n] = log_probs[i][:n]
			tlp_i += n
		assert(tlp_i == len(dev_labels))
		preds = torch.argmax(tight_log_probs, 1)
		preds[0] = 1
		preds[-1] = 2
		preds[preds == 2] = 0
		metric = torchmetrics.classification.BinaryF1Score()
		score = metric(preds, dev_labels)
		model.train()
		print('epoch %i, clock %f, loss %f, f1 %f' % (ep_i, time.time() - start_time, epoch_loss, score))
	
	model.eval()
	return model
