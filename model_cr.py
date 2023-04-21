import math
import neuralcoref
import numpy as np
import random
import spacy
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
	def __init__(self, device, d_model, nhead, dim_feedforward, num_layers, pred_cutoff):
		super().__init__()

		self.device = device
		self.d_model = d_model
		self.pred_cutoff = pred_cutoff

		self.pos_embeddings = PositionalEncoding(d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dtype=torch.float32)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.decoder = nn.Linear(d_model, 3, dtype=torch.float32)
		self.log_softmax = nn.LogSoftmax(dim=-1)
	
	def forward(self, X):
		rep = self.pos_embeddings(X)
		rep = self.transformer_encoder(rep)
		rep = self.decoder(rep)
		rep = self.log_softmax(rep)
		return rep

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

class CorefResolver():
	def __init__(self):
		self.nlp = spacy.load('en_core_web_sm')
		neuralcoref.add_to_pipe(self.nlp, max_match=30, max_dist_match=50)
		self.tokenizer = self.nlp.tokenizer

	def get_resolved(self, sents):
		text = ' '.join(sents)

		tokens = self.tokenizer(text)
		sent_i, sent_p = 0, 0
		sent_ends = [0] * len(sents)
		for token_i, token in enumerate(tokens):
			while token.text != sents[sent_i][sent_p : sent_p + len(token.text)]:
				sent_p += 1
				if sent_p + len(token.text) > len(sents[sent_i]):
					sent_i += 1
					assert(sent_i <= len(sents))
					sent_p = 0
			sent_p += len(token.text)
			sent_ends[sent_i] = max(sent_ends[sent_i], token_i)

		doc = self.nlp(text)
		repl_dict = dict() # start token idx -> (end token idx, replcr)
		for clust in doc._.coref_clusters:
			replcr = ' '.join([t.text for t in clust.main])
			for mention in clust.mentions:
				repl_dict[mention.start] = (mention.end, replcr)
		res_sents = []
		token_i = 0
		for i in range(len(sents)):
			res_sent = []
			while token_i <= sent_ends[i]:
				if token_i in repl_dict:
					end_i, replcr = repl_dict[token_i]
					res_sent.append(replcr)
					token_i = end_i
				else:
					res_sent.append(tokens[token_i].text)
					token_i += 1
			res_sents.append(' '.join(res_sent))
		return res_sents

def train_model(args, train_data, dev_data):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device', device)

	start_time = time.time()
	coref_resolver = CorefResolver()
	for ex in train_data:
		ex.sents = coref_resolver.get_resolved(ex.sents)
	for ex in dev_data:
		ex.sents = coref_resolver.get_resolved(ex.sents)
	del coref_resolver
	print('resolved coref', time.time() - start_time)

	tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
	sent_embr = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
	sent_embr_model = SentenceEmbedder(device=device, tokenizer=tokenizer, sent_embr=sent_embr).to(device)
	d_model = sent_embr.embeddings.word_embeddings.embedding_dim

	sent_embs = [sent_embr_model(train_data[i].sents) for i in range(len(train_data))]
	dev_x = nn.utils.rnn.pad_sequence([sent_embr_model(dev_data[i].sents) for i in range(len(dev_data))])
	del sent_embr_model
	print('embedded sentences', time.time() - start_time)
	dev_labels = torch.LongTensor([label for j in range(len(dev_data)) for label in dev_data[j].eval_labels])

	model = Transformer(device=device, d_model=d_model, nhead=4, dim_feedforward=2048, num_layers=3, pred_cutoff=0.2).to(device)
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
			max_sents_len = max([len(train_data[ex_idxs[i+j]].sents) for j in range(cbatch_size)])
			batch_coref_scores = torch.zeros(cbatch_size, max_sents_len, max_sents_len).to(device)
			log_probs = torch.transpose(model.forward(batch_x).cpu(), 0, 1)
			labels = torch.LongTensor([label for j in range(cbatch_size) for label in train_data[ex_idxs[i+j]].labels])
			tight_log_probs = torch.zeros(labels.shape[0], 3, dtype=torch.float32)
			tlp_i = 0
			for j in range(cbatch_size):
				n = len(train_data[ex_idxs[i+j]].labels)
				tight_log_probs[tlp_i:tlp_i+n] = log_probs[j][:n]
				tlp_i += n
			loss = loss_fn(tight_log_probs, labels)

			epoch_loss += loss.item()
			model.zero_grad()
			loss.backward()
			optimizer.step()

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
