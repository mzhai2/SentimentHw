import numpy as np
import theano
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers.base import Layer
from lasagne.random import get_rng
from sklearn.metrics import recall_score, precision_score
import random
import math
import cPickle as pickle
from collections import defaultdict
import os
from parseRaw import *

seed = 1
random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))
# theano.config.compute_test_value = 'warn'

class GaussianDropoutLayer(Layer):
	def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
		super(GaussianDropoutLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(seed=np.random.randint(10e6))
		self.p = p
		self.rescale = rescale

	def get_output_for(self, input, deterministic=False, **kwargs):
		if deterministic or self.p == 0:
			return input
		else:
			retain_prob = 1 - self.p
			if self.rescale:
				input /= retain_prob

			input_shape = self.input_shape
			if any(s is None for s in input_shape):
				input_shape = input.shape
			input *= self._srng.normal(size=input_shape, avg=1.0, std=T.sqrt(self.p / (1.0 - self.p)), dtype=theano.config.floatX)
		return input

class Base(object):
	def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
		self.end_epoch = end_epoch
		self.start_epoch = 0
		self.macro_batch_size = macro_batch_size
		self.micro_batch_size = micro_batch_size
		self.best_loss = 100
		self.best_bias = 0
		self.best_f1 = 0
		self.model = None
		self.name = name

	def run(self):
		self.load_data()
		self.build_model()
		self.read_model_data()
		self.train_model()
		self.read_model_data()
		self.evaluate_model()
	def save_best(self,loss,bias,f1):
		if f1 > self.best_f1:
			self.best_bias = bias
			self.best_f1 = f1
			self.write_model_data()
			self.best_loss = loss
		# if loss < self.best_loss:
			# self.write_model_data()
	def read_model_data(self,load_embedding):
		print('Loading Model')
		embeddingFile = os.path.join('./data/', 'embedding.params')
		modelFile = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
		if load_embedding:
			lasagne.layers.set_all_param_values(self.embedding,embedding)
		if os.path.isfile(modelFile):
			with open(filename, 'r') as f:
				self.start_epoch, loss, bias, f1, params = pickle.load(f)
			self.end_epoch += self.start_epoch
			lasagne.layers.set_all_param_values(self.model,params)
			self.best_loss = loss
			self.best_bias = bias
			self.best_f1 = f1
	def write_model_data(self):
		filename = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
		with open(filename, 'w') as f:
			pickle.dump((self.epoch, self.best_loss, self.best_bias, self.best_f1, lasagne.layers.get_all_param_values(self.model)), f)

	def find_best_threshold(self,scores,test):
		best_f1 = 0
		best_bias = 0
		if (test==True):
			best_pred = np.rint(scores+self.best_bias)
			best_pred[best_pred > 1] = 1
			best_bias = self.best_bias
		else:
			for bias in np.arange(0,5,0.20):
				rb_pred = np.around(scores+bias)
				rb_prec = precision_score(self.labels[1].astype(np.int32,copy=False),rb_pred)
				rb_rec = recall_score(self.labels[1].astype(np.int32,copy=False),rb_pred)
				rb_f1 = 2*rb_prec*rb_rec/(rb_prec+rb_rec)
				if rb_f1 > best_f1:
					best_f1 = rb_f1
					best_pred = rb_pred
					best_bias = bias
		return best_pred.astype(np.int32,copy=False),best_bias

	def build_model(self):
		print("Building model and compiling functions...")
		self.sentences_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.sentences[0].shape[1:], dtype=theano.config.floatX), borrow=True)
		self.masks_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.masks[0].shape[1:], dtype=theano.config.floatX), borrow=True)
		self.labels_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.labels[0].shape[1:], dtype=theano.config.floatX), borrow=True)

		sentences_in = T.ftensor3('sentences')
		masks_in = T.fmatrix('masks')
		labels_in = T.fvector('labels')
		i = T.iscalar()
		
		flattened = self.define_layers(sentences_in,masks_in,embedding)

		self.model = flattened
		prediction = T.clip(5*lasagne.layers.get_output(flattened),1.0e-7, 1.0 - 1.0e-7)
		test_prediction = T.clip(5*lasagne.layers.get_output(flattened, deterministic=True), 1.0e-7, 1.0 - 1.0e-7)

		loss,test_loss = self.define_losses(prediction,test_prediction,labels_in)

		params = lasagne.layers.get_all_params(flattened, trainable=True)
		updates = lasagne.updates.adadelta(loss, params)
		
		self.train_fn = theano.function([i], [loss, T.cast(T.round(prediction),'int32'), prediction], updates=updates, allow_input_downcast=True,
			givens={
			sentences_in: self.sentences_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			masks_in: self.masks_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			labels_in: self.labels_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size]})
		self.train_rest_fn = theano.function([i], [loss, T.cast(T.round(prediction),'int32'), prediction], updates=updates, allow_input_downcast=True,
			givens={
			sentences_in: self.sentences_macro_batch[:i],
			masks_in: self.masks_macro_batch[:i],
			labels_in: self.labels_macro_batch[:i]})
		self.test_fn = theano.function([i], [test_loss, T.cast(T.round(test_prediction),'int32'), test_prediction], allow_input_downcast=True,
			givens={
			sentences_in: self.sentences_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			masks_in: self.masks_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			labels_in: self.labels_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size]})
		self.test_rest_fn = theano.function([i], [test_loss, T.cast(T.round(test_prediction),'int32'), test_prediction], allow_input_downcast=True,
			givens={
			sentences_in: self.sentences_macro_batch[:i],
			masks_in: self.masks_macro_batch[:i],
			labels_in: self.labels_macro_batch[:i]})
	def define_layers(self,sentences_in,masks_in, embedding):
		Q_N_HIDDEN = 40
		C_N_HIDDEN = 200
		GRAD_CLIP = 10
		GRAD_STEPS = 40

		sentence_layer = lasagne.layers.InputLayer(shape=(None, 56, 1), input_var=sentences_in)
		embbedding_layer = lasagne.layers.EmbeddingLayer(sentence_layer, input_size=self.vocab_size, output_size=300, W=embedding)
		conv_layer =lasagne.layers.Conv1DLayer(embbedding_layer,56,3,pad='same')
		mask_layer = lasagne.layers.InputLayer(shape=(None, 56), input_var=masks_in)
		resetgate = lasagne.layers.Gate(W_cell=None)
		updategate = lasagne.layers.Gate(W_cell=None)
		hidden_update = lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
		forward_layer = lasagne.layers.GRULayer(conv, C_N_HIDDEN, resetgate=c_resetgate, updategate=c_updategate, hidden_update=c_hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=mask_layer, only_return_final=True)
		backward_layer = lasagne.layers.GRULayer(conv, C_N_HIDDEN, resetgate=c_resetgate, updategate=c_updategate, hidden_update=c_hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=mask_layer, only_return_final=True, backwards=True)
		sum_layer = lasagne.layers.ElemwiseSumLayer([forward_layer,backward_layer])
		output_layer = lasagne.layers.DenseLayer(sum_layer, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

		# concat_layer = lasagne.layers.ConcatLayer([q_forward_1,c_sum])
		# output_layer = lasagne.layers.DenseLayer(concat_layer, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
		
		flattened = lasagne.layers.FlattenLayer(output_layer, outdim=1)
		return flattened

	def define_losses(self, prediction, test_prediction, l_in):
		loss = lasagne.objectives.squared_error(prediction, l_in).mean()
		test_loss = lasagne.objectives.squared_error(test_prediction, l_in).mean()
		return loss, test_loss

	def set_all(self, i, macro_batch_index):
		self.c_macro_batch.set_value(self.sentences[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
		self.c_mask_macro_batch.set_value(self.masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
		self.l_macro_batch.set_value(self.labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
	def set_all_rest(self, i, idx):
		self.c_macro_batch.set_value(np.lib.pad(self.sentences[i][-idx:], [(0,self.macro_batch_size-idx),(0,0),(0,0)],'constant'), borrow=True)
		self.c_mask_macro_batch.set_value(np.lib.pad(self.masks[i][-idx:], [(0,self.macro_batch_size-idx),(0,0)],'constant'), borrow=True)
		self.l_macro_batch.set_value(np.lib.pad(self.labels[i][-idx:], [(0,self.macro_batch_size-idx)],'constant'), borrow=True)

	def train_model(self):
		print('Starting Train')
		for epoch in xrange(self.start_epoch,self.end_epoch):
			self.epoch = epoch

			train_loss = 0
			dev_loss = 0

			train_pred = []
			dev_pred = []

			train_scores =[]
			dev_scores = []

			macro_batch_count = self.sentences[0].shape[0] // self.macro_batch_size
			micro_batch_count = self.macro_batch_size // self.micro_batch_size
			for macro_batch_index in xrange(macro_batch_count):
				self.set_all(0, macro_batch_index)
				for micro_batch_index in xrange(micro_batch_count):
					tl, tp, ts = self.train_fn(micro_batch_index)
					train_loss += tl
					train_pred.extend(tp)
					train_scores.extend(ts)
			if self.sentences[0].shape[0] % self.macro_batch_size != 0:
				idx = self.sentences[0].shape[0]%self.macro_batch_size
				self.set_all_rest(0, idx)
				tl, tp, ts = self.train_rest_fn(idx)
				train_loss += tl
				train_pred.extend(tp)
				train_scores.extend(ts)

			macro_batch_count = self.sentences[1].shape[0] // self.macro_batch_size
			micro_batch_count = self.macro_batch_size // self.micro_batch_size
			for macro_batch_index in xrange(macro_batch_count):
				self.set_all(1, macro_batch_index)
				for micro_batch_index in xrange(micro_batch_count):
					dl, dp, ds = self.test_fn(micro_batch_index)
					dev_loss += dl
					dev_pred.extend(dp)
					dev_scores.extend(ds)
			if self.sentences[1].shape[0] % self.macro_batch_size != 0:
				idx = self.sentences[1].shape[0]%self.macro_batch_size
				self.set_all_rest(1, idx)
				dl, dp, ds = self.test_rest_fn(idx)
				dev_loss += dl
				dev_pred.extend(dp)
				dev_scores.extend(ds)
			
			train_loss/=self.sentences[0].shape[0]
			dev_loss/=self.sentences[1].shape[0]

			train_pred = np.asarray(train_pred,dtype=np.int32)
			train_scores = np.asarray(train_scores,dtype=np.float32)

			dev_pred = np.asarray(dev_pred,dtype=np.int32)
			dev_scores = np.asarray(dev_scores,dtype=np.float32)

			prec = precision_score(self.labels[1].astype(np.int32,copy=False),dev_pred)
			rec = recall_score(self.labels[1].astype(np.int32,copy=False),dev_pred)
			f1 = 2*prec*rec/(prec+rec)

			rb_dev_pred, dev_bias = self.find_best_threshold(dev_scores,False)
			rb_prec = precision_score(self.labels[1].astype(np.int32,copy=False),rb_dev_pred)
			rb_rec = recall_score(self.labels[1].astype(np.int32,copy=False),rb_dev_pred)
			rb_f1 = 2*rb_prec*rb_rec/(rb_prec+rb_rec)

			self.save_best(dev_loss, dev_bias, rb_f1)
			print('Epoch',self.epoch+1,'T-L:',train_loss,'D-L:',dev_loss,'D-P', rb_prec, 'D-R', rb_rec, 'D-F1',rb_f1,'Bias',dev_bias, 'Best Loss', self.best_loss, 'Best D-F1', self.best_f1)

	def evaluate_model(self):
		print('Starting Test')
		test_loss = 0
		test_pred = []
		test_scores = []

		macro_batch_count = self.sentences[2].shape[0] // self.macro_batch_size
		micro_batch_count = self.macro_batch_size // self.micro_batch_size
		for macro_batch_index in xrange(macro_batch_count):
			self.set_all(2, macro_batch_index)
			for micro_batch_index in xrange(micro_batch_count):
				tl, tp, ts = self.test_fn(micro_batch_index)
				test_loss += tl
				test_pred.extend(tp)
				test_scores.extend(ts)
		if self.sentences[2].shape[0] % self.macro_batch_size != 0:
			idx = self.sentences[2].shape[0]%self.macro_batch_size
			self.set_all_rest(2, idx)
			tl, tp, ts = self.test_rest_fn(idx)
			test_loss += tl
			test_pred.extend(tp)
			test_scores.extend(ts)

		test_loss/=self.sentences[2].shape[0]
		test_scores = np.asarray(test_scores,dtype=np.float32)
		
		rb_test_pred, _ = self.find_best_threshold(test_scores,True)
		rb_prec = precision_score(self.labels[2].astype(np.int32,copy=False),rb_test_pred)
		rb_rec = recall_score(self.labels[2].astype(np.int32,copy=False),rb_test_pred)
		rb_f1 = 2*rb_prec*rb_rec/(rb_prec+rb_rec)
		print('Test-L:',test_loss,'Test-P', rb_prec, 'Test-R', rb_rec, 'Test-F1',rb_f1, 'Bias',self.best_bias)
	
	def load_data(self):
		sentences = []
		masks = []
		labels = []
		splits = ['train', 'test']

		with open('./data/embedding.pickle', 'rb') as f:
			newEmbedding,word2vocab,vocab2word,vocab_size = pickle.load(f)

		for split in splits:
			filepath = os.path.join('./data/', '%s.%s' % (split, 'npz'))
			data = np.load(filepath)
			sentences.append(data['sentences'])
			masks.append(data['masks'].astype(np.float32))
			labels.append(data['labels'].astype(np.float32))
			data.close()
		
		dev_split_idx = sentences[0].shape[0]//10
		sentences.insert(1,sentences[0,dev_split_idx])
		sentences[0] = sentences[0][dev_split_idx:]
		masks.insert(1,masks[0,dev_split_idx])
		masks[0] = masks[0][dev_split_idx:]
		labels.insert(1,labels[0,dev_split_idx])
		labels[0] = labels[0][dev_split_idx:]


		self.sentences = sentences
		self.labels = labels
		self.masks = mask

def main():
	lstm = Base('bidir_context_convolutional_lstm',2000,200,35)
	lstm.run()
if __name__ == '__main__':
	main()