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
from parseRaw import default_word, default_vocab

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
	def read_model_data(self):
		print('Loading Model')
		modelFile = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
		if os.path.isfile(modelFile):
			with open(modelFile, 'rb') as f:
				self.start_epoch, self.best_loss, self.best_bias, self.best_f1, params = pickle.load(f)
			self.end_epoch += self.start_epoch
			lasagne.layers.set_all_param_values(self.model,params)
	def write_model_data(self):
		filename = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
		with open(filename, 'wb') as f:
			pickle.dump((self.epoch, self.best_loss, self.best_bias, self.best_f1, lasagne.layers.get_all_param_values(self.model)), f, protocol=-1)

	@staticmethod
	def toOrigScale(array):
		return np.around(5*np.asarray(array,dtype=np.float32)).astype(np.int32)
	def score(self,pred,labels):
		prec = precision_score(self.toOrigScale(labels),self.toOrigScale(pred), average='micro')
		rec = recall_score(self.toOrigScale(labels),self.toOrigScale(pred), average='micro')
		f1 = 2*prec*rec/(prec+rec)
		return prec,rec,f1
	def find_best_threshold(self,scores,test):
		best_f1 = 0
		best_bias = 0
		if (test==True):
			best_pred = np.rint(scores+self.best_bias)
			best_pred[best_pred > 1] = 1
			best_bias = self.best_bias
		else:
			for bias in np.arange(0,5,0.20):
				pred = scores+bias
				prec,rec,f1 = self.score(pred, self.labels[1])
				if f1 > best_f1:
					best_f1 = f1
					best_pred = np.copy(pred)
					best_bias = bias
		return best_pred,best_bias

	def build_model(self):
		print("Building model and compiling functions...")
		self.sentences_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.sentences[0].shape[1:], dtype=np.int32), borrow=True)
		self.masks_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.masks[0].shape[1:], dtype=np.int8), borrow=True)
		self.labels_macro_batch = theano.shared(np.empty((self.macro_batch_size,) + self.labels[0].shape[1:], dtype=theano.config.floatX), borrow=True)

		sentences_in = T.imatrix('sentences')
		masks_in = T.bmatrix('masks')
		labels_in = T.fvector('labels')
		i = T.iscalar()
		
		flattened = self.define_layers(sentences_in,masks_in)

		self.model = flattened
		prediction = T.clip(lasagne.layers.get_output(flattened),1.0e-7, 1.0 - 1.0e-7)
		test_prediction = T.clip(lasagne.layers.get_output(flattened, deterministic=True), 1.0e-7, 1.0 - 1.0e-7)

		loss,test_loss = self.define_losses(prediction,test_prediction,labels_in)

		params = lasagne.layers.get_all_params(flattened, trainable=True)
		updates = lasagne.updates.adadelta(loss, params)
		
		self.train_fn = theano.function([i], [loss, prediction], updates=updates,
			givens={
			sentences_in: self.sentences_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			masks_in: self.masks_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			labels_in: self.labels_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size]})
		self.train_rest_fn = theano.function([i], [loss, prediction], updates=updates,
			givens={
			sentences_in: self.sentences_macro_batch[:i],
			masks_in: self.masks_macro_batch[:i],
			labels_in: self.labels_macro_batch[:i]})
		self.test_fn = theano.function([i], [test_loss, test_prediction],
			givens={
			sentences_in: self.sentences_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			masks_in: self.masks_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size],
			labels_in: self.labels_macro_batch[i * self.micro_batch_size:(i + 1) * self.micro_batch_size]})
		self.test_rest_fn = theano.function([i], [test_loss, test_prediction],
			givens={
			sentences_in: self.sentences_macro_batch[:i],
			masks_in: self.masks_macro_batch[:i],
			labels_in: self.labels_macro_batch[:i]})
	def define_layers(self,sentences_in,masks_in):
		N_HIDDEN = 200
		GRAD_CLIP = 10
		GRAD_STEPS = 40
		MAX_LENGTH = 56
		sentence_layer = lasagne.layers.InputLayer(shape=(None,MAX_LENGTH), input_var=sentences_in)
		embedding_layer = lasagne.layers.EmbeddingLayer(sentence_layer, input_size=self.vocab_size, output_size=100, W=self.embedding)
		
		conv_layer = lasagne.layers.Conv1DLayer(embedding_layer,MAX_LENGTH,3,pad='same')
		
		# mask_layer = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH), input_var=masks_in)
		
		resetgate = lasagne.layers.Gate(W_cell=None)
		updategate = lasagne.layers.Gate(W_cell=None)
		hidden_update = lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
		forward_layer = lasagne.layers.GRULayer(conv_layer, N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
		backward_layer = lasagne.layers.GRULayer(conv_layer, N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
		
		# forward_layer = lasagne.layers.GRULayer(conv_layer, N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=mask_layer, only_return_final=True)
		# backward_layer = lasagne.layers.GRULayer(conv_layer, N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=mask_layer, only_return_final=True, backwards=True)
		
		sum_layer = lasagne.layers.ElemwiseSumLayer([forward_layer,backward_layer])
		output_layer = lasagne.layers.DenseLayer(sum_layer, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
		
		flattened = lasagne.layers.FlattenLayer(output_layer, outdim=1)
		return flattened

	def define_losses(self, prediction, test_prediction, l_in):
		loss = lasagne.objectives.binary_crossentropy(prediction, l_in).mean()
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction, l_in).mean()
		return loss, test_loss

	def set_all(self, i, macro_batch_index):
		self.sentences_macro_batch.set_value(self.sentences[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
		self.masks_macro_batch.set_value(self.masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
		self.labels_macro_batch.set_value(self.labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size], borrow=True)
	def set_all_rest(self, i, idx):
		self.sentences_macro_batch.set_value(np.lib.pad(self.sentences[i][-idx:], [(0,self.macro_batch_size-idx),(0,0)],'constant'), borrow=True)
		self.masks_macro_batch.set_value(np.lib.pad(self.masks[i][-idx:], [(0,self.macro_batch_size-idx),(0,0)],'constant'), borrow=True)
		self.labels_macro_batch.set_value(np.lib.pad(self.labels[i][-idx:], [(0,self.macro_batch_size-idx)],'constant'), borrow=True)

	def train_model(self):
		print('Starting Train')
		for epoch in xrange(self.start_epoch,self.end_epoch):
			self.epoch = epoch

			train_loss = 0
			dev_loss = 0

			train_pred = []
			dev_pred = []

			macro_batch_count = self.sentences[0].shape[0] // self.macro_batch_size
			micro_batch_count = self.macro_batch_size // self.micro_batch_size
			for macro_batch_index in xrange(macro_batch_count):
				self.set_all(0, macro_batch_index)
				for micro_batch_index in xrange(micro_batch_count):
					tl, tp = self.train_fn(micro_batch_index)
					train_loss += tl
					train_pred.extend(tp)
			if self.sentences[0].shape[0] % self.macro_batch_size != 0:
				idx = self.sentences[0].shape[0]%self.macro_batch_size
				self.set_all_rest(0, idx)
				tl, tp = self.train_rest_fn(idx)
				train_loss += tl
				train_pred.extend(tp)

			macro_batch_count = self.sentences[1].shape[0] // self.macro_batch_size
			micro_batch_count = self.macro_batch_size // self.micro_batch_size
			for macro_batch_index in xrange(macro_batch_count):
				self.set_all(1, macro_batch_index)
				for micro_batch_index in xrange(micro_batch_count):
					dl, dp = self.test_fn(micro_batch_index)
					dev_loss += dl
					dev_pred.extend(dp)
			if self.sentences[1].shape[0] % self.macro_batch_size != 0:
				idx = self.sentences[1].shape[0]%self.macro_batch_size
				self.set_all_rest(1, idx)
				dl, dp = self.test_rest_fn(idx)
				dev_loss += dl
				dev_pred.extend(dp)
			
			train_loss/=self.sentences[0].shape[0]
			dev_loss/=self.sentences[1].shape[0]

			dev_pred, dev_bias = self.find_best_threshold(dev_pred,False)

			train_prec,train_rec,train_f1 = self.score(train_pred,self.labels[0])
			dev_prec,dev_rec,dev_f1 = self.score(dev_pred,self.labels[1])
			self.save_best(dev_loss, dev_bias, dev_f1)
			print('Epoch',self.epoch+1,'T-L:',train_loss,'T-F1:',train_f1,'D-L:',dev_loss,'D-P', dev_prec, 'D-R', dev_rec, 'D-F1',dev_f1,'Bias',dev_bias, 'Best Loss', self.best_loss, 'Best D-F1', self.best_f1)

	def evaluate_model(self):
		print('Starting Test')
		test_loss = 0
		test_pred = []

		macro_batch_count = self.sentences[2].shape[0] // self.macro_batch_size
		micro_batch_count = self.macro_batch_size // self.micro_batch_size
		for macro_batch_index in xrange(macro_batch_count):
			self.set_all(2, macro_batch_index)
			for micro_batch_index in xrange(micro_batch_count):
				tl, tp = self.test_fn(micro_batch_index)
				test_loss += tl
				test_pred.extend(tp)
		if self.sentences[2].shape[0] % self.macro_batch_size != 0:
			idx = self.sentences[2].shape[0]%self.macro_batch_size
			self.set_all_rest(2, idx)
			tl, tp = self.test_rest_fn(idx)
			test_loss += tl
			test_pred.extend(tp)

		test_loss/=self.sentences[2].shape[0]
		test_pred, test_bias = self.find_best_threshold(test_pred,True)

		test_prec,test_rec,test_f1 = self.score(test_pred,self.labels[2])

		print('Test-L:',test_loss,'Test-P', test_prec, 'Test-R', test_rec, 'Test-F1', test_f1, 'Bias',self.best_bias)
	
	def load_data(self):
		sentences = []
		masks = []
		labels = []
		splits = ['train', 'test']

		with open('./data/embedding.pickle', 'rb') as f:
			self.embedding,self.word2vocab,self.vocab2word,self.vocab_size = pickle.load(f)

		for split in splits:
			filepath = os.path.join('./data/', '%s.%s' % (split, 'npz'))
			data = np.load(filepath)
			sentences.append(data['sentences'].astype('int32'))
			masks.append(data['masks'].astype('int8'))
			labels.append(data['labels'].astype(theano.config.floatX)/5)
			data.close()

		dev_split_idx = sentences[0].shape[0]//10
		sentences.insert(1,sentences[0][:dev_split_idx])
		sentences[0] = sentences[0][dev_split_idx:]
		masks.insert(1,masks[0][:dev_split_idx])
		masks[0] = masks[0][dev_split_idx:]
		labels.insert(1,labels[0][:dev_split_idx])
		labels[0] = labels[0][dev_split_idx:]

		self.sentences = sentences
		self.labels = labels
		self.masks = masks

def main():
	lstm = Base('bidir_context_convolutional_lstm',2000,200,35)
	lstm.run()
if __name__ == '__main__':
	main()