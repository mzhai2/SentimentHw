from collections import defaultdict
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from keras.utils import np_utils
splits = ['train', 'test']

print('loading embeddings')
# embedding = Word2Vec.load_word2vec_format('/Users/Mike/Documents/data/GoogleNews-vectors-negative300.bin', binary=True)
embedding = Word2Vec.load_word2vec_format('/Users/Mike/Documents/data/wiki_nyt.skip.word.100.vectors.bin', binary=True)
# nyt
# 18231 1879

dim = 100
word_set = set()
labels = []
maxLength = 0
num_sentences = [0,0]
print('begin parse')
for i,split in enumerate(splits):
	with open('./data/' + split + '.tsv', 'r') as f:
		for line in f.readlines():
			if line == '\n': # for last line
				continue
			line = line.split('\t')
			sentence = line[2].split(' ')
			if split != 'test':
				for word in sentence:
					word_set.update([word])
			if maxLength < len(sentence):
				maxLength = len(sentence)
			num_sentences[i] += 1
		print('num sentences',split,num_sentences[i])
def default_word():
	return 'UNK'
def default_vocab():
	return 2
word2vocab = defaultdict(default_vocab)
vocab2word = defaultdict(default_word)
vocab_size = 3
known = 0
unknown = 0

vocab2word[0] = ''
vocab2word[1] = '<END>'
vocab2word[2] = '<UNK>'

word2vocab[''] = 0
word2vocab['<END>'] = 1
word2vocab['<UNK>'] = 2

newEmbedding = [[0]*dim,[0]*dim,[0]*dim] # empty, end, unknown
for word in word_set:
	if word in embedding:
		newEmbedding.append(embedding[word])
	else:
		newEmbedding.append([0]*dim)
		unknown += 1
	word2vocab[word] = vocab_size
	vocab2word[vocab_size] = word
	vocab_size +=1

print vocab_size,unknown
with open('./data/embedding.pickle', 'wb') as f:
	pickle.dump([np.asarray(newEmbedding,np.float32),word2vocab,vocab2word,vocab_size],f,protocol=2)

for i,split in enumerate(splits):
	sentences = np.zeros((num_sentences[i],maxLength),dtype=np.int32)
	masks = np.ones((num_sentences[i],maxLength),dtype=np.int32)
	labels = np.zeros(num_sentences[i],dtype=np.int32)
	with open('./data/' + split + '.tsv', 'r') as f:
		for sentence_count, line in enumerate(f.readlines()):
			if line == '\n': # for last line
				continue
			line = line.split('\t')
			if split in split[:1]:
				labels[sentence_count] = int(line[3])
			for idx, word in enumerate(line[2].split(' ')):
				sentences[sentence_count,idx] = word2vocab[word]
			masks[sentence_count,idx+1:] = 0
	with open('./data/' + split + '.npz', 'w') as f:
		np.savez(f,sentences=sentences, masks=masks, labels=labels)

print('Max Sentence Length: ' + str(maxLength))
print('Recognized Words: ' + str(vocab_size-unknown))
print('Unrecognized Words: ' + str(unknown))
