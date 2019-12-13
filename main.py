
import math

import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
import re

from tqdm import tqdm

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from language_model import LanguageModel
from allennlp.data.dataset_readers import DatasetReader, LanguageModelingReader, SimpleLanguageModelingDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.training.trainer import Trainer
from preprocessing import Preprocessing
from allennlp.predictors import SentenceTaggerPredictor

from metrics import Perplexity

torch.manual_seed(1)

PATH = '../cache/'

preprocessing = Preprocessing(inp_path=PATH+'data.csv', out_path=PATH+'prep_data.csv')

reader = SimpleLanguageModelingDatasetReader()

train_dataset = reader.read(cached_path(PATH + 'prep_data.csv'))


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

vocab = Vocabulary.from_instances(train_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

lstm_model = LanguageModel(contextualizer=lstm, text_field_embedder=word_embeddings,
                           vocab=vocab)
print(vocab.get_vocab_size('tokens'))

optimizer = optim.SGD(lstm_model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=lstm_model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  #validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)

trainer.train()