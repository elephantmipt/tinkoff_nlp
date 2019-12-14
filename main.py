import math

import torch
import torch.nn as nn
from itertools import chain
import torch.optim as optim
import warnings
from typing import Dict, Iterable, Union, Optional
from allennlp.data.tokenizers import WordTokenizer, Token
import youtokentome as yttm

import numpy as np
import re

from tqdm import tqdm

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from language_model import LanguageModel
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.dataset_readers import DatasetReader, SimpleLanguageModelingDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, MultiHeadSelfAttention

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from preprocessing import TrainTestSplit

warnings.filterwarnings("ignore")

torch.manual_seed(1)

PATH = '../cache/'

train_test = TrainTestSplit(inp_path=PATH+'data.csv', out_path=PATH+'prep_data.csv',train_path=PATH+'train_data.csv',
                            test_path=PATH+'test_data.csv', bpe_path=PATH+'bpe.model')




class LanguageModelingBpeReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 bpe: bool = False,
                 bpe_model_path: str = '') -> None:
        super().__init__(False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.bpe = bpe
        if bpe:
            self.yttm = yttm.BPE(model=bpe_model_path)
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

    def text_to_instance(self,  # type: ignore
                         sentence: str) -> Instance:
        if self.bpe:

            tokenized = self.yttm.encode(sentence, output_type=yttm.OutputType.SUBWORD)
            tokenized = [Token(x) for x in tokenized]
        else:
            tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        with open(file_path) as file:
            for sentence in file:
                instance = self.text_to_instance(sentence)
                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                    yield instance

reader = LanguageModelingBpeReader(bpe=True, bpe_model_path=PATH+'bpe.model')

train_dataset = reader.read(cached_path(PATH + 'train_data.csv'))
test_dataset = reader.read(cached_path(PATH + 'test_data.csv'))


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

vocab = Vocabulary.from_instances(chain(train_dataset, test_dataset))

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

lstm_model = LanguageModel(contextualizer=lstm, text_field_embedder=word_embeddings,
                           vocab=vocab)

transformer = MultiHeadSelfAttention(attention_dim=16, input_dim=EMBEDDING_DIM, num_heads=8, values_dim=16)
transformer_model = LanguageModel(contextualizer=transformer, text_field_embedder=word_embeddings, vocab=vocab)


optimizer = optim.Adam(transformer_model.parameters())
iterator = BucketIterator(batch_size=16, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=transformer_model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=test_dataset,
                  patience=3,
                  num_epochs=100,
                  serialization_dir='./tb/transformer')

trainer.train()