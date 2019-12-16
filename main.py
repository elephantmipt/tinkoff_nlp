import argparse
import math
import random
import os

import torch
from optim import RAdam
from itertools import chain
import torch.optim as optim
import warnings
from typing import Dict, Iterable, Union, Optional
from allennlp.data.tokenizers import WordTokenizer, Token
import youtokentome as yttm
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper


from allennlp.data import Instance
from allennlp.data.fields import TextField
from language_model import LanguageModel

from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from preprocessing import TrainTestSplit

from torch.optim.lr_scheduler import ReduceLROnPlateau


warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='Language model argument parser')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch', default=8, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'radam'])

parser.add_argument('--arch', default='stacked', type=str, choices=['mhsa' 'lstm', 'stacked'])

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')

parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')

parser.add_argument('--drop', '--dropout', default=0.1, type=float,
                    metavar='Dropout', help='Dropout ratio')


'''parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')'''
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--dataset-path', default='', type=str, metavar='PATH',
                    help='path to dataset and bpe model to save')

parser.add_argument('--serialization-path', default='./tb', type=str, metavar='PATH',
                    help='path for tensorboard and tmp files')

parser.add_argument('--bpe', action='store_true')

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--mistakes-rate', default='0', type=float,
                    help='probability of mistake made by user (after bpe train)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


PATH = args.dataset_path
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
torch.manual_seed(args.manualSeed)




train_test = TrainTestSplit(inp_path=PATH+'data.csv', out_path=PATH+'prep_data.csv',train_path=PATH+'train_data.csv',
                            test_path=PATH+'test_data.csv', bpe_path=PATH+'bpe.model', bpe=args.bpe,
                            mistakes_rate=args.mistakes_rate)


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

reader = LanguageModelingBpeReader(tokenizer=WordTokenizer(), bpe=args.bpe, bpe_model_path=PATH+'bpe.model')

train_dataset = reader.read(cached_path(PATH + 'train_data.csv'))
test_dataset = reader.read(cached_path(PATH + 'test_data.csv'))


EMBEDDING_DIM = 128
HIDDEN_DIM = 128

vocab = Vocabulary.from_instances(chain(train_dataset, test_dataset))

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, dropout=args.drop))

lstm_model = LanguageModel(contextualizer=lstm, text_field_embedder=word_embeddings,
                           vocab=vocab)

transformer = MultiHeadSelfAttention(attention_dim=32, input_dim=EMBEDDING_DIM, num_heads=8,
                                     values_dim=32, attention_dropout_prob=args.drop)

transformer_model = LanguageModel(contextualizer=transformer, text_field_embedder=word_embeddings, vocab=vocab)

stacked_transformer = StackedSelfAttentionEncoder(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=2,
                                                  projection_dim=32, feedforward_hidden_dim=32, num_attention_heads=4)

stacked_transformer_model = LanguageModel(contextualizer=stacked_transformer,
                                                      text_field_embedder=word_embeddings,
                                                      vocab=vocab)
if args.arch == 'mhsa':
    model = transformer_model
elif args.arch == 'lstm':
    model = lstm_model
elif args.arch == 'stacked':
    model = stacked_transformer_model
else:
    raise TypeError

if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer.lower() == 'radam':
    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
else:
    raise TypeError
iterator = BucketIterator(batch_size=args.batch, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)

scheduler = _PyTorchLearningRateSchedulerWrapper(ReduceLROnPlateau(optimizer, patience=4))


if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=test_dataset,
                  patience=10,
                  num_epochs=args.epochs,
                  serialization_dir=args.serialization_path,
                  cuda_device=cuda_device,
                  learning_rate_scheduler=scheduler)

trainer.train()