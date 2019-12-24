import torch
import math
import os
from language_model import LanguageModel
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training import util as training_util
from typing import Dict, Iterable, Union, Optional
from allennlp.data.tokenizers import WordTokenizer, Token
import youtokentome as yttm
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from preprocessing import mistakes_maker
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, MultiHeadSelfAttention
import argparse
import numpy as np
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

parser = argparse.ArgumentParser(description='Language model argument parser')
parser.add_argument('--bpe', action='store_true')
parser.add_argument('--mistakes-rate', default=0.01, type=float,
                    help='mistakes rate for data')
parser.add_argument('--model-path', default='./', type=str, metavar='PATH',
                    help='path to model')
parser.add_argument('--dataset-path', default='./', type=str, metavar='PATH',
                    help='path to dataset')
parser.add_argument('--bpe-path', default='./bpe.model', type=str, metavar='PATH',
                    help='path to bpe model')

parser.add_argument('--file', default=None, type=str,
                    help='path to log')

parser.add_argument('--vocabulary-path', default='./tb/vocabulary', type=str, metavar='PATH',
                    help='path to vocabulary')

parser.add_argument('--arch', default='stacked', type=str, choices=['mhsa', 'lstm', 'stacked'])

parser.add_argument('--random-seed', default=0, type=int,
                    help='random seed')


args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LanguageModelingBpeReaderMistakes(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 bpe: bool = False,
                 bpe_model_path: str = '',
                 mistakes_rate: float = 0.01) -> None:
        super().__init__(False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.bpe = bpe
        self.mistakes_rate = mistakes_rate
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
        with open(file_path, encoding='utf-8') as file:
            for sentence in file:
                sentence = mistakes_maker(sentence, self.mistakes_rate, args.random_seed)
                instance = self.text_to_instance(sentence)
                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                    yield instance


reader = LanguageModelingBpeReaderMistakes(tokenizer=WordTokenizer(),
                                           bpe=args.bpe,
                                           bpe_model_path=args.bpe_path,
                                           mistakes_rate=args.mistakes_rate)


val_dataset = reader.read(cached_path(args.dataset_path))

EMBEDDING_DIM = 64
HIDDEN_DIM = 64

vocab = Vocabulary.from_files(args.vocabulary_path)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, dropout=0))

lstm_model = LanguageModel(contextualizer=lstm, text_field_embedder=word_embeddings,
                           vocab=vocab)

transformer = MultiHeadSelfAttention(attention_dim=16, input_dim=EMBEDDING_DIM, num_heads=2,
                                     values_dim=16, attention_dropout_prob=0)

transformer_model = LanguageModel(contextualizer=transformer, text_field_embedder=word_embeddings, vocab=vocab)

stacked_transformer = StackedSelfAttentionEncoder(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=2,
                                                  projection_dim=16, feedforward_hidden_dim=16, num_attention_heads=2,
                                                  attention_dropout_prob=0)

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


with open(args.model_path, 'rb') as inp:
    model.load_state_dict(torch.load(inp, map_location='cpu'))

optimizer = torch.optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=1, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=None,
                  validation_dataset=val_dataset,
                  patience=10,
                  num_epochs=0)
np.random.seed(0)
with torch.no_grad():
    val_loss, num_batches = trainer._validation_loss()  # ну да, не очень, но коротко
    val_metrics = training_util.get_metrics(trainer.model, val_loss, num_batches, reset=True)
print(val_metrics)
if args.file is not None:
    with open(args.file, 'a') as out:
        out.write(str(args.bpe)+';'+str(args.mistakes_rate)+';'+str(val_metrics['perplexity'])+'\n')
