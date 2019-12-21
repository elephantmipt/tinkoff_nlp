import torch
import math
from language_model import LanguageModel
from itertools import chain
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
import argparse

PATH = '../logs/stacked/'

parser = argparse.ArgumentParser(description='Language model argument parser')
parser.add_argument('--bpe', action='store_true')
parser.add_argument('--mistakes-rate', default=0., type=float,
                    help='mistakes rate for data')
parser.add_argument('--path', default='', type=str, metavar='PATH',
                    help='path to dataset and bpe model')

args = parser.parse_args()

from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

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

EMBEDDING_DIM = 32
HIDDEN_DIM = 32

vocab = Vocabulary.from_instances(chain(train_dataset, test_dataset))

with open(PATH + 'test_data.csv', 'r') as inp:
    with open(PATH + 'test_data_m.csv', 'w') as  out:
        for line in inp:
            out.write(mistakes_maker(line, args.mistakes_rate) + '\n')

test_dataset_m = reader.read(cached_path(PATH + 'test_data_m.csv'))

#vocab = Vocabulary.from_instances(chain(train_dataset, test_dataset, test_dataset_m))

stacked_transformer = StackedSelfAttentionEncoder(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=2,
                                                  projection_dim=16, feedforward_hidden_dim=16, num_attention_heads=2,
                                                  attention_dropout_prob=0.2)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

model = LanguageModel(contextualizer=stacked_transformer,
                      text_field_embedder=word_embeddings,
                      vocab=vocab)





with open(PATH + 'best.th', 'rb') as inp:
    model.load_state_dict(torch.load(inp, map_location='cpu'))

optimizer = torch.optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=1, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=test_dataset_m,
                  patience=10,
                  num_epochs=0)
with torch.no_grad():
    val_loss, num_batches = trainer._validation_loss()
    val_metrics = training_util.get_metrics(trainer.model, val_loss, num_batches, reset=True)
print(val_metrics)
