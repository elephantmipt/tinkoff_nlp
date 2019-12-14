import torch
from itertools import chain
from typing import Dict, Iterable, Union, Optional, List
import torch.optim as optim
import math
from tqdm import tqdm


from language_model import LanguageModel
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from preprocessing import Preprocessing, _preproc
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


torch.manual_seed(1)

PATH = '../cache/'

BPE = False

preprocessing = Preprocessing(inp_path=PATH+'data.csv', train_path=PATH+'train_data.csv',
                              test_path=PATH+'test_data.csv', bpe=BPE, model_path=PATH+'bpe_yttm.model')


class LanguageModelReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

    def text_to_instance(self,  # type: ignore
                         sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as file:
            pbar = tqdm(file)
            pbar.set_description('applying regular expressions')
            for sentence in pbar:
                sentence = _preproc(sentence)
                instance = self.text_to_instance(sentence)
                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                    yield instance

class LanguageModelBpeReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

    def text_to_instance(self,  # type: ignore
                         sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as file:
            pbar = tqdm(file)
            pbar.set_description('applying regular expressions')
            for sentence in pbar:
                sentence = _preproc(sentence)
                instance = self.text_to_instance(sentence)
                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                    yield instance


reader = LanguageModelReader(max_sequence_length=30)


train_dataset = reader.read(cached_path(PATH + 'train_data.csv'))
test_dataset = reader.read(cached_path(PATH + 'test_data.csv'))


EMBEDDING_DIM = 32
HIDDEN_DIM = 32


vocab = Vocabulary.from_instances(chain(train_dataset,test_dataset))

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

transformer = MultiHeadSelfAttention(num_heads=8, input_dim=EMBEDDING_DIM,
                                     attention_dim=16, values_dim=16)

transformer_model = LanguageModel(contextualizer=transformer,
                                  text_field_embedder=word_embeddings,
                           vocab=vocab)

lstm_model = LanguageModel(contextualizer=lstm, text_field_embedder=word_embeddings,
                           vocab=vocab)
print(vocab.get_vocab_size('tokens'))

  #elmo=Elmo(num_output_representations=512, )

optimizer = optim.Adam(transformer_model.parameters())
iterator = BucketIterator(batch_size=16, sorting_keys=[("source", "num_tokens")])
iterator.index_with(vocab)
trainer_transformer = Trainer(model=transformer_model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=test_dataset,
                  patience=3,
                  num_epochs=100,
                  serialization_dir='./tb/transformer',
                  )

trainer_transformer.train()