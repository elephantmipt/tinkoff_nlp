from typing import Iterator, List, Dict
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
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from metrics import Perplexity

torch.manual_seed(1)


class PosDatasetReader(DatasetReader):
    """

    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        reg_numbers = r'[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*'
        with open(file_path) as f:
            for line in f:
                message = line.strip()
                message = re.sub(pattern=reg_numbers, repl=' ', string=message)
                message = message.split()

                yield self.text_to_instance([Token(word) for word in message])


class LstmTagger(LanguageModel):
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            dropout: float = None,
            num_samples: int = None,
            sparse_embeddings: bool = False,
            bidirectional: bool = False,
            initializer: InitializerApplicator = None,
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if contextualizer.is_bidirectional() is not bidirectional:
            raise ConfigurationError(
                "Bidirectionality of contextualizer must match bidirectionality of "
                "language model. "
                f"Contextualizer bidirectional: {contextualizer.is_bidirectional()}, "
                f"language model bidirectional: {bidirectional}"
            )

        self._contextualizer = contextualizer
        self._bidirectional = bidirectional

        # The dimension for making predictions just in the forward
        # (or backward) direction.
        if self._bidirectional:
            self._forward_dim = contextualizer.get_output_dim() // 2
        else:
            self._forward_dim = contextualizer.get_output_dim()

        # TODO(joelgrus): more sampled softmax configuration options, as needed.
        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(
                num_words=vocab.get_vocab_size(),
                embedding_dim=self._forward_dim,
                num_samples=num_samples,
                sparse=sparse_embeddings,
            )
        else:
            self._softmax_loss = _SoftmaxLoss(
                num_words=vocab.get_vocab_size(), embedding_dim=self._forward_dim
            )

        # This buffer is now unused and exists only for backwards compatibility reasons.
        self.register_buffer("_last_average_loss", torch.zeros(1))

        self._perplexity = Perplexity()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        if initializer is not None:
            initializer(self)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"perplexity": self.loss.get_loss(reset)}

    
