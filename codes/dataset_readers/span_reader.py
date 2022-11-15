import itertools
from collections import Counter
from typing import Dict, Iterable, List

import torch
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import Field, TextField, SequenceLabelField, ListField, MetadataField, SpanField, TensorField, \
    AdjacencyField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from overrides import overrides
from .utils import convert_to_span, _is_divider


@DatasetReader.register('span_reader')
class SpanReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 is_pubmed: bool = False,
                 **kwargs):
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        # 是否是Pubmed数据集，pubmed数据格式和Nicta数据在格式上有不同
        self.is_pubmed = is_pubmed
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    @overrides
    def text_to_instance(self, sentences: List[str], labels: List[str] = None) -> Instance:

        text_fields: List[Field] = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            text_fields.append(TextField(tokens[:self.max_tokens]))
        text_field_list = ListField(text_fields)

        fields = {
            'tokens': text_field_list,
            'metadata': MetadataField({'sentences': sentences,
                                       'labels': labels})
        }

        if labels is not None:
            # 同时获得bio标签和span的标签
            bio_labels, typed_spans = convert_to_span(labels)

            # 得到span及其标签数据
            span_type, span_range = zip(*typed_spans)
            span_list_field = ListField(
                [SpanField(span_start=s, span_end=e,
                           sequence_field=text_field_list) for s, e in span_range])
            fields['spans'] = span_list_field
            fields['span_labels'] = SequenceLabelField(labels=list(span_type),
                                                       sequence_field=span_list_field, label_namespace='stags')

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as data_file:
            line_chunks = (
                lines
                for is_divider, lines in itertools.groupby(data_file, _is_divider)
                if not is_divider
            )

            for lines in self.shard_iterable(line_chunks):

                if self.is_pubmed:
                    fields = [line.strip().split('\t', 1)
                              for line in lines if not line.startswith("###")]

                else:
                    fields = [line.strip().split('|', 2)
                              for line in lines if not line.startswith("###")]

                fields = [list(field) for field in zip(*fields)]
                if self.is_pubmed:
                    labels = [f[:2] for f in fields[0]]
                    sentences = fields[1]
                    yield self.text_to_instance(sentences, labels)
                elif len(fields) > 2:
                    _, labels, sentences = fields
                    assert len(labels) == len(sentences)
                    yield self.text_to_instance(sentences, labels)
                else:
                    yield self.text_to_instance(fields[0])

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:

        for text_field in instance.fields["tokens"]:
            text_field._token_indexers = self.token_indexers

@DatasetReader.register('biaffine_reader')
class BiaffineReader(SpanReader):

    @overrides
    def text_to_instance(self, sentences: List[str], labels: List[str] = None) -> Instance:

        text_fields: List[Field] = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            text_fields.append(TextField(tokens[:self.max_tokens]))
        text_field_list = ListField(text_fields)
        meta_field = {'sentences': sentences, 'labels': labels}

        fields = {
            'tokens': text_field_list,
            'tags': SequenceLabelField(labels = labels,
                                       sequence_field=text_field_list, label_namespace='tags'),
        }

        if labels is not None:
            # 获得gold的span及其标签
            bio_labels, typed_spans = convert_to_span(labels)
            bi_typed_spans = list(typed_spans)

            # 将邻接矩阵设置成对称形式，即（2，3）和（3，2)的标签是相同的
            for stype, (start, end) in typed_spans:
                if (stype, (end, start)) not in typed_spans:
                    bi_typed_spans.append((stype, (end, start)))

            span_type, span_range = zip(*bi_typed_spans)
            span_matrix = AdjacencyField(indices=span_range, labels=span_type, padding_value=-1,
                                         sequence_field=text_field_list, label_namespace="tags")
            fields['span_matrix'] = span_matrix
            meta_field.update({'typed_spans': typed_spans})
            fields['metadata'] = MetadataField(meta_field)

        assert 'tags' in fields

        return Instance(fields)

