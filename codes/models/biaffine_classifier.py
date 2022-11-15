# torch.backends.cudnn.deterministic = True
from typing import Optional, Dict, List, Any
from itertools import groupby
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder, \
    TimeDistributed
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from overrides import overrides
from torch import nn

from codes.utils.my_span_extractor import BiaffineEncoder
from codes.utils.my_spanf1_measure import F1Measure


@Model.register('biaffine_classifier')
class BiaffineClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sent_encoder: Seq2VecEncoder,
                 biaf_input_size: int = 128,
                 ctx_sent_encoders: List[Seq2SeqEncoder] = None,
                 label_namespace: str = "tags",
                 use_span_encoder:bool = False,
                 # gamma: float = 0.,
                 o_tag_weight: float = 1.,
                 max_span_len: int = 18,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):

        super().__init__(vocab, **kwargs)

        self.vocab = vocab
        self.o_index = self.vocab.add_token_to_namespace('NOLABEL', namespace=label_namespace)
        num_classes = vocab.get_vocab_size(namespace=label_namespace)

        self.label_namespace = label_namespace
        # 将词进行tokenize后映射为ID
        self.text_field_embedder = text_field_embedder
        # 将句子表示为向量
        self.sent_encoder = TimeDistributed(sent_encoder)
        # 上下文敏感的向量表示
        if ctx_sent_encoders is not None:
            self.ctx_sent_encoders = nn.ModuleList(ctx_sent_encoders)

        self._use_span_encoder = use_span_encoder
        self._o_tag_weight = o_tag_weight
        self._max_span_len = max_span_len
        # self._gamma = gamma

        # 获得句子间的边界信息
        text_encoder_dim = sent_encoder.get_output_dim() if ctx_sent_encoders is None \
            else ctx_sent_encoders[-1].get_output_dim()

        self.biaffine_scorer = BiaffineEncoder(
            input_size=text_encoder_dim,
            biaf_input_size=biaf_input_size,
            output_size=num_classes,
            combination='x,y',
            num_width_embeddings=40,
            span_width_embedding_dim=50,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout) if dropout else None
        # 评价指标
        self.id2tag=self.vocab.get_index_to_token_vocabulary('tags')
        self.tag2id=self.vocab.get_token_to_index_vocabulary('tags')
        self.f1_measure = F1Measure(self.id2tag,filter=['f1'])

        initializer(self)

    # evalute on predicted results on the metrics
    def update_metrics(self, logits, tags, sent_mask):
        # class_probabilities = logits * 0.

        # for i, instance_tags in enumerate(pred_tags):
        #     for j, tag_id in enumerate(instance_tags):
        #         class_probabilities[i, j, tag_id] = 1

        for metric in self.metrics.values():
            metric(logits, tags, sent_mask)

        if self.calculate_span_f1:
            self._f1_metric(logits, tags, sent_mask)

    def get_sent_embedding(self, tokens: TextFieldTensors, mask: torch.BoolTensor):

        # word_embeded: shape(batch_size, num_sent, num_words, embed_dim)
        word_embedded = self.text_field_embedder(tokens, num_wrapping_dims=1)
        sent_embedded = self.sent_encoder(word_embedded, mask)

        # shape: batch_size, num_sent
        sent_mask = mask[:, :, 0]
        for _encoder in self.ctx_sent_encoders:
            sent_embedded = _encoder(
                inputs=sent_embedded,
                mask=sent_mask
            )
        if self.dropout:
            sent_embedded = self.dropout(sent_embedded)
        return sent_embedded

    def boundary_smooth(self, span_matrix:torch.LongTensor):


        pass
    
    
    def forward(self,
                tokens: TextFieldTensors,
                span_matrix: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:


        # mask中0表示padding的部分
        mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sent_embeded = self.get_sent_embedding(tokens, mask)

        # 得到biaffine的结果
        span_logits = self.biaffine_scorer(sent_embeded)
        # the mask for the sentences
        # shape: batch_size, num_sent
        sent_mask = mask[:, :, 0]
        block_mask = sent_mask.unsqueeze(-1) * sent_mask.unsqueeze(-2)

        # 上三角的block_mask
        # block_mask_triu = block_mask.triu()
        # token_loss_weight = block_mask_triu
        token_loss_weight = block_mask
        if self._o_tag_weight < 1.:
            token_loss_weight = torch.ones_like(span_matrix).masked_fill(span_matrix == -1, value = self._o_tag_weight)
            token_loss_weight *= block_mask

        
        # token_loss_weight=token_loss_weight.triu()

        span_matrix.masked_fill_(span_matrix == -1, value=self.o_index)
        loss = sequence_cross_entropy_with_logits(logits=span_logits.contiguous(), targets=span_matrix,
                                                  weights=token_loss_weight, average='token')
        # loss = sequence_cross_entropy_with_logits(logits=span_logits.contiguous(), targets=span_matrix,
        #                                           weights=token_loss_weight, average='token',
        #                                           alpha=[0.762768096,0.752995009,0.736232761,0.915876933,0.963543226,1.868583975,0.7],
        #                                           gamma=2
        #                                           )

        #解码时同时考虑span_logist上三角和下三角
        # mix_span_logits=span_logits+torch.transpose(span_logits,1,2)

        # block_mask_triu = block_mask.triu()
        # pred_label_index = torch.argmax(mix_span_logits, dim=-1)
        # pred_label_mask = block_mask_triu & (pred_label_index != self.o_index)
        # gold_label_mask = block_mask_triu & (span_matrix != self.o_index)

        # self.spanf1_measure(mix_span_logits.contiguous(), span_matrix, pred_label_mask, gold_label_mask)
        
        #计算F1时只考虑上三角的, F值的计算原理是，
        # 精确率 = 预测不为O标签的正确的个数 / predict标 签个数
        # 召回率 = 预测不为O标签的正确的个数 / gold标签个数
        block_mask_triu = block_mask.triu()
        pred_label_index = torch.argmax(span_logits, dim=-1)
        pred_label_mask = block_mask_triu & (pred_label_index != self.o_index)
        gold_label_mask = block_mask_triu & (span_matrix != self.o_index)

        seq_lengths = block_mask[:, 0, :].sum(dim=-1)

        pred_tags=self.decoding(span_logits.contiguous(),seq_lengths)

        self.f1_measure(span_logits.contiguous(),metadata,pred_tags,self.id2tag)


        output_dict = {
            "loss": loss,
            "tokens":tokens,
            "sent_embeded":sent_embeded,
            "span_logits": span_logits,
            "block_mask": block_mask,
            "gold_span_label": span_matrix,
            "metadata":metadata,
            "vocab": self.vocab.get_index_to_token_vocabulary(namespace=self.label_namespace)
        }

        return output_dict

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        span_logits = output_dict['span_logits']
        block_mask = output_dict['block_mask']

        # 直接对biaffine的输出，输出最大概率的标签作为预测的结果
        pred_label_index = torch.argmax(span_logits, dim = -1)
        pred_label_mask = block_mask.triu() & (pred_label_index != self.o_index)

        # 对span和span对应的标签进行
        span_probs = span_logits.softmax(dim = -1).max(dim=-1)[0]
        spans = torch.nonzero(pred_label_mask)
        probs = torch.masked_select(span_probs, mask=pred_label_mask)
        values = torch.masked_select(pred_label_index, mask=pred_label_mask)
        span_value = torch.cat((spans, values.unsqueeze(-1), probs.unsqueeze(-1)), dim=-1)

        pred_spans = []
        for k, g in groupby(span_value.tolist(), key=lambda x:x[0]):
            span_label = []
            #batch_id, start, end and label
            for _, s, e, l, p in list(g):
                span_label.append(
                    [(self.vocab.get_token_from_index(l, namespace=self.label_namespace), (s, e)), p])
            pred_spans.append(span_label)

        # pred_spans = self._viterbi_decode(span_logits, block_mask[..., 0].sum(dim=-1))
        gold_spans = [meta['typed_spans'] for meta in output_dict['metadata']]

        output_dict['pred_tags'] = pred_spans
        output_dict['gold_tags'] = gold_spans
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return self.f1_measure.get_metric(reset)

    def get_labels(self,paths: torch.Tensor, tags: torch.Tensor, lens: torch.Tensor):
        batch_size = paths.size(0)
        paths_list = paths.tolist()
        tags_list = tags.tolist()
        lens_list = lens.tolist()
        outputs = []
        for i in range(batch_size):
            paths = paths_list[i]
            tags = tags_list[i]
            length = lens_list[i]
            segments = []
            while length > 0:
                segments.append((paths[length - 1], tags[length - 1]))
                length = length - paths[length - 1]
            segments.reverse()
            start_id = 0
            output = []
            for length, tag_index in segments:
                output.append((self.id2tag.get(tag_index), (start_id, start_id + length - 1)))
                start_id += length
            outputs.append(output)
        return outputs

    # 对logits进行解码
    def decoding(self,transitions: torch.Tensor, real_seqlen: torch.Tensor,max_span_len: int = 18):
        # transitions: (batch_size, seq_len, max_seg_len, n_tags)
        batch_size, seq_len, _, n_tags = transitions.size()

        nolabel_index = self.tag2id.get('NOLABEL', 0)
        transitions[..., int(nolabel_index)] = -100

        alpha = transitions.new_empty(batch_size, seq_len + 1).fill_(-1e9)
        path = transitions.new_empty(batch_size, seq_len + 1, dtype=torch.int).fill_(-1)
        pred_tags = transitions.new_empty(batch_size, seq_len + 1, dtype=torch.int).fill_(-1)

        # set all tags in the first place to zero
        alpha[:, 0] = 0

        for ending_pos in range(1, seq_len + 1):
            length = min(max_span_len, ending_pos)
            # alpha_rev: (batch_size, length)
            alpha_rev = alpha[:, ending_pos - length: ending_pos]
            # transition: (batch_size, length, n_tags)
            transition = transitions[:, ending_pos - length: ending_pos, ending_pos - 1, :]

            # f: (batch_size, length, n_tags)
            f = alpha_rev.unsqueeze(2) + transition
            f = f.view(batch_size, -1)
            alpha[:, ending_pos], tmp = f.max(-1)

            pred_tags[:, ending_pos] = tmp.fmod(n_tags)
            path[:, ending_pos] = length - torch.div(tmp, n_tags, rounding_mode='floor')

        return self.get_labels(path[:, 1:], pred_tags[:, 1:], real_seqlen)

