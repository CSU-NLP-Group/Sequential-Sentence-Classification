from typing import Optional

import torch
import torch.nn as nn
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.util import get_range_vector, get_device_of





class SpanConcatEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 combination: str = "x,y",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 dropout: float = 0.3):

        super().__init__()
        self._input_size = input_size
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        self._span_width_embedding: Optional[Embedding] = None

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )
        self.W = nn.Linear(self.get_output_dim(), output_size)
        self._dropout = nn.Dropout(p = dropout)


    def get_output_dim(self) -> int:
        combined_dim = util.get_combined_dim(self._combination, [self._input_size, self._input_size])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    def forward(self, sequence_tensor_start: torch.FloatTensor,
                sequence_tensor_end: torch.FloatTensor):

        span_embeddings = self._embed_spans(sequence_tensor_start, sequence_tensor_end)
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since `SpanField` use inclusive indices.
            # But here we do not add 1 beacuse we often initiate the span width
            # embedding matrix with `num_width_embeddings = max_span_width`
            # shape (batch_size, num_spans)

            batch_size, sequence_length, _, __ = span_embeddings.size()
            range_vector = get_range_vector(size=sequence_length, device=get_device_of(span_embeddings))
            # (sequence_length, sequence_length)
            span_width_matrix = torch.abs(range_vector.unsqueeze(0) - range_vector.unsqueeze(-1))

            if self._bucket_widths:
                span_width_matrix = util.bucket_values(
                    span_width_matrix, num_total_buckets=self._num_width_embeddings  # type: ignore
                )

            # Embed the span widths and concatenate to the rest of the representations.
            # (sequence_length, sequence_length, embedding_dim)
            span_width_embeddings = self._span_width_embedding(span_width_matrix)
            # (batch_size, sequence_length, sequence_length, embedding_dim)
            span_width_embeddings = span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return self.W(span_embeddings)

    def _embed_spans(
        self,
        sequence_tensor_start: torch.FloatTensor,
        sequence_tensor_end: torch.FloatTensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = sequence_tensor_start.size()

        # (batch_size, sequence_length, embedding_dim)
        start_embeddings = self._dropout(
            sequence_tensor_start.unsqueeze(-2).expand(-1, -1, sequence_length, -1))
        end_embeddings = self._dropout(
            sequence_tensor_end.unsqueeze(1).expand(-1, sequence_length, -1, -1))

        combined_tensors = util.combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )

        return combined_tensors



class BiaffineEncoder(nn.Module):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    `x = span_start_embeddings` and `y = span_end_embeddings`.

    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Registered as a `SpanExtractor` with name "endpoint".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    """

    def __init__(
        self,
        input_size: int,
        biaf_input_size: int,
        output_size: int,
        combination: str = "x,y",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
        use_span_encoder: bool = False,
        dropout: float = 0.3
    ) -> None:
        super().__init__()

        self._use_span_encoder = use_span_encoder
        self._dropout = nn.Dropout(p=dropout)
        self.U = nn.Parameter(
            torch.randn(biaf_input_size + 1, output_size, biaf_input_size + 1)
        ) # +1 是因为有bias

        if use_span_encoder:
            self.span_encoder = SpanConcatEncoder(
                input_size=biaf_input_size, output_size=output_size,dropout=dropout,
                combination=combination,num_width_embeddings=num_width_embeddings,
                span_width_embedding_dim=span_width_embedding_dim,bucket_widths=bucket_widths
                )

        self.start_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=biaf_input_size), nn.ReLU())
        self.end_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=biaf_input_size), nn.ReLU())

        nn.init.xavier_uniform_(self.U)


    def forward(
        self, sequence_tensor: torch.FloatTensor,
    ):
        """
         Given a sequence tensor, extract spans, concatenate width embeddings
         when need and return representations of them.

         # Parameters

         sequence_tensor_start : `torch.FloatTensor`, required.
             A tensor of shape (batch_size, sequence_length, embedding_size)
             representing an embedded sequence of words.
         sequence_tensor_end : `torch.LongTensor`, required.
             A tensor of shape `(batch_size, num_spans, 2)`, where the last
             dimension represents the inclusive start and end indices of the
             span to be extracted from the `sequence_tensor`.
         span_matrix_mask : `torch.BoolTensor`, optional (default = `None`).
             A tensor of shape (batch_size, sequence_length) representing padded
             elements of the sequence.
         # Returns

         A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
         where `embedded_span_size` depends on the way spans are represented.
         """

        sequence_tensor_start = self.start_layer(sequence_tensor)
        sequence_tensor_end = self.end_layer(sequence_tensor)


        sequence_tensor_start = torch.cat((sequence_tensor_start,
                                           torch.ones_like(sequence_tensor_start[..., :1])), dim=-1)
        sequence_tensor_end = torch.cat((sequence_tensor_end,
                                         torch.ones_like(sequence_tensor_end[..., :1])), dim=-1)

        # biaffine x*U*y
        biaf_output_fwd = torch.einsum('bxi,ioj,byj->bxyo', self._dropout(sequence_tensor_start),
                                       self.U, self._dropout(sequence_tensor_end))
        biaf_output_bwd = torch.einsum('bxi,ioj,byj->bxyo', self._dropout(sequence_tensor_end),
                                       self.U, self._dropout(sequence_tensor_start))
        
        biaf_fwd_triu=self.tri(biaf_output_fwd,'u')
        biaf_bwd_tril=self.tri(biaf_output_bwd,'l',-1)

        biaf_output=biaf_fwd_triu+biaf_bwd_tril
        

        if self._use_span_encoder:
            biaf_output += self.span_encoder(sequence_tensor_start, sequence_tensor_end)

        return biaf_output 


    def tri(self,matrices,direction='u',diagonal=0):
        # matrices: batch_size * seq_len1 * seq_len2 * num_class

        # matrices: batch_size * num_class * seq_len1 * seq_len2
        matrices=torch.transpose(torch.transpose(matrices,1,3),2,3)
        if direction=='u':
            matrices=matrices.triu(diagonal)
        elif direction=='l':
            matrices=matrices.tril(diagonal)
        # matrices: batch_size * seq_len1 * seq_len2 * num_class
        matrices=torch.transpose(torch.transpose(matrices,1,3),1,2)
        return matrices

            

