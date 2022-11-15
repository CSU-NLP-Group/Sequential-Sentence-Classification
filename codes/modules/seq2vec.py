from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from torch.nn.parameter import Parameter
import torch
from allennlp.modules.attention.attention import Attention



@Seq2VecEncoder.register('my_att')
class MyAttention(Seq2VecEncoder):

    def __init__(self,
                input_size,
                num_query,
                attention:Attention
                ):
        super().__init__()
        self.input_size=input_size
        self.num_query=num_query
        self.attention=attention
        self.query=Parameter(torch.randn(num_query,input_size))
        self.projection_k=torch.nn.Linear(input_size,input_size)
        self.projection_v=torch.nn.Linear(input_size,input_size)

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return self.input_size * self.num_query

    def weighting(self,w:torch.Tensor,v:torch.Tensor):
        # w:batch_size * seq_len
        # v:batch_size * seq_len * embedding_size
        return torch.einsum('bs,bsi->bi',w,v)

    def forward(self, sequences: torch.Tensor, mask: torch.BoolTensor = None):
        # sequences: batch_size * sequence_len * embedding_size
        # mask: batch_size * sequence_len
        k=self.projection_k(sequences)
        v=self.projection_v(sequences)
        result=[self.weighting(self.attention(q.expand(k.shape[0],k.shape[-1]),k,mask),v) for q in self.query]

        # sequences: batch_size *  (num_query*embedding_size)
        result=torch.cat(result,dim=-1)

        return result
