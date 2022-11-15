from builtins import print
from typing import Optional

from overrides import overrides
import torch

from allennlp.nn.util import dist_reduce_sum
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric



def restore_token_list(spans):
    token_labels = []
    for label, (start, end) in spans:
        token_labels.extend([label] * int(end - start + 1))
    return token_labels
    
class F1Measure(Metric):


    def __init__(self,vocab,filter):
        self.filter=filter
        self.vocab=vocab
        self.num_class=len(vocab)-1

        self.total_token_pred=0
        self.total_token_correct = 0

        self.total_pred=[0 for _ in range(self.num_class)]
        self.total_gold=[0 for _ in range(self.num_class)]
        self.total_correct=[0 for _ in range(self.num_class)]


    def __call__(self,
                 span_logits,
                 meta_data,
                 pred_tags,
                 vocab):

        reversed_vocab = {v: k for k, v in vocab.items()}
        nolabel_index = reversed_vocab.get('NOLABEL', 0)
        gold_tags=[m['typed_spans'] for m in meta_data]

        span_logits[..., int(nolabel_index)] = -100
        
        for ps, gs, meta in zip(pred_tags, gold_tags, meta_data):
            ps = sorted(ps, key=lambda x: x[1][1])
            gs = sorted(gs, key=lambda x: x[1][1])

            # 计算sentence级别的micro-f1
            pred_labels = restore_token_list(ps)
            gold_labels = restore_token_list(gs)

            assert len(pred_labels) == len(gold_labels)
            self.total_token_pred += len(pred_labels)
            for p, g in zip(pred_labels, gold_labels):
                if p == g:
                    self.total_token_correct += 1
                
            # 计算span级别的micro-f1
            for p in ps:
                try:
                    self.total_pred[int(reversed_vocab[p[0]])]+=1
                except:
                    print('except')
                if p in gs:
                    self.total_correct[int(reversed_vocab[p[0]])] += 1

            for g in gs:
                self.total_gold[int(reversed_vocab[g[0]])] += 1

    def get_metric(self, reset: bool = False):

        metrics={'sent-f1':self.total_token_correct/self.total_token_pred}

        for i in range(self.num_class):
            t=self.vocab[i]
            precision = 0 if self.total_correct[i]==0 else self.total_correct[i] / self.total_pred[i]
            recall = 0 if self.total_correct[i]==0 else self.total_correct[i] / self.total_gold[i]
            metrics['span-precision-'+t]=precision
            metrics['span-recall-'+t]=recall
            metrics['span-f1-'+t]= 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        precision = sum(self.total_correct) / sum(self.total_pred)
        recall = sum(self.total_correct) / sum(self.total_gold)
        metrics['span-precision']=precision
        metrics['span-recall']=recall
        metrics['span-f1']=0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


        keys_to_remove=[]
        for k in metrics:
            if not any([f in k for f in self.filter]):
                keys_to_remove.append(k)
        for k in keys_to_remove:
            metrics.pop(k)
                
        if reset:
            self.reset()
        
        return metrics

    def reset(self):

        self.total_token_pred=0
        self.total_token_correct = 0

        self.total_pred=[0 for _ in range(self.num_class)]
        self.total_gold=[0 for _ in range(self.num_class)]
        self.total_correct=[0 for _ in range(self.num_class)]
