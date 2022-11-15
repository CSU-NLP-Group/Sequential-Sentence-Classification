from time import sleep
from typing import List, Dict, Any, Optional
from allennlp.training import GradientDescentTrainer
from allennlp.training.callbacks.callback import TrainerCallback
import torch

@TrainerCallback.register('FGM')
class FGM(TrainerCallback):

    def __init__(self,emb_name:str,epsilon:float=1.0,loss_weight:float=1.0):
        self.emb_name=emb_name
        self.epsilon=epsilon
        self.loss_weight=loss_weight

    def on_backward(self, trainer: "GradientDescentTrainer", batch_outputs: Dict[str, torch.Tensor], backward_called: bool, **kwargs) -> bool:
        
        # 反向传播获得梯度
        if not backward_called:
            batch_outputs['loss'].backward()
        # 根据梯度在embedding上添加扰动，同时备份embedding
        embeddings={}
        for n,p in trainer.model.named_parameters():
            if p.requires_grad and self.emb_name in n:
                embeddings[n]=p.data.clone()
                norm=torch.norm(p.grad)
                if norm !=0 and not torch.isnan(norm):
                    theta=self.epsilon * p.grad / norm
                    p.data.add_(theta)
        # 再次进行前向传播计算adversarial loss，并进行反向传播，恢复embedding
        adv_loss=self.loss_weight*trainer.model(batch_outputs['tokens'],batch_outputs['gold_span_label'],batch_outputs['metadata'])['loss']
        adv_loss.backward()
        for n,p in trainer.model.named_parameters():
            if n in embeddings:
                p.data=embeddings[n]
        
        return True
