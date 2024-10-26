from transformers import Trainer
import torch
import os
import tensorflow as tf
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

# Parts have been taken and adapted from https://github.com/havenhq/mamba-chat
class MambaTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs= False):
        input_ids = inputs.pop("input_ids")
        output = model(input_ids)
        lm_logits = output.logits

        labels = inputs.pop("labels").to(lm_logits.device)
       
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index = 0)
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return (lm_loss, {'logits': output.logits}) if return_outputs else lm_loss 
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.model.save_pretrained(output_dir)

    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """"
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(self.args.warmup_ratio*num_training_steps), num_training_steps)
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    
    
# combined from https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/schedules/learning_rate_schedule.py#L572 and https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    """
    Used to realize a scheduler that does cosine decay not to a learning rate of 0 but to alpha*initial_lr.
    """
    alpha = 0.01 
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, (1-alpha) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) + alpha)


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
  