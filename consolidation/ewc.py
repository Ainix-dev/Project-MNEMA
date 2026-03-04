# consolidation/ewc.py
"""
EWC (Elastic Weight Consolidation) prevents catastrophic forgetting during LoRA updates.
It computes the Fisher Information Matrix (importance of each weight) from prior behavior,
then adds a penalty to the loss to resist changing important weights.

For LoRA, we only apply EWC to the adapter weights (A and B matrices).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import cfg


class EWC:
    def __init__(self, model, dataloader: DataLoader):
        self.model = model
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n
        }
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader: DataLoader) -> dict:
        """Estimate Fisher Information Matrix via diagonal approximation."""
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        self.model.train()
        for batch in dataloader:
            self.model.zero_grad()
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # causal LM
            )

            log_probs = -outputs.loss
            log_probs.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and "lora_" in n and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        # Normalize
        n_batches = len(dataloader)
        for n in fisher:
            fisher[n] /= n_batches

        self.model.eval()
        return fisher

    def penalty(self, model) -> torch.Tensor:
        """
        Compute EWC penalty term.
        Add this to your training loss: total_loss = task_loss + ewc_lambda * ewc.penalty(model)
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_" in n and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss
