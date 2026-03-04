# consolidation/trainer.py
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from consolidation.ewc import EWC
from config import cfg


class MemoryDataset(Dataset):
    """Converts high-strength memories into training examples."""

    def __init__(self, memories: list[dict], tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        for mem in memories:
            # Format as a simple instruction → completion pair
            # The model learns "given context, recall this"
            prompt = f"<|system|>Remember this information:</s><|user|>What do you know?</s><|assistant|>{mem['content']}</s>"
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append({
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "strength": mem["strength"],  # used as sample weight
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def consolidate(model, tokenizer, memory_store, ewc_state=None):
    """
    The "sleep phase": micro-train LoRA adapter on high-strength memories.
    
    CRITICAL SAFEGUARDS:
    1. Verify base is frozen before training
    2. EWC penalty guards against overwriting prior adapter knowledge
    3. Gradient clipping prevents explosive updates
    4. Save adapter checkpoint before + after for rollback
    """
    candidates = memory_store.get_consolidation_candidates()

    if len(candidates) < cfg.consolidation_trigger_count:
        print(f"[Consolidation] Only {len(candidates)} candidates, need {cfg.consolidation_trigger_count}. Skipping.")
        return ewc_state

    print(f"[Consolidation] Starting sleep phase with {len(candidates)} memories...")

    # SAFETY CHECK: verify base still frozen
    for name, param in model.named_parameters():
        if "lora_" not in name and "modules_to_save" not in name:
            assert not param.requires_grad, f"BASE WEIGHT IS TRAINABLE: {name}"

    # Save adapter checkpoint for rollback
    model.save_pretrained(cfg.adapter_path + "_backup")

    dataset = MemoryDataset(candidates, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Compute EWC Fisher before this training run
    # (protects what the adapter already learned in previous consolidations)
    if ewc_state is None and len(candidates) > 0:
        ewc_state = EWC(model, dataloader)

    # Optimizer — only update LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.consolidation_lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=2, num_training_steps=len(dataloader) * cfg.consolidation_epochs
    )

    model.train()
    for epoch in range(cfg.consolidation_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            task_loss = outputs.loss

            # EWC penalty: resist changing what we already learned
            ewc_loss = ewc_state.penalty(model) if ewc_state else 0.0
            total_batch_loss = task_loss + cfg.ewc_lambda * ewc_loss

            total_batch_loss.backward()

            # Gradient clipping — CRITICAL to prevent destructive updates
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += task_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Consolidation] Epoch {epoch+1}/{cfg.consolidation_epochs} | Loss: {avg_loss:.4f}")

    model.eval()

    # Save updated adapter
    model.save_pretrained(cfg.adapter_path)
    print(f"[Consolidation] Adapter saved to {cfg.adapter_path}")

    # Compute new EWC state (includes this run's knowledge)
    new_ewc_state = EWC(model, dataloader)
    return new_ewc_state
