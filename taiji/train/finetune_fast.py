"""
态极快速 Fine-tuning 脚本
========================

使用少量高质量对话数据快速 fine-tuning。
"""
import os
import sys
import json
import time
import logging
import torch
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Finetune")

CONFIG = {
    "model_path": "taiji/evolution_data/upgraded_models/pretrained/best",
    "data_file": "taiji/training_data/taiji_conversation_data.jsonl",
    "output_dir": "taiji/evolution_data/upgraded_models/finetuned",
    "max_seq_len": 256,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "max_steps": 1000,
    "save_every": 100,
    "log_every": 5,
}


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"Loading data: {data_file}")
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "messages" in item:
                        text = self._format(item["messages"])
                        if text and len(text) > 20:
                            self.samples.append(text)
                except:
                    continue
        logger.info(f"Loaded {len(self.samples)} samples")

    def _format(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs = []
    padded_labels = []
    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))
    return torch.stack(padded_inputs), torch.stack(padded_labels)


def main():
    from taiji.loader import load_model, save_model

    cfg = CONFIG

    logger.info(f"Loading model: {cfg['model_path']}")
    model, tokenizer = load_model(cfg['model_path'])
    num_params = model.get_num_parameters()
    if isinstance(num_params, dict):
        num_params = num_params.get('total', 0)
    logger.info(f"Parameters: {num_params:,}")

    dataset = SimpleDataset(cfg["data_file"], tokenizer, cfg["max_seq_len"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    logger.info(f"Starting training: max_steps={cfg['max_steps']}, batch={cfg['batch_size']}")
    model.train()

    global_step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while global_step < cfg["max_steps"]:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, labels = next(data_iter)

        output = model(input_ids)
        logits = output.logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss = loss / cfg["gradient_accumulation_steps"]
        loss.backward()

        if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % cfg["log_every"] == 0:
            elapsed = time.time() - start_time
            logger.info(f"Step {global_step}/{cfg['max_steps']} | Loss: {loss.item() * cfg['gradient_accumulation_steps']:.4f} | Elapsed: {elapsed:.0f}s")

        if global_step % cfg["save_every"] == 0:
            save_path = os.path.join(cfg["output_dir"], f"step_{global_step}")
            save_model(model, tokenizer, save_path)
            logger.info(f"Saved: {save_path}")

    final_path = os.path.join(cfg["output_dir"], "final")
    save_model(model, tokenizer, final_path)
    best_path = os.path.join(cfg["output_dir"], "best")
    save_model(model, tokenizer, best_path)

    elapsed = time.time() - start_time
    logger.info(f"Done! Steps: {global_step}, Time: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
