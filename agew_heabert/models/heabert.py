"""HeaBERT masked-language-model pretraining stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agew_heabert.stage_utils import check_result, require_paths


def run(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    train_path = Path(config["train_path"])
    validation_path = Path(config["validation_path"])
    output_dir = Path(config["output_dir"])
    if check_only:
        return check_result(
            "pretrain-heabert",
            [train_path, validation_path, output_dir],
            base_model=config["base_model"],
        )
    require_paths({"train_path": train_path, "validation_path": validation_path})
    try:
        import torch
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForWholeWordMask,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("HeaBERT pretraining requires the 'ml' dependency group") from exc

    set_seed(int(config.get("seed", 42)))
    revision = config["base_model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=revision)
    model = AutoModelForMaskedLM.from_pretrained(config["base_model"], revision=revision)

    class LineDataset(torch.utils.data.Dataset):
        def __init__(self, path: Path) -> None:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.examples = tokenizer(
                lines,
                truncation=True,
                max_length=int(config.get("max_sequence_length", 512)),
            )["input_ids"]

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return {"input_ids": self.examples[index]}

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("train_batch_size", 64)),
        per_device_eval_batch_size=int(config.get("evaluation_batch_size", 64)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        num_train_epochs=float(config.get("epochs", 30)),
        seed=int(config.get("seed", 42)),
        save_strategy="steps",
        eval_strategy="steps",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=LineDataset(train_path),
        eval_dataset=LineDataset(validation_path),
        data_collator=DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return {"stage": "pretrain-heabert", "status": "completed", "output_dir": str(output_dir)}
