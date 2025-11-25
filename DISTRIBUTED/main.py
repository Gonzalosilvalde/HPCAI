import torch
import time
import os
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler

from torch.utils.data import DataLoader
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 384,
    "batch_size": 16,  # Por GPU
    "num_epochs": 5,
    "learning_rate": 3e-5,
    "num_warmup_steps": 500,
    "subset_size": 10000,
    "doc_stride": 128,
    "num_workers": 4,
}

STRATEGY = "ddp"
NUM_GPUS = 2
NUM_NODES = 2
USE_PROFILER = False


class SQuADDataModule(pl.LightningDataModule):
    """DataModule para manejar la carga y preprocesamiento de datos"""

    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        max_length: int = 384,
        doc_stride: int = 128,
        batch_size: int = 8,
        subset_size: int = 10000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.train_dataset = None

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def setup(self, stage=None):
        """Prepara los datos - optimizado para multi-GPU"""
        if stage == "fit" or stage is None:
            if self.trainer.is_global_zero:
                print(f"\n[GPU 0] Loading and caching SQuAD dataset...")
                load_dataset("squad", split="train")

            # sincronizar
            if hasattr(self.trainer.strategy, "barrier"):
                self.trainer.strategy.barrier()

            # carga en cache
            rank = self.trainer.global_rank
            print(f"[GPU {rank}] Loading dataset from cache...")

            dataset = load_dataset("squad", split="train")
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

            if rank == 0:
                print(f"[GPU 0] Preprocessing {len(dataset)} examples...")

            self.train_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
            self.train_dataset.set_format("torch")

            if rank == 0:
                print(f"[GPU 0] Dataset ready: {len(self.train_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False,
        )


class BERTSQuADModule(pl.LightningModule):
    """LightningModule que encapsula el modelo BERT para Q&A"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        learning_rate: float = 3e-5,
        num_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # calcular training steps
        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def train_with_lightning(
    config: Dict[str, Any],
    strategy: str = "ddp",
    num_gpus: int = 2,
    use_profiler: bool = False,
):

    print(f"\n{'='*70}")
    print(f"PYTORCH LIGHTNING DISTRIBUTED TRAINING")
    print(f"{'='*70}")
    print(f"Strategy: {strategy}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Batch size per GPU: {config['batch_size']}")
    print(f"Effective batch size: {config['batch_size'] * num_gpus}")
    print(f"Dataset size: {config['subset_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"{'='*70}\n")

    # Verificar GPU
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(min(num_gpus, torch.cuda.device_count())):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA not available, falling back to CPU")
        num_gpus = 1
        strategy = "auto"

    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])

    data_module = SQuADDataModule(
        tokenizer=tokenizer,
        max_length=config["max_length"],
        doc_stride=config["doc_stride"],
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    model = BERTSQuADModule(
        model_name=config["model_name"],
        learning_rate=config["learning_rate"],
        num_warmup_steps=config["num_warmup_steps"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="bert-squad-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        every_n_epochs=5,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]

    logger = TensorBoardLogger("./lightning_logs", name="bert_squad_2gpu")

    profiler = None
    if use_profiler:
        profiler = AdvancedProfiler(
            dirpath="./profiler_logs",
            filename="profile_2gpu.txt",
        )

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        deterministic=False,
        num_nodes=NUM_NODES,
    )

    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")

    start_time = time.time()
    trainer.fit(model, data_module)
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Average time per epoch: {total_time/config['num_epochs']:.2f}s")
    print(f"Final loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"{'='*70}\n", flush=True)

    return model, trainer, total_time


def compare_strategies():

    strategies = [
        ("ddp", "DDP - DistributedDataParallel (Recommended)"),
        # ("ddp_spawn", "DDP Spawn - Alternative DDP"),
        # ("fsdp", "FSDP - Fully Sharded Data Parallel"),
    ]

    results = {}

    for strategy, description in strategies:
        print(f"\n{'#'*70}")
        print(f"# Testing: {description}")
        print(f"{'#'*70}\n")

        try:
            model, trainer, total_time = train_with_lightning(
                config=CONFIG,
                strategy=strategy,
                num_gpus=NUM_GPUS,
                use_profiler=False,
            )
            results[strategy] = {
                "time": total_time,
                "final_loss": trainer.callback_metrics.get("train_loss", None),
                "success": True,
            }
        except Exception as e:
            print(f"ERROR with {strategy}: {str(e)}")
            results[strategy] = {
                "time": None,
                "final_loss": None,
                "success": False,
                "error": str(e),
            }

    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    for strategy, result in results.items():
        if result["success"]:
            print(
                f"{strategy:15} | Time: {result['time']:.2f}s | Loss: {result['final_loss']:.4f}"
            )
        else:
            print(f"{strategy:15} | FAILED: {result['error']}")
    print(f"{'='*70}\n")


def main():
    train_with_lightning(
        config=CONFIG,
        strategy=STRATEGY,
        num_gpus=NUM_GPUS,
        use_profiler=USE_PROFILER,
    )


if __name__ == "__main__":
    main()
