from contextlib import nullcontext
import torch
import time
import os

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import ProfilerActivity

from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from datasets import load_dataset

from tqdm.auto import tqdm

CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 384,
    "batch_size": 8,
    "num_epochs": 10,
}


activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

TRAINING_PROFILING_RUNS = {
    "With profiler": {
        "profiler": torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=2, warmup=100, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/profile/"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ),
        "save_profiler_time_table": True,
        "save_tensorboard_metrics": False,
        "save_model": False,
    },
    "No profiler": {
        "profiler": None,
        "save_profiler_time_table": False,
        "save_tensorboard_metrics": True,
        "save_model": True,
    },
}

writer = SummaryWriter()


class SQuADTrainer:
    def __init__(
        self, model_name="bert-base-uncased", max_length=384, batch_size=4, num_epochs=3
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # Iniciar modelo y tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

        # Parametros
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.doc_stride = 128

        self.training_history = {
            "epochs": [],
            "loss_per_epoch": [],
            "time_per_epoch": [],
            "loss_per_step": [],
        }

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

    def prepare_data(self):
        print("\nLoading SQuAD dataset...")
        dataset = load_dataset("squad", split="train")

        subset_size = 10000
        dataset = dataset.select(range(min(subset_size, len(dataset))))

        print(f"Preprocessing {len(dataset)} examples...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def train(
        self,
        profiler=None,
        save_profiler_time_table: bool = False,
        save_tensorboard_metrics: bool = False,
    ):
        train_dataset = self.prepare_data()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = AdamW(self.model.parameters(), lr=3e-5)
        num_training_steps = len(train_dataloader) * self.num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Model: BERT-Base-uncased")
        print(f"Dataset: SQuAD")
        print(f"Number of examples: {len(train_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Total training steps: {num_training_steps}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Learning rate: 3e-5")
        print(f"{'='*70}\n", flush=True)

        global_step = 0
        total_train_time = 0
        all_losses = []

        start_time = time.time()
        self.model.train()

        with profiler if profiler is not None else nullcontext() as prof:
            for epoch in range(self.num_epochs):
                epoch_start_time = time.time()
                epoch_loss = 0
                epoch_losses = []

                for step, batch in enumerate(train_dataloader):
                    print(
                        f"Epoch[{epoch}/ {self.num_epochs}]({step}/{num_training_steps // self.num_epochs})",
                        end="\r",
                        flush=True,
                    )

                    step_start_time = time.time()

                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(**batch)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    step_time = time.time() - step_start_time
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    epoch_losses.append(loss_value)
                    all_losses.append(loss_value)
                    learning_rate = lr_scheduler.get_last_lr()[0]

                    # Métricas
                    if save_tensorboard_metrics:
                        writer.add_scalar("Loss", loss_value, global_step)
                        writer.add_scalar("Step time", step_time, global_step)
                        writer.add_scalar("Learning rate", learning_rate, global_step)

                    if profiler != None:
                        prof.step()

                    global_step += 1

                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / len(train_dataloader)

                self.training_history["epochs"].append(epoch + 1)
                self.training_history["loss_per_epoch"].append(avg_epoch_loss)
                self.training_history["time_per_epoch"].append(epoch_time)

            total_train_time = time.time() - start_time

            if save_tensorboard_metrics:
                writer.add_text("total_time_s", f"{total_train_time}")
                writer.add_text("final_loss", f"{avg_epoch_loss}")
                writer.add_text("batch_size", f"{self.batch_size}")
                writer.add_text("num_epochs", f"{self.num_epochs}")

                writer.flush()

            if save_profiler_time_table and profiler != None:
                os.makedirs("./runs/profile", exist_ok=True)
                with open("runs/profile/profiler_summary.txt", "w") as f:
                    f.write(
                        prof.key_averages().table(
                            sort_by=(
                                "cuda_time_total"
                                if torch.cuda.is_available()
                                else "cpu_time_total"
                            ),
                            row_limit=40,
                        )
                    )

            return self.model, {
                "total_time": total_train_time,
                "epochs": self.num_epochs,
                "final_loss": avg_epoch_loss,
                "history": self.training_history,
            }

    def save_model(self, output_dir="./bert_squad_model"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


def main():

    for iteration_name, iter_config in TRAINING_PROFILING_RUNS.items():
        print(f"\n\nRUNNING ITERATION: {iteration_name}\n\n")
        trainer = SQuADTrainer(**CONFIG)

        trainer.train(
            profiler=iter_config["profiler"],
            save_tensorboard_metrics=iter_config["save_tensorboard_metrics"],
            save_profiler_time_table=iter_config["save_profiler_time_table"],
        )

        if iter_config["save_model"]:
            trainer.save_model(
                output_dir=f"./bert_squad_model_{iteration_name.replace(' ', '_')}"
            )


if __name__ == "__main__":
    main()
