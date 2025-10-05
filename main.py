import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import time
from tqdm.auto import tqdm
import json
import os

import profiling


class SQuADTrainer:
    def __init__(self, model_name="bert-base-uncased", max_length=384, batch_size=8, num_epochs=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
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
            'epochs': [],
            'loss_per_epoch': [],
            'time_per_epoch': [],
            'loss_per_step': []
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
            
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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
    
    def train(self):
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
        print(f"{'='*70}\n")
        
        global_step = 0
        total_train_time = 0
        all_losses = []
        
        self.model.train()
        overall_start = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_losses = []
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
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
                
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_loss/(step+1):.4f}',
                    'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                    'step_time': f'{step_time:.2f}s'
                })
                
                global_step += 1
            
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['loss_per_epoch'].append(avg_epoch_loss)
            self.training_history['time_per_epoch'].append(epoch_time)
            
            print(f"Average Loss: {avg_epoch_loss:.4f}")
        
        total_train_time = time.time() - overall_start
        
        
        self.save_training_metrics(total_train_time, global_step)
        
        return self.model, {
            'total_time': total_train_time,
            'epochs': self.num_epochs,
            'final_loss': avg_epoch_loss,
            'history': self.training_history
        }
    
    def save_training_metrics(self, total_time, total_steps):
        metrics = {
            'model': 'bert-base-uncased',
            'dataset': 'SQuAD',
            'configuration': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'max_length': self.max_length,
                'learning_rate': 3e-5
            },
            'results': {
                'total_training_time_seconds': total_time,
                'total_training_time_minutes': total_time / 60,
                'average_time_per_epoch_seconds': total_time / self.num_epochs,
                'total_steps': total_steps,
                'average_time_per_step_seconds': total_time / total_steps,
            },
            'history': self.training_history
        }
        
        os.makedirs('training_results', exist_ok=True)
        with open('training_results/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training metrics saved to: training_results/training_metrics.json\n")
    
    def save_model(self, output_dir="./bert_squad_model"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    config = {
        "model_name": "bert-base-uncased",
        "max_length": 384,
        "batch_size": 8,  # Decrease for less GPU memory, increase for faster training
        "num_epochs": 3,  # Increase for longer training time
    }
    
    trainer = SQuADTrainer(**config)
    

    profiling.start_profiling(trainer.train, "./profile")   
    
    trainer.save_model()
    
if __name__ == "__main__":
    main()
