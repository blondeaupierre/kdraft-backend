import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, EarlyStoppingCallback, GPT2Config, GPT2TokenizerFast
import logging
from typing import Dict, Optional


class DraftModelFinetuner:
    """A class to train a GPT-2 model on a draft sequence dataset."""

    def __init__(
            self,
            model_output_dir: str,
            tokenizer_path: str,
            tokenized_train_dataset_path: str,
            tokenized_val_dataset_path: str,
            max_length: int,
            batch_size: int,
            num_epochs: int,
            device: str = None,
            log_level: str = "INFO"
    ):
        # Set up logger
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        self.model_output_dir = model_output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        #TODO
        # Charger GPT-2 Small depuis Hugging Face
        # self.logger.info("Loading GPT-2 Small from 'gpt2'")
        # self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        model_path = "resources/trained_models/gpt2_lol_100k"
        self.logger.info(f"Loading model from {model_path} and tokenizer from {tokenizer_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

        # Charger le tokenizer personnalisé
        self.logger.info(f"Loading tokenizer from {tokenizer_path}")

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Redimensionner les embeddings du modèle pour correspondre au tokenizer
        self.logger.info(f"Resizing model embeddings to match tokenizer vocab size: {len(self.tokenizer)}")
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tokenized_train_dataset_path = tokenized_train_dataset_path
        self.tokenized_val_dataset_path = tokenized_val_dataset_path

        self.trainer = self._build_trainer()

    def _load_datasets(self):
        self.logger.info("Loading pre-tokenized datasets from disk")
        train_dataset = load_from_disk(self.tokenized_train_dataset_path)
        val_dataset = load_from_disk(self.tokenized_val_dataset_path)
        return train_dataset, val_dataset

    def _build_trainer(self):
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        train_dataset, val_dataset = self._load_datasets()
        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.model_output_dir,
                num_train_epochs=self.num_epochs,
                learning_rate=5e-6,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,  # limit to 3 checkpoints
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                fp16=True,
                weight_decay=0.01,
                warmup_ratio=0.05,
                lr_scheduler_type="cosine"
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def train(self):
        self.logger.info("Starting training on initial dataset")
        self.trainer.train()
        self.logger.info("Training completed")

        # Save model and tokenizer
        self.model.save_pretrained(self.model_output_dir)