import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, EarlyStoppingCallback, GPT2Config
import logging


class DraftModelTrainer:
    """A class to train a GPT-2 model on a draft sequence dataset."""

    def __init__(
            self,
            train_output_dir: str,
            tokenizer_path: str,
            train_dataset_path: str,
            val_dataset_path: str,
            max_length: int,
            batch_size: int,
            num_epochs: int,
            device: str = None,
            log_level: str = "INFO"
    ):
        # Set up logger
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        self.train_output_dir = train_output_dir
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path

        self.train_dataset, self.val_dataset = self._load_datasets()
        self.model = self._build_model()
        self.trainer = self._build_trainer()

    def _load_datasets(self):
        self.logger.info(f"Loading datasets from {self.train_dataset_path} and {self.val_dataset_path}")

        train_df = pd.read_csv(self.train_dataset_path)
        val_df = pd.read_csv(self.val_dataset_path)

        if "sequence" not in train_df.columns or "sequence" not in val_df.columns:
            raise ValueError("Datasets must contain a 'sequence' column")

        train_dataset = Dataset.from_pandas(train_df).map(
            self._encode,
            remove_columns=["sequence"],
            desc="Encoding training dataset",
            batched = True
        )
        val_dataset = Dataset.from_pandas(val_df).map(
            self._encode,
            remove_columns=["sequence"],
            desc="Encoding validation dataset",
            batched=True
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        return train_dataset, val_dataset

    def _encode(self, examples):
        return self.tokenizer(
            [seq.split(',') for seq in examples["sequence"]],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def _build_model(self):
        # Load model and tokenizer
        config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_positions=self.max_length,
            n_embd=256,
            n_layer=8,
            n_head=4,
        )
        model = GPT2LMHeadModel(config)
        model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")
        return model

    def _build_trainer(self):
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.train_output_dir,
                num_train_epochs=self.num_epochs,
                learning_rate=1e-4,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,  # required if using early stopping
                metric_for_best_model="eval_loss",  # monitor validation loss
            ),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

    def train(self):
        self.logger.info("Starting training")
        self.trainer.train()
        self.model.save_pretrained(self.train_output_dir)
        self.logger.info("Training completed")
