import pandas as pd
import math
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, Trainer, DataCollatorForLanguageModeling
from typing import Dict, Optional
import logging


class DraftModelEvaluator:
    """A class to evaluate a GPT-2 model on a draft sequence dataset."""

    def __init__(
            self,
            model_path: str,
            tokenizer_path: str,
            test_dataset_path: str,
            max_length: int,
            device: str = None,
            log_level: str = "INFO"
    ):
        # Set up logger
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        # Validate inputs
        if not all([model_path, tokenizer_path, test_dataset_path]):
            raise ValueError("model_path, tokenizer_path, and dataset_path must be provided")

        # Load model and tokenizer
        self.logger.info(f"Loading model from {model_path} and tokenizer from {tokenizer_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.max_length = max_length

        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")

        # Load and prepare dataset
        self.logger.info(f"Loading dataset from {test_dataset_path}")
        try:
            test_df = pd.read_csv(test_dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {test_dataset_path}")
        if "sequence" not in test_df.columns:
            raise ValueError("Dataset must contain a 'sequence' column")

        self.test_dataset = Dataset.from_pandas(test_df)
        self.test_dataset = self.test_dataset.map(
            self._encode,
            remove_columns=["sequence"],
            desc="Encoding test dataset"
        )
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Initialize data collator and trainer
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator
        )

    def _encode(self, examples):
        """Encode dataset examples for language modeling."""
        return self.tokenizer(
            examples["sequence"].split(','),
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def evaluate(self):
        """Evaluate the model on the test dataset and return metrics."""
        self.logger.info("Starting evaluation")
        results = self.trainer.evaluate(eval_dataset=self.test_dataset)
        if "eval_loss" in results:
            perplexity = math.exp(results["eval_loss"])
            self.logger.info(f"Evaluation completed. Perplexity: {perplexity:.2f}")
        else:
            self.logger.warning("No eval_loss found in results")
        return results

    def save_results(self, output_path: Optional[str] = None):
        """Save evaluation results to a file or print to console."""
        results = self.evaluate()
        output_str = f"Test results: {results}"
        if output_path:
            self.logger.info(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                f.write(output_str)
        else:
            print(output_str)
