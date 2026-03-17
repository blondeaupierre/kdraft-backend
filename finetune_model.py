import yaml

from src.finetuner import DraftModelFinetuner

finetuner = DraftModelFinetuner(
    model_output_dir="resources/trained_models/NFT_2025-06-01_gpt2_lol_100k",
    tokenizer_path="resources/tokenizer",
    tokenized_train_dataset_path="resources/datasets/tokenized/tokenized_train",
    tokenized_val_dataset_path="resources/datasets/tokenized/tokenized_val",
    max_length=50,
    batch_size=8,
    num_epochs=5,
    log_level="INFO"
)

finetuner.train()
