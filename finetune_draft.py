import yaml

from src.finetuner import DraftModelFinetuner

with open("config/config_finetune.yaml", "r") as f:
    config = yaml.safe_load(f)

finetuner = DraftModelFinetuner(
    model_path="resources/trained_models/60000_drafts_models",
    model_output_dir="resources/trained_models/model_from_25_16",
    tokenizer_path=config["tokenizer_path"],
    train_dataset_path=config["train_dataset_path"],
    val_dataset_path=config["val_dataset_path"],
    max_length=config["max_length"],
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"],
    log_level="INFO"
)

finetuner.train()
