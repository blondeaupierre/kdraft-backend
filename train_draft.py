import yaml
from src.trainer import DraftModelTrainer

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

trainer = DraftModelTrainer(
    train_output_dir=config["model_path"],

    tokenizer_path=config["tokenizer_path"],
    train_dataset_path=config["train_dataset_path"],
    val_dataset_path=config["val_dataset_path"],
    max_length=config["max_length"],
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"],
    log_level="INFO"
)

trainer.train()
