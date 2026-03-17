from src.trainer import DraftModelTrainer

trainer = DraftModelTrainer(
    train_output_dir="resources/trained_models/drafts_context_tokens_model",
    tokenizer_path="resources/tokenizer",
    train_dataset_path="resources/datasets/train_all_drafts.csv",
    val_dataset_path="resources/datasets/val_all_drafts.csv",
    max_length=50,
    batch_size=64,
    num_epochs=150,
    log_level="INFO"
)

trainer.train()
