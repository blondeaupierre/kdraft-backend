import json
import matplotlib.pyplot as plt

json_path = "../../resources/trained_models/gpt2_lol_100k/checkpoint-104110/trainer_state.json"

with open(json_path, "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

train_steps = []
train_loss = []
eval_steps = []
eval_loss = []

for entry in log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label="Train Loss", color="blue")
plt.plot(eval_steps, eval_loss, label="Eval Loss", color="orange")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)
plt.show()
