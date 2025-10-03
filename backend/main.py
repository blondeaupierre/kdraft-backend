import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.generator import DraftModelGenerator

# Charger config
with open("config/generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Instancier ton générateur une seule fois (au lancement du serveur)
generator = DraftModelGenerator(
    model_path=config["model_path"],
    tokenizer_path=config["tokenizer_path"],
    champions_path=config["champions_path"],
    draft_tokens_path=config["draft_tokens_path"],
    draft_max_length=config["draft_max_length"]
)

# Créer app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # l'origine de ton front Angular
    allow_credentials=True,
    allow_methods=["*"],  # autorise POST, GET, etc.
    allow_headers=["*"],  # autorise tous les headers
)


@app.get("/health")
def health():
    return {"status": "ok"}


class Input(BaseModel):
    draft_sequence: str


@app.post("/generate")
def generate(input: Input):
    outputs = generator.generate_sequence(partial_sequence=input.draft_sequence)
    result_topk = generator.compute_topk(outputs, config["top_k"])
    return result_topk
