import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.generator import DraftModelGenerator
from src.utils.team_perplexity import build_draft_sequence

# Charger config
with open("config/generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)

generator = DraftModelGenerator(
    model_path=config["model_path"],
    tokenizer_path=config["tokenizer_path"],
    champions_path=config["champions_path"],
    draft_tokens_path=config["draft_tokens_path"],
    draft_max_length=config["draft_max_length"]
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # origin from angular front end
    allow_credentials=True,
    allow_methods=["*"],  # autorize POST, GET, etc.
    allow_headers=["*"],  # autorize all headers
)


@app.get("/health")
def health():
    return {"status": "ok"}


class Input(BaseModel):
    AS_TEAM: str
    VS_TEAM: str
    SIDE: str
    PATCH: str
    draft_sequence: str


@app.post("/generate")
def generate(input: Input):
    full_sequence = build_draft_sequence(
        as_team=input.AS_TEAM,
        vs_team=input.VS_TEAM,
        side=input.SIDE,
        patch=input.PATCH, draft_sequence=input.draft_sequence,
        team_vocab_path=config["team_vocab_path"]
    )

    outputs = generator.generate_sequence(partial_sequence=full_sequence)
    result_topk = generator.compute_topk(outputs, config["top_k"])

    return result_topk
