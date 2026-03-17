from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.generator import DraftModelGenerator
from src.utils.team_perplexity import build_draft_sequence

generator = DraftModelGenerator(
    model_path="resources/trained_models/FT_top_teams_offline_2025-06-01_gpt2_lol_100k",
    tokenizer_path="resources/tokenizer",
    champions_path="resources/vocab/champions.txt",
    draft_tokens_path="resources/vocab/draft_tokens.txt",
    draft_max_length=50
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
        patch=input.PATCH,
        draft_sequence=input.draft_sequence,
        top_teams_json_path="resources/data/top_teams_offline_2025-06-01.json"
    )

    outputs = generator.generate_sequence(partial_sequence=full_sequence)
    result_topk = generator.compute_topk(outputs, 3)
    print(result_topk)

    return result_topk
