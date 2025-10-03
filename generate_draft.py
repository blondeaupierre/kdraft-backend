import yaml

from src.generator import DraftModelGenerator

with open("config/generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)

generator = DraftModelGenerator(
    model_path=config["model_path"],
    tokenizer_path=config["tokenizer_path"],
    champions_path=config["champions_path"],
    draft_tokens_path=config["draft_tokens_path"],
    draft_max_length=config["draft_max_length"]
)
outputs = generator.generate_sequence("[AS_TEAM],Gen.G,[VS_TEAM],Hanwha Life Esports,[SIDE],RED,[PATCH],25.17,<BOS>,[BLUE_BAN1],Galio,[RED_BAN1],Azir,[BLUE_BAN2],Yunara,[RED_BAN2],Wukong,[BLUE_BAN3],Pantheon,[RED_BAN3],JarvanIV,[BLUE_PICK1],Orianna,")
result_topk = generator.compute_topk(outputs, config["top_k"])
print(result_topk)
