from src.generator import DraftModelGenerator

generator = DraftModelGenerator(
    model_path="resources/trained_models/FT_top_teams_offline_2025-06-01_gpt2_lol_100k",
    tokenizer_path="resources/tokenizer",
    champions_path="resources/vocab/champions.txt",
    draft_tokens_path="resources/vocab/draft_tokens.txt",
    draft_max_length=50
)
outputs = generator.generate_sequence("[AS_TEAM],Gen.G,[VS_TEAM],Hanwha Life Esports,[SIDE],RED,[PATCH],25.17,<BOS>,[BLUE_BAN1],Galio,[RED_BAN1],Azir,[BLUE_BAN2],Yunara,[RED_BAN2],Wukong,[BLUE_BAN3],Pantheon,[RED_BAN3],JarvanIV,[BLUE_PICK1],Orianna,")
result_topk = generator.compute_topk(outputs, 3)
print(result_topk)
