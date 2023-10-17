from argparse import ArgumentParser, Namespace
from typing import List, Optional

from titan_mteb_model import BedrockTitanEmbedding
from mteb import MTEB

# https://huggingface.co/spaces/mteb/leaderboard

TASKS = [
    ## Classification
    # "Banking77Classification", ✅
    # "AmazonCounterfactualClassification", ✅
    # Clustering 
    # "StackExchangeClusteringP2P",
    # Ranking 
    # "SciDocsRR",
    #STS
    # "STS22"
    ## Retrieval
    # "NQ", 
    # "SciFact", ✅
    # "ArguAna", ✅
    # "ClimateFEVER", # too  big
    # "HotpotQA", # too big
    # "Touche2020",
    "TRECCOVID",
    "FiQA2018",
    "MSMARCO"
    # "QuoraRetrieval", # too big
]


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="amazon.titan-embed-text-v1", help="Bedrock Model id")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--aws_profile", type=str, default="hf-sm")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    model = BedrockTitanEmbedding(model=args.model, profile=args.aws_profile)

    for task in TASKS:
        print(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, batch_size=1, output_folder=args.output_dir, eval_splits=eval_splits)
