from argparse import ArgumentParser, Namespace
from typing import List, Optional

from titan_mteb_model import BedrockTitanEmbedding
from mteb import MTEB

TASK_LIST_RETRIEVAL = [
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "SCIDOCS",
]

RETRIEVAL_TASKS = [
    # "SciFact",
    "ArguAna",
    "ClimateFEVER",
    "FiQA2018",
    "HotpotQA",
    "QuoraRetrieval",
    "Touche2020",
    "TRECCOVID",
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

    for task in RETRIEVAL_TASKS:
        print(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, batch_size=1, output_folder=args.output_dir, eval_splits=eval_splits)
