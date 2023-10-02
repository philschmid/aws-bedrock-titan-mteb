import boto3
import json
import numpy as np
from time import sleep
from tqdm import tqdm


# https://github.com/embeddings-benchmark/mteb/tree/4d75ddf448c93b4b879e60e110061f7dcf76ae42#using-a-custom-model
class BedrockTitanEmbedding:
    def __init__(self, model="amazon.titan-embed-text-v1", profile=None, with_sleep=False) -> None:
        self.model_id = model
        session = boto3.Session(region_name="us-east-1", profile_name=profile)
        self.bedrock = session.client("bedrock-runtime")
        self.with_sleep = with_sleep

    def get_embeddings(self, sentence):
        """Call the model to get the embeddings for the given sentences."""
        input_body = {"inputText": sentence}
        response = self.bedrock.invoke_model(
            body=json.dumps(input_body),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        np_array = np.array(response_body.get("embedding"))
        if self.with_sleep:
            sleep(
                0.1
            )  # add sleep to avoid throttling https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html
        return np_array

    def encode(self, sentences, batch_size=1, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        return np.array(
            [self.get_embeddings(sentences[idx]) for idx in tqdm(range(0, len(sentences)), desc="encode")],
        )


if __name__ == "__main__":
    model = BedrockTitanEmbedding(profile="hf-sm")
    r = model.encode("Hello world")
