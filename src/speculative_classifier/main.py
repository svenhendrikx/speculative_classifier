import torch.nn.functional as F
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich import print

from functools import reduce

import logging
from data import PROMPT, TEST_CASES
logging.basicConfig(level=logging.INFO)


class MyPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs):
        return super().preprocess(PROMPT.format(inputs), **tokenizer_kwargs)

    def postprocess(self, logits):
        probs = F.softmax(logits[0][0, -1, :],
                          dim=-1,
                          )

        match_tokens = {'positive': 33733,
                        'negative': 36183,
                        'neutral': 25627,
                        }

        prediction = list(reduce(lambda acc, cur: acc if probs[acc[1]] > probs[cur[1]] else cur,
                                 match_tokens.items(),
                                 )
                          )[0]
        return prediction


model_name = "google/gemma-2b"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained(model_name)


def run_testcases(tcs: list,
                  model,
                  tokenizer,
                  ):

    pipe = MyPipeline(model=model,
                      tokenizer=tokenizer,
                      )
    results = list(map(lambda tc: pipe(tc['text'])[0] == tc['expected_sentiment'],
                       tcs,
                       )
                   )

    return sum(results) / len(results)


print(run_testcases(tcs=TEST_CASES,
                    model=model,
                    tokenizer=tokenizer,
                    )
      )
