import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from rich import print

from functools import reduce, partial
from typing import List
import operator

import logging
from data import PROMPT, TEST_CASES
logging.basicConfig(level=logging.INFO)

# Define test cases

model_name = "MBZUAI/LaMini-GPT-1.5B"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained(model_name)


def sentiment_probabilities(sentence: str,
                            model,
                            tokenizer,
                            ):
    prompt = PROMPT.format(sentence)

    # Encode a text inputs
    indexed_tokens = tokenizer.encode(prompt)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict next token
    with torch.no_grad():
        outputs = model(tokens_tensor)

    # Apply softmax on the last dimension to convert logits to probabilities
    probs = F.softmax(outputs.logits[0, -1, :],
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


def run_testcases(tcs: list,
                  model,
                  tokenizer,
                  ):

    _classify = partial(sentiment_probabilities,
                        model=model,
                        tokenizer=tokenizer,
                        )
    results = list(map(lambda tc: _classify(tc['text']) == tc['expected_sentiment'],
                       tcs,
                       )
                   )

    return sum(results) / len(results)


print(run_testcases(tcs=TEST_CASES,
                    model=model,
                    tokenizer=tokenizer,
                    )
      )
