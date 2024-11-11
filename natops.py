
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from llm.llama import LlamaLLM
from prompts.prompt_natops import get_prompt_templates


NATOP_SYMBOLS = [
    '=',
    '<',
    '>',
    '!',
    '|'
]


def determine_natops_templates(model, alignments, limit_templates=None, verbose=False):
    alignments_with_natops = []

    for alignment in alignments:
        sym2res = {}
        for symbol in NATOP_SYMBOLS:
            # Get prompts for a specific symbol
            prompts = get_prompt_templates(alignment['claim'], alignment['evidence'], symbol)

            # Limit the number of prompts
            if limit_templates:
                prompts = prompts[:limit_templates]

            # Get responses for all prompts
            sym2res[symbol] = []
            for prompt_idx, prompt in enumerate(prompts):
                # Query LLM
                _, res_probs = model.query(prompt)

                # Store
                sym2res[symbol].append({k: round(v, 5) for k, v in res_probs.items()})

        # Store
        alignments_with_natops.append((alignment['claim'], alignment['evidence'], sym2res, alignment['signal']))

    return alignments_with_natops


if __name__ == "__main__":
    import pprint
    model = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    alignments = [
        {'claim': 'Oliver Twist is a timeless story', 'evidence': "Oliver Twist; or, The Parish Boy's Progress", 'signal': 'E'},
        {'claim': 'by Charles Dickens', 'evidence': 'by English author Charles Dickens', 'signal': 'S'}
    ]

    assigned_natops = determine_natops_templates(
        model=model,
        alignments=alignments
    )

    for an in assigned_natops:
        print("-"*10)
        pprint.pp(an)
