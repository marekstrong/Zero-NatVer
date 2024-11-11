
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from llm.llama import LlamaLLM
from prompts.prompt_templates import build_template_chunk
from utils import get_only_alnumspace


def process_chunking(model, claim_text, break_coeff, verbose=False):

    def is_only_symbols(input_string):
        for char in input_string:
            if char.isalnum():
                return False
        return True

    separator = "*"

    # Prompt
    system_text, user_text = build_template_chunk(claim_text)
    prompt = {
        'type': "followandbreak",
        'system_text': system_text,
        'user_text': user_text,
        'params': {
          'temperature': 0.0,
          'top_p': 1.0,
          'max_new_tokens': 1024,
          'stop_tokens': None
        },
        'prefix': None,
        'follow': claim_text,
        'separator': '*'
    }

    # Query LLM
    res_text = model.query(prompt)

    # Fix cases where newline characters are not followed by separators
    res_text_lines = res_text.split('\n')
    for i in range(len(res_text_lines)):
        if not res_text_lines[i].startswith(separator):
            res_text_lines[i] = separator + res_text_lines[i]
    res_text = '\n'.join(res_text_lines)

    # Parse
    parsed = []
    for r in res_text.split(f'\n{separator}'):
        r = r.strip()
        if r.startswith(separator):
            r = r[len(separator):].strip()

        if not r:
            continue

        # Ignore symbol-only splits
        if is_only_symbols(r):
            continue

        if r[-1] in [',', '.', '!', '?', ';', ':']:
            r = r[:-1]
        if r:
            parsed.append(r)

    # Check that claim splits match the original input claim text
    claim_text_alnum = get_only_alnumspace(" ".join(claim_text), remove_spaces=True)
    claim_splits_alnum = get_only_alnumspace(" ".join(parsed), remove_spaces=True)
    if claim_text_alnum != claim_splits_alnum:
        cexception("Claim splits do not match the input claim text!", use_embed=True, local_vars={**locals(), **globals()})

    return parsed


if __name__ == "__main__":
    model = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    claim_text = "Oliver Twist is a timeless story by Charles Dickens"

    claim_splits = process_chunking(
        model=model,
        claim_text=claim_text,
        break_coeff=1.0,
        verbose=False
    )
    print("-"*10)
    print(claim_splits)
