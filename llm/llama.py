
import sys, os
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{cur_dir}/../")  # for log_config

import argparse
import json
import logging
import socket
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_config import get_logger, set_global_log_level
from llm.generate_proofs import GenerateProofs


logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class LlamaLLM:

    def __init__(self, model_id, device="cuda", debug=False):
        self.model_id = model_id
        self.device = device

        if debug:
            set_global_log_level(logging.DEBUG)
        else:
            set_global_log_level(logging.ERROR)

        self.tokenizer, self.model = self.initialise_model(model_id=self.model_id)

    def initialise_model(self, model_id):
        s_time = time.time()
        logger.info(f"> Loading LLM: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        model.eval()

        logger.info(f"Done. (t={(time.time() - s_time):.2f}s)")
        return tokenizer, model

    def build_prompt(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        return prompt

    def query(self, data):
        if 'type' not in data:
            data['type'] = "text-generation"

        # Build prompt
        prompt = self.build_prompt(
            system=data['system_text'],
            user=data['user_text']
        )

        # Infer
        g = GenerateProofs(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            generation_params=data['params']
        )

        # Process according to the query type
        if data['type'] == "text-generation":
            return g.infer_sample(prompt=prompt, prefix=data.get('prefix', None))
        elif data['type'] == "hypotheses":
            return g.infer_hypotheses(prompt=prompt, prefix=data.get('prefix', None), hypotheses=data['hypotheses'])
        elif data['type'] == "yes/no":
            return g.infer_yesno(prompt=prompt, prefix=data.get('prefix', None), return_token_probs=data.get('return_token_probs', True))
        elif data['type'] == "followandbreak":
            return g.infer_followandbreak(prompt=prompt, prefix=data.get('prefix', None), follow=data["follow"], separator=data['separator'])
        elif data['type'] == "proofalign":
            return g.infer_align(prompt=prompt, prefix=data.get('prefix', None), chunks=data["chunks"], separator=data['separator'], end_strings=data['end_strings'])
        else:
            raise Exception(f"Type {data['type']} not supported!")

        return None


if __name__ == "__main__":
    llm = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct", debug=True)

    prompt = {
        'type': "text-generation",
        'system_text': "You are a helpful assistant.",
        'user_text': "What is 4*6?",
        'params': {
          'temperature': 0.7,
          'top_p': 1.0,
          'max_new_tokens': 128,
          'stop_tokens': None
        },
        'prefix': None
    }
    res = llm.query(prompt)
    print(res)
