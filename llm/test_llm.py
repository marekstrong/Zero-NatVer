
import sys, os
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{cur_dir}/../")  # for prompts


import re
import unittest
import warnings

from llama import LlamaLLM
from prompts.prompt_templates import build_template_chunk, build_template_align


class TestLlamaQueries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=FutureWarning)
        cls.llm = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    def test_textgeneration(self):
        prompt = {
            'type': "text-generation",
            'system_text': "You are a helpful assistant.",
            'user_text': "What is 4*6?",
            'params': {
              'temperature': 0.0,
              'top_p': 1.0,
              'max_new_tokens': 128,
              'stop_tokens': None
            },
            'prefix': None
        }
        res = self.llm.query(prompt)

        self.assertTrue("24" in res)

    def test_hypotheses(self):
        prompt = {
            'type': "hypotheses",
            'system_text': "You are a helpful assistant.",
            'user_text': "What is the capital city of France?",
            'params': {
              'temperature': 0.0,
              'top_p': 1.0,
              'max_new_tokens': 10,
              'stop_tokens': None
            },
            'prefix': None,
            'hypotheses': ['London', 'New York', 'Paris', 'Prague']
        }
        res = self.llm.query(prompt)
        res_sorted = list(sorted(res, key=lambda x: x[1]['log_total'], reverse=True))
        best_res_sorted = res_sorted[0]

        self.assertTrue("Paris" in best_res_sorted)

    def test_yesno(self):
        prompt = {
            'type': "yes/no",
            'system_text': "You are a helpful assistant.",
            'user_text': "The capital city of Tajikistan is Dushanbe. Yes or No?",
            'params': {
              'temperature': 1.0,
              'top_p': 1.0,
              'max_new_tokens': 10,
              'stop_tokens': None
            },
            'prefix': None
        }
        res = self.llm.query(prompt)[0]
        self.assertTrue(res)

    def test_followandbreak(self):
        claim = "Oliver Twist is a timeless story by Charles Dickens"

        system_text, user_text = build_template_chunk(claim)

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
            'follow': claim,
            'separator': '*'
        }
        res = self.llm.query(prompt)

        # Parse
        res = [n.strip() for n in res.strip().split('*') if n]

        # Compare
        exected = ['Oliver Twist is a timeless story', 'by Charles Dickens']
        self.assertListEqual(exected, res)

    def test_proofalign(self):
        claim = "Oliver Twist is a timeless story by Charles Dickens"
        evidence = "Oliver Twist; or, The Parish Boy's Progress, is the second novel by English author Charles Dickens."
        expressions = ['Oliver Twist is a timeless story', 'by Charles Dickens']
        separator = "*"

        system_text, user_text, prefix = build_template_align(claim, evidence, expressions, separator)

        prompt = {
            'type': "text-generation",
            'system_text': system_text,
            'user_text': user_text,
            'params': {
                'temperature': 0.0,
                'top_p': 1.0,
                'max_new_tokens': 2048,
                'stop_tokens': None
            },
            'prefix': prefix
        }
        res = prefix + self.llm.query(prompt)

        # Claim splits are in the output
        for claim_split in expressions:
            self.assertTrue(claim_split in res)

        # Aligned text must appear in the evidence text
        for block in res.split(f"\n{separator}"):
            aligned_text = block.split(':', 1)[1]
            if not aligned_text.strip():
                continue
            match = re.search(r'"([^"]+)"', aligned_text)
            first_quoted_text = match.group(1)
            self.assertTrue(first_quoted_text in evidence)


if __name__ == '__main__':
    unittest.main()
