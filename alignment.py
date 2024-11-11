
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from llm.llama import LlamaLLM
from prompts.prompt_templates import build_template_align
from utils import get_only_alnumspace


def extract_first_quoted_text(text):
    match = re.search(r'"(.*?)"', text)
    if match:
        return match.group(1)
    return None


def process_alignment(model, claim_text, evidence_text, claim_splits, separator="*", verbose=False):
    # Prompt
    system_text, user_text, prefix = build_template_align(claim_text, evidence_text, claim_splits, separator)
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

    # Query LLM
    res_text = prefix + model.query(prompt)

    # Remove 'Note that...' text. This is probably unique to this specific model and prompt template.
    if res_text.split('\n')[-1].lower().startswith('note that'):
        res_text = '\n'.join(res_text.split('\n')[:-1])

    # Parse claim-evidence pairs
    parsed_claim_evidence_pairs = []
    claim_splits_index = 0

    for block in res_text.split(f"\n{separator}")[1:]:
        colons_in_current_claim = claim_splits[claim_splits_index].count(":")

        # Parse claim and evidence
        if block.count(":") < colons_in_current_claim + 1:
            parsed_claim = block[:len(claim_splits[claim_splits_index])].strip()
            parsed_evidence_line = block[len(claim_splits[claim_splits_index]):].strip()

        else:
            parsed = block.split(":", colons_in_current_claim + 1)
            parsed_claim = ":".join(parsed[:-1])
            parsed_evidence_line = parsed[-1].strip()

        # Parse only lower-cased chars from evidence line
        parsed_evidence_line_alnum_lower = get_only_alnumspace(parsed_evidence_line,remove_spaces=True).lower()

        # Handle None evidence
        if parsed_evidence_line_alnum_lower.startswith("none"):
            obj = {
                'line': block,
                'claim': parsed_claim[len(separator):].strip(),
                'evidence': parsed_evidence_line,
                'signal': "N"
            }
            parsed_claim_evidence_pairs.append(obj)

        # Handle Refute
        elif parsed_evidence_line_alnum_lower.startswith("refute"):
            obj = {
                'line': block,
                'claim': parsed_claim[len(separator):].strip(),
                'evidence': " ".join(parsed_evidence_line.split(" ")[1:]),
                'signal': "R2"
            }
            parsed_claim_evidence_pairs.append(obj)

        # Handle Support
        elif parsed_evidence_line_alnum_lower.startswith("support"):
            obj = {
                'line': block,
                'claim': parsed_claim[len(separator):].strip(),
                'evidence': " ".join(parsed_evidence_line.split(" ")[1:]),
                'signal': "S2"
            }
            parsed_claim_evidence_pairs.append(obj)

        # Check again for None accuring in the middle of the text
        elif not parsed_evidence_line.startswith("\"") and "\"none\"" in parsed_evidence_line.lower():
            obj = {
                'line': block,
                'claim': parsed_claim[len(separator):].strip(),
                'evidence': parsed_evidence_line,
                'signal': 'N'
            }
            parsed_claim_evidence_pairs.append(obj)

        # Otherwise, parse claim, evidence, and signal
        else:
            if parsed_evidence_line.startswith("\"") and parsed_evidence_line.count("\"") >= 2:
                # Extract quoted text and following text
                evidence_quoted_text = extract_first_quoted_text(parsed_evidence_line)
                evidence_following_text = parsed_evidence_line[len(evidence_quoted_text)+2:]
            else:
                # Use text only as following text for signals
                evidence_quoted_text = ""
                evidence_following_text = parsed_evidence_line

            # Extract signals
            signal = None
            if 'not support' in evidence_following_text.lower():
                signal = "R"
            elif 'support' in evidence_following_text.lower() and 'refute' not in evidence_following_text.lower():
                signal = "S"
            elif 'refute' in evidence_following_text.lower() and 'support' not in evidence_following_text.lower():
                signal = "R"
            elif 'same entity' in evidence_following_text.lower():
                signal = "E"

            # We need either evidence text or signal
            if not evidence_quoted_text and not signal:
                # Couldn't align
                obj = {
                    'line': block,
                    'claim': parsed_claim[len(separator):].strip(),
                    'evidence': "none-error",
                    'signal': "N"
                }
            else:
                obj = {
                    'line': block,
                    'claim': parsed_claim[len(separator):].strip(),
                    'evidence': evidence_quoted_text.strip(),
                    'signal': signal
                }
            parsed_claim_evidence_pairs.append(obj)

        claim_splits_index += 1
        if claim_splits_index >= len(claim_splits):
            break

    return parsed_claim_evidence_pairs


if __name__ == "__main__":
    model = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    claim_text = "Oliver Twist is a timeless story by Charles Dickens"
    evidence_text = "Oliver Twist; or, The Parish Boy's Progress, is the second novel by English author Charles Dickens."
    claim_splits = ['Oliver Twist is a timeless story', 'by Charles Dickens']
    separator = "*"

    alignments = process_alignment(
        model=model,
        claim_text=claim_text,
        evidence_text=evidence_text,
        claim_splits=claim_splits,
        separator=separator,
        verbose=False
    )
    for alignment in alignments:
        print("-"*10)
        print(alignment)
