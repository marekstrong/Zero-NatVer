
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from alignment import process_alignment
from chunking import process_chunking
from llm.llama import LlamaLLM
from natops import determine_natops_templates
from utils import load_jsonl, remove_substrings_absent_from_original


def process_chunk_and_align(model, claim_text, evidence_text, break_coeff, separator="*", verbose=False):
    # Remove new lines that would cause problems during alignment
    claim_text = claim_text.replace('\n', '')
    evidence_text = evidence_text.replace('\n', '')

    # STEP1: CHUNKING
    claim_splits = process_chunking(
        model=model,
        claim_text=claim_text,
        break_coeff=break_coeff,
        verbose=verbose
    )

    # STEP2: ALIGNMENT
    alignments = process_alignment(
        model=model,
        claim_text=claim_text,
        evidence_text=evidence_text,
        claim_splits=claim_splits,
        separator=separator,
        verbose=verbose
    )

    return alignments


def process_single_pass(args, model, claim_id, claim_text, evidence_text, language=None, claim_splits=None):
    # Chunk and align
    alignments = process_chunk_and_align(
        model=model,
        claim_text=claim_text,
        evidence_text=evidence_text,
        break_coeff=args.break_coeff,
        verbose=args.verbose
    )

    # Check for 'None alignment'
    all_aligned = True
    for alignment in alignments:
        if alignment['evidence'].lower().startswith("none") or alignment['signal'] == 'N':
            all_aligned = False
            break

    # Post-constraining
    if args.align_constrains_type and args.align_constrains_type == "post":
        for n in range(len(alignments)):
            if alignments[n]['signal'] == "N":
                continue
            alignments[n]['evidence'] = remove_substrings_absent_from_original(evidence_text, alignments[n]['evidence'])

    # Output object
    result = {
        'claim': claim_text,
        'evidence': evidence_text,
        'alignments': alignments
    }

    if all_aligned:
        # NatOp Assignments
        assigned_natops = determine_natops_templates(
            model=model,
            alignments=alignments,
            limit_templates=args.limit_templates,
            verbose=args.verbose
        )
        result['natops'] = assigned_natops
    else:
        result['natops'] = "None Alignment"

    return result


def process(args, model, data):
    # Process instance
    natops = []
    if args.multilang:
        language = data['language']
    else:
        language = None
    pipeline_data = process_single_pass(args, model, data['id'], data[args.claim_location], data[args.evidence_location], language=language)
    natops.append(pipeline_data['natops'])

    # Store
    data['alignments'] = pipeline_data['alignments']
    data['natops'] = natops

    return data


def parse_args():
    parser = argparse.ArgumentParser(prog="Zero-NatVer")
    parser.add_argument('-j', '--jsonl', help="jsonl file")
    parser.add_argument('--claim-location', default="claim", help="comma-separated json location of claim text")
    parser.add_argument('--evidence-location', default="evidence", help="comma-separated json location of evidence text")
    parser.add_argument('-o', '--output', help="output path")
    parser.add_argument('--break-coeff', type=float, default=1.0, help="Coefficient for claim splitting")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--multilang', action='store_true')
    parser.add_argument('--limit-templates', type=int, default=None, help="Used only the first N templates")
    parser.add_argument('--align-constrains-type', help="Type of constraining during alignment")
    parser.add_argument('--skip', type=int, help="Skip N instances")

    args = parser.parse_args()
    return args


def main():
    # Arguments
    args = parse_args()
    if args.verbose:
        print(f"\nARGUMENTS:\n{args}\n")

    # Load data
    claims_data = load_jsonl(args.jsonl, verbose=True)
    if args.skip:
        claims_data = claims_data[args.skip:]

    # Load model
    model = LlamaLLM(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check that the output file does not exist
        if output_path.exists():
            cexception(f"Output file already exists: {output_path}")

    # Main loop
    with tqdm(total=len(claims_data)) as pbar:
        for data in claims_data:
            # Process
            processed_data = process(args, model, data)

            # Output
            if args.multilang:
                with output_path.open("a", encoding='utf-8') as fw:
                    fw.write(f"{json.dumps(processed_data, ensure_ascii=False)}\n")
            else:
                with output_path.open("a") as fw:
                    fw.write(f"{json.dumps(processed_data)}\n")

            pbar.update(1)


if __name__ == "__main__":
    main()
