#!/bin/bash
set -eu

python ./zeronatver.py \
	-j "test/data/scifact_withevidence_norm.jsonl" \
	-o "test/out.jsonl" \
	--align-constrains-type "post" \
	--claim-location "claim_preprocessed" \
	--evidence-location "evidence_preprocessed"
