
# Zero-NatVer

[https://arxiv.org/abs/2410.03341](https://arxiv.org/abs/2410.03341)

## Setup

Clone Zero-NatVer:
```bash
git clone https://github.com/marekstrong/Zero-NatVer
cd Zero-NatVer
```

Then, setup a new conda environment:

```bash
conda env create -f environment.yml
conda activate zeronatver
```

### LLama3
The current Zero-NatVer codebase is based on Llama3 (meta-llama/Meta-Llama-3-8B-Instruct). To download and test the model, run the following script:
```bash
python ./llm/test_llm.py
```
This script runs several unit tests to test that the core functionality works and that all query types are supported.

## Quick Start


You can run Zero-NatVer on a preprocessed version of *SciFact* using the following command:

```bash
python ./zeronatver.py \
  -j "test/data/scifact_withevidence_norm.jsonl" \
  -o "test/out.jsonl" \
  --align-constrains-type "post" \
  --claim-location "claim_preprocessed" \
  --evidence-location "evidence_preprocessed"
```


## Citation

If you find this work useful, please cite us:

```
@article{strong2024zero,
  title={Zero-Shot Fact Verification via Natural Logic and Large Language Models},
  author={Strong, Marek and Aly, Rami and Vlachos, Andreas},
  journal={arXiv preprint arXiv:2410.03341},
  year={2024}
}
```