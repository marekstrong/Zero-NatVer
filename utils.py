
import json
import re
from pathlib import Path


def load_json(fpath):
    fpath = Path(fpath)
    with fpath.open("r") as f:
        return json.load(f)


def load_jsonl(fpath, verbose=False):
    fpath = Path(fpath)

    data = []
    with fpath.open("r") as f:
        for line in f:
            line_d = json.loads(line)
            data.append(line_d)

    if verbose:
        print(f"Instances loaded: {len(data)}")

    return data


def get_only_alnumspace(text, remove_spaces=False):
    if remove_spaces:
        return ''.join([c for c in text if c.isalnum()])
    else:
        return ''.join([c for c in text if c.isalnum() or c == " "])


def get_alnum_string(input_string):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string).lower()
    return cleaned_string


def remove_substrings_absent_from_original(original, excerpts):
    excerpts_splits = excerpts.split(" ")

    clean_excerpts = []
    last_found_idx = 0
    last_found = None
    for n in range(len(excerpts_splits)):
        consider = " ".join(excerpts_splits[last_found_idx:n+1])
        if get_alnum_string(consider) in get_alnum_string(original):
            last_found = consider
        else:
            if last_found is not None:
                clean_excerpts.append(last_found)

            last_found_idx = n
            last_found = None

    if last_found is not None:
        clean_excerpts.append(last_found)

    return " ".join(clean_excerpts)
