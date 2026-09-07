#!/usr/bin/env python3
"""Rebuild data/gpqa_diamond.json from the gated GPQA release (Rein et al., 2023).

GPQA's license asks users not to publish the questions, so the file the paper used is not
in this repository. This script downloads gpqa_diamond.csv from the Hugging Face dataset
Idavidrein/gpqa (log in first with an account that has accepted its terms), builds the 198
records exactly as the paper's runs saw them, and checks the result against the checksum of
the original file.

Record format (CSV row order, i = 0..197):
    prompt  = question + "\\n\\nChoices:\\n(A) c0\\n(B) c1\\n(C) c2\\n(D) c3\\n", where
              [c0..c3] is [correct, incorrect 1, incorrect 2, incorrect 3] shuffled with
              random.Random(i).shuffle (the fields are used verbatim, whitespace included)
    answer  = letter of the correct choice
    id      = "gpqa_diamond_{i}", plus the record id, domain fields, raw question and
              correct-answer text from the CSV

    huggingface-cli login
    python data/build_gpqa_diamond.py                       # download, build, verify
    python data/build_gpqa_diamond.py --csv gpqa_diamond.csv  # from a local copy
"""
import argparse
import csv
import hashlib
import json
import os
import random

REPO = "Idavidrein/gpqa"
FILENAME = "gpqa_diamond.csv"
REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"          # dataset commit the paper used
EXPECTED_SHA256 = "b18c9dad2621abd9fa5eb514346c6ddf8bc2e8ad3fab7bbcd765da36e75b1cf5"
CHOICE_FIELDS = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpqa_diamond.json")


def build_record(i, row):
    order = [0, 1, 2, 3]
    random.Random(i).shuffle(order)
    choices = [row[CHOICE_FIELDS[j]] for j in order]
    prompt = (row["Question"] + "\n\nChoices:\n(A) " + choices[0] + "\n(B) " + choices[1]
              + "\n(C) " + choices[2] + "\n(D) " + choices[3] + "\n")
    return {
        "prompt": prompt,
        "answer": "ABCD"[order.index(0)],
        "source": "Idavidrein/gpqa (diamond)",
        "id": f"gpqa_diamond_{i}",
        "record_id": row["Record ID"],
        "high_level_domain": row["High-level domain"],
        "subdomain": row["Subdomain"],
        "writer_subjective_question_complexity": "",     # empty in the paper's file; kept for the schema
        "raw_question": row["Question"],
        "correct_answer_text": row["Correct Answer"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="local gpqa_diamond.csv (skips the download)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    path = args.csv
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    records = [build_record(i, row) for i, row in enumerate(rows)]
    data = json.dumps(records, indent=2, ensure_ascii=True).encode()
    with open(args.out, "wb") as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    status = "matches the paper's file" if digest == EXPECTED_SHA256 else "DOES NOT match the paper's file"
    print(f"wrote {args.out}: {len(records)} records, sha256 {digest[:16]}... ({status})")
    if digest != EXPECTED_SHA256:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
