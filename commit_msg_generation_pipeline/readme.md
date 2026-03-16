# Security-related Commit Message Generation from Diffs

A concise two-step pipeline for generating commit messages from Git diffs with a local LLM.

## Overview

This script converts each diff into a commit message in **two stages**:

1. **Diff → Structured summary**
   - Extracts a security-aware JSON summary from the unified diff
   - Captures intent, key changes, touched files, entities, evidence, security effect, and tests

2. **Summary → Commit message**
   - Generates a concise developer-style commit message from the summary
   - Adds a subject, bullets, risk line, and evidence refs

This design reduces hallucination by forcing the model to write from a structured intermediate summary instead of directly from raw diffs.

## Input

A CSV file with at least:

- `patch`: unified diff text

Optional:

- `clean_message`: ignored during generation

## Output

The script writes a CSV with two added columns:

- `summary`: JSON summary from step 1
- `gen_message`: generated commit message from step 2



## Requirements

Install:

```bash
pip install pandas torch transformers tqdm
