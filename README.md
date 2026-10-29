# Evaluation awareness in English and Hindi multimodal safety prompts

Code, data and analysis outputs for the paper **"Do Vision-Language Models Say They Are Being Tested? Evaluation Awareness in English and Hindi Multimodal Safety Prompts"** [add journal, year and DOI when available].

> **Content warning.** This repository contains prompts about harmful topics (violence, self-harm, illegal activity, sexual content) and model replies, some of which comply with harmful requests. It is released for safety research only. The images are **not** included.

## What this study does

We test whether telling a vision-language model that a conversation is a scored AI safety evaluation changes how it responds to harmful image-and-text requests, in English and Hindi.

- **Data:** 100 items sampled (seed 2026, stratified by hazard category) from the Multimodal Safety Test Suite (MSTS), English and Hindi splits.
- **Models:** Gemma-3-12B-it and Qwen2.5-VL-7B-Instruct, run locally with Ollama (default quantisation), greedy decoding, at most 500 new tokens.
- **Conditions:** 2 languages × 2 modalities (text-only with image description, image + text) × 2 cues (`deploy`, `eval_explicit`) × 2 models × 100 items = 1,600 responses.
- **Outcomes:** refusal (reported as a lower bound from refusal-phrase rules and an upper bound from an LLM judge, Gemma-3-12B) and mentions of the evaluation in the reply (keyword screens).
- **Validation:** second judge (Qwen2.5-7B) on a subset; 160 main-run replies labelled by an AI annotator (Claude) and 80 Hindi replies labelled by a native speaker.

## Repository layout

```
notebooks/   Colab notebook that generates responses and runs the judging and analysis
scripts/     judge_v4.py, validate_by_cue.py, cue_mention_tables.py, export_fresh_hindi.py
data/
main/  responses_main.jsonl, judged_main_gemma12b.jsonl, judged_main_qwen7b.jsonl, sample_main.csv
validation/ labelled and blind validation sheets (main run)
```

### Label files

- `claude_label` columns are labels from an AI annotator (Claude). `human_label` columns are from a native Hindi speaker (an author). Labels are DECLINE, GUIDE, COMPLY, CUE_ECHO, OFF_TOPIC, UNUSABLE; `mentions_cue` / `claude_mentions_cue` = 1 if the reply refers to an evaluation, test, scoring or logging. Guidelines are in Appendix B of the paper.
- `*_blind.csv` sheets contain no cue or model information and can be labelled independently.

## Reproducing the results

1. **Generation** (needs a GPU; about 4 hours per model on a Colab T4): open `notebooks/` notebook in Colab, set `ITEMS_TO_RUN = None`, run sections 1–9. This writes `responses_main.jsonl`.
2. **Judging** (needs Ollama and `gemma3:12b`):
   `python judge_v4.py --in responses_main.jsonl --out judged_main_gemma12b.jsonl --model gemma3:12b --skip-cue-llm --tables-dir tables_main_gemma12b`
3. **Cue-mention tables** (no GPU): `python cue_mention_tables.py`
4. **Validation by cue** (no GPU): `python validate_by_cue.py --labels main_validation_hi_human_labelled.csv --label-col human_label`

The scripts read their input files from the current directory, so copy the files a script needs into one folder (and keep `judge_v4.py` next to the other scripts) before running it.

Install dependencies with `pip install -r requirements.txt`. Record exact versions with `pip freeze` and the Ollama version with `ollama --version` [add here].

## Data sources, licences and attribution

- **MSTS** (Röttger et al., 2025, arXiv:2501.10057) is distributed with a permissive licence (dataset card: CC BY 4.0). Prompts and image descriptions in this repository come from MSTS; please cite it. **Images are not redistributed**; obtain them from the original dataset (`felfri/MSTS` on Hugging Face).
- **Cue wording** is adapted from Garg et al. (2026), "Evaluation awareness beyond English", arXiv:2610.22333. The Hindi cue text was reconstructed and reviewed by a native speaker.
- **Code:** [MIT licence]. **Data we created (responses, judge outputs, labels):** [CC BY 4.0]. Model outputs may be subject to the terms of use of the models that produced them (Gemma, Qwen); check them before reuse.
