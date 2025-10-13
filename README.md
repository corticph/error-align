<p align="center">
  <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/logo.svg" alt="ErrorAlign Logo" width="85%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%203.11%20|%203.12%20|%203.13-green" alt="Python Versions">
  <img src="https://img.shields.io/codecov/c/github/corticph/error-align/main.svg?style=flat-square" alt="Coverage" style="margin-left:5px;">
  <img src="https://github.com/corticph/error-align/actions/workflows/lint.yml/badge.svg?branch=main" alt="Linting" style="margin-left:5px;">
  <img src="https://img.shields.io/pypi/v/error-align.svg" alt="PyPI" style="margin-left:5px;">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" style="margin-left:5px;">
</p>

<br/>

**Text-to-text alignment algorithm for speech recognition error analysis.** ErrorAlign helps you dig deeper into your speech recognition projects by accurately aligning each word in a reference transcript with the model-generated transcript. Unlike traditional methods, such as Levenshtein-based alignment, it is not restricted to simple one-to-one alignment, but can map a single reference word to multiple words or subwords in the model output. This enables quick and reliable identification of error patterns in rare words, names, or domain-specific terms that matter most for your application.


__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) | [Work-in-Progress](#wip) | [Citation and Research](#citation) |




<a name="installation">

## Installation

```
pip install error-align
```


<a name="quickstart">

## Quickstart
```python
from error_align import error_align

ref = "Some things are worth noting!"
hyp = "Something worth nothing period?"

alignments = error_align(ref, hyp)
```

Resulting alignments:
```python
Alignment(SUBSTITUTE: "Some"- -> "Some"),
Alignment(SUBSTITUTE: -"thing" -> "things"),
Alignment(DELETE: "are"),
Alignment(MATCH: "worth" == "worth"),
Alignment(SUBSTITUTE: "noting" -> "nothing"),
Alignment(INSERT: "period")
```

<a name="wip">

## Work-in-Progress

- Optimization for longform text.
- Efficient word-level first-pass.
- C++ version with Python bindings.



<a name="citation">

## Citation and Research

```
@article{corticph2021alignment,
  title={A Text-To-Text Alignment Algorithm for Better Evaluation of Modern Speech Recognition Systems},
  author={corticph, Lasse and Havtorn, Jakob and Igel, Christian and Maal{\o}e, Lars and Tan, Zheng-Hua},
  journal={arXiv preprint arXiv:2509.24478},
  year={2025}
}
```

__To reproduce results from the paper:__
- Install with extra evaluation dependencies - only supported with Python 3.12:
  - `pip install error-align[evaluation]`
- Clone this repository:
  - `git clone https://github.com/corticph/error-align.git`
- Navigate to the evaluation directory:
  - `cd error-align/evaluation`
- Transcribe a dataset for evaluation. For example:
  - `python transcribe_dataset.py --model_name whisper --dataset_name commonvoice --language_code fr`
- Run evaluation script on the output file. For example:
  - `python evaluate_dataset.py --transcript_file transcribed_data/whisper_commonvoice_test_fr.parquet`

__Notes:__
- To reproduce results on the `primock57` dataset, first run: `python prepare_primock57.py`.
- Use the `--help` flag to see all available options for `transcribe_dataset.py` and `evaluate_dataset.py`.
- All results reported in the paper are based on the test sets.

__Collaborators:__

<br/>

<div>
  <a href="https://corti.ai">
    <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/corti.png" alt="Corti" height="75">
  </a>
  <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/empty.png" alt="" width="30">
  <a href="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/aau.png">
    <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/aau.png" alt="Aalborg University" height="75">
  </a>
  <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/empty.png" alt="" width="30">
  <a href="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/pcai.png">
    <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/pcai.png" alt="Pioneer Centre for Artificial Intelligence" height="75">
  </a>
  <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/empty.png" alt="" width="30">
  <a href="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/dtu.png">
    <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/dtu.png" alt="Technical University of Denmark" height="75">
  </a>
  <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/empty.png" alt="" width="30">
  <a href="https://www.ku.dk/">
    <img src="https://raw.githubusercontent.com/corticph/error-align/refs/heads/main/.github/assets/ucph.png" alt="University of Copenhagen" height="75">
  </a>
</div>

---
