import json
import re
from collections import Counter
from decimal import Decimal
from functools import partial
from pathlib import Path
from time import time

import click
import numpy as np
from datasets import Dataset
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from error_align import ErrorAlign
from error_align.baselines import OptimalWordAlign, PowerAlign, RapidFuzzWordAlign
from error_align.baselines.power.power.pronounce import PronouncerLex
from error_align.baselines.utils import clean_example, normalize_evaluation_segment
from error_align.utils import Alignment, OpType


def paired_approximate_permutation_test(
    metric_values_a: list[int],
    metric_values_b: list[int],
    num_rounds: int = 9999,
    seed: int = 42,
):
    """Perform a paired approximate permutation test to compare two sets of metric values.

    This test returns the significance level (p-value) of the null hypothesis that the two methods are equivalent.
    We test whether method A is significantly better (i.e., has a smaller metric value) than method B (one-sided test).

    Args:
        metric_values_a (list[float]): Metric values for method A.
        metric_values_b (list[float]): Metric values for method B.
        num_rounds (int): Number of permutation rounds.
        seed (int): Random seed for reproducibility.

    Returns:
        float: p-value indicating the significance of the difference between the two methods.

    """
    import numpy as np

    assert len(metric_values_a) == len(metric_values_b), "Metric value lists must have the same length."
    rng = np.random.default_rng(seed)

    # We test whether A is better than B (i.e., positive and large observed_diffs)
    observed_diffs = metric_values_b - metric_values_a
    total_observed_diff = np.sum(observed_diffs)
    count = 0

    for _ in range(num_rounds):
        signs = rng.choice([-1, 1], size=observed_diffs.shape)
        permuted_diff = np.sum(observed_diffs * signs)
        # If observed_diffs is indeed positive and large, the case permuted_diff >= total_observed_diff is rare
        if permuted_diff >= total_observed_diff:
            count += 1

    # If permuted_diff >= total_observed_diff is rare, p_value is small
    p_value = (count + 1) / (num_rounds + 1)
    return p_value


def get_phoneme_base_edits(ref: str, hyp: str, phoneme_converter: callable):
    """Compute the number of phoneme edits between reference and hypothesis.

    Args:
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.
        phoneme_converter (callable): A function to convert text to phonemes.

    Returns:
        int: The number of phoneme edits.

    """
    ref_phonemes = list(filter(lambda p: p != "|", phoneme_converter(normalize_evaluation_segment(ref))))
    hyp_phonemes = list(filter(lambda p: p != "|", phoneme_converter(normalize_evaluation_segment(hyp))))
    return Levenshtein.distance(
        ref_phonemes,
        hyp_phonemes,
        weights=(1, 1, 2),
    )


def replace_repeated_patterns(hyp, min_repeats=10):
    """Replace repeated patterns in the hypothesis to counter severe hallucinations.

    Args:
        hyp (str): The hypothesis transcription.
        min_repeats (int): Minimum number of repeats to consider for replacement.

    """
    pattern = re.compile(r"(.+?)\1{" + str(min_repeats - 1) + r",}")
    for m in reversed(list(pattern.finditer(hyp))):
        x, y = m.span()
        hyp = hyp[:x] + m.groups()[0] + hyp[y:]
    return hyp


def get_error_alignments(ref: str, hyp: str, beam_size: int):
    """Get error alignments using ErrorAlign with beam search.

    Args:
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.
        beam_size (int): The beam width for beam search.

    Returns:
        List[Alignment]: A list of alignment objects.

    """
    return ErrorAlign(ref=ref, hyp=hyp).align(beam_size=beam_size, pbar=False)


def get_optimal_word_alignments(ref: str, hyp: str):
    """Get optimal word alignments using OptimalWordAlign.

    Args:
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.

    Returns:
        List[Alignment]: A list of alignment objects.

    """
    return OptimalWordAlign(ref=ref, hyp=hyp).align()


def get_rapidfuzz_word_alignments(ref: str, hyp: str):
    """Get word alignments using RapidFuzzWordAlign.

    Args:
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.

    Returns:
        List[Alignment]: A list of alignment objects.

    """
    return RapidFuzzWordAlign(ref=ref, hyp=hyp).align()


def get_power_alignments(ref: str, hyp: str):
    """Get word alignments using PowerAlign.

    Args:
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.

    Returns:
        List[Alignment]: A list of alignment objects.

    """
    return PowerAlign(ref=ref, hyp=hyp).align()


def compute_edits(alignments: list[Alignment], phoneme_converter: callable = None):
    """Compute the number of edits in a list of alignments.

    Args:
        alignments (list[Alignment]): The list of alignments to evaluate.

    Returns:
        int: The total number of edits.

    """
    if phoneme_converter is None:
        normalize = normalize_evaluation_segment
    else:

        def normalize(text):
            phonemes = phoneme_converter(normalize_evaluation_segment(text))
            # return phonemes
            return list(filter(lambda p: p != "|", phonemes))

    num_edits = 0
    for a in alignments:
        if a.op_type == OpType.MATCH:
            continue
        if a.op_type == OpType.DELETE:
            num_edits += len(normalize(a.ref))
        elif a.op_type == OpType.INSERT:
            num_edits += len(normalize(a.hyp))
        elif a.op_type == OpType.SUBSTITUTE:
            nref = normalize(a.ref)
            nhyp = normalize(a.hyp)
            num_edits += Levenshtein.distance(
                nref,
                nhyp,
                weights=(1, 1, 2),
            )
            num_edits += abs(len(nref) - len(nhyp))
    return num_edits


def evaluate_method(method: callable, ref: str, hyp: str, phoneme_converter: callable = None):
    """Evaluate a given alignment method.

    Args:
        method (callable): The alignment method to evaluate.
        ref (str): The reference transcription.
        hyp (str): The hypothesis transcription.

    Returns:
        Tuple[float, int]: A tuple containing the time taken and number of edits.

    """
    start_time = time()
    alignments = method(ref=ref, hyp=hyp)
    time_taken = time() - start_time
    char_edits = compute_edits(alignments)
    phoneme_edits = compute_edits(alignments, phoneme_converter=phoneme_converter)
    return alignments, time_taken, char_edits, phoneme_edits


@click.command()
@click.option(
    "--transcript_file",
    type=str,
    help="Path to the transcript file.",
    required=True,
)
@click.option(
    "--only_error_align",
    help="Whether to only output error alignments.",
    is_flag=True,
)
@click.option(
    "--beam_size",
    type=int,
    help="Beam width for beam search.",
    default=100,
)
@click.option(
    "--save_results",
    help="Path to save the results.",
    is_flag=True,
)
def main(transcript_file: str, only_error_align: bool, beam_size: int, save_results: bool):
    """Main function to evaluate the alignment algorithms."""
    # Validate input file and load dataset.
    transcript_file = Path(transcript_file)
    assert transcript_file.exists(), f"Transcript file {transcript_file} does not exist."
    assert transcript_file.suffix == ".parquet", f"Transcript file {transcript_file} is not a parquet file."
    dataset = Dataset.from_parquet(transcript_file.as_posix())

    # Parse transcript file of format: .../model_dataset_subset_language.parquet
    model_name, dataset_name, subset_name, language_code = transcript_file.stem.split("_")

    # Get cleaning functions and prepare dataset.
    global clean_example
    clean_example = partial(clean_example, lang=language_code)
    dataset = dataset.map(clean_example)

    # Collect candidate methods.
    methods = {
        "Power": get_power_alignments,
        "Error": partial(get_error_alignments, beam_size=beam_size),
        "Optimal": get_optimal_word_alignments,
        "RapidFuzz": get_rapidfuzz_word_alignments,
    }
    if only_error_align:
        del methods["Power"]
        del methods["Optimal"]
        del methods["RapidFuzz"]
    elif language_code != "en":
        del methods["Power"]

    # Run evaluation.
    metrics = {k: Counter() for k in methods}
    char_edits_all = {k: [] for k in methods}
    phoneme_converter = (
        PronouncerLex("/home/lb/repos/power-asr/lex/cmudict.rep.json").pronounce if language_code == "en" else None
    )

    # dataset = dataset.select(range(792, 794))

    c_n, p_n = 0, 0
    for example in tqdm(dataset):
        ref, hyp = example["ref"], example["hyp"]
        hyp = replace_repeated_patterns(hyp)
        if ref.strip() == "" or hyp.strip() == "":
            continue

        try:
            for method_name, method in methods.items():
                _, duration, char_edits, phoneme_edits = evaluate_method(
                    method,
                    ref,
                    hyp,
                    phoneme_converter=phoneme_converter,
                )
                metrics[method_name]["duration"] += duration
                metrics[method_name]["character_edits"] += char_edits
                if phoneme_converter is not None:
                    metrics[method_name]["phoneme_edits"] += phoneme_edits
                char_edits_all[method_name].append(char_edits)
        except AssertionError as e:
            assert method_name == "Power", "Unexpected method failure: " + str(e)
            continue

        c_n += Levenshtein.distance(
            normalize_evaluation_segment(ref),
            normalize_evaluation_segment(hyp),
            weights=(1, 1, 2),
        )
        if phoneme_converter is not None:
            p_n += get_phoneme_base_edits(ref, hyp, phoneme_converter)

    # Print resume.
    print(f"Evaluation results for {model_name} on {dataset_name}/{subset_name} ({language_code}):")

    if phoneme_converter is not None:
        edit_types = ["character_edits", "phoneme_edits"]
    else:
        edit_types = ["character_edits"]

    for edits in edit_types:
        print(f"\nUsing {edits}:")
        for method_name, method_metrics in metrics.items():
            abs_edits = method_metrics[edits]
            norm_edits = c_n if edits == "character_edits" else p_n
            score = norm_edits / abs_edits if abs_edits > 0 else 1.0
            duration = method_metrics["duration"]
            print(f"{method_name}: score = {score:.4f} | edits = {abs_edits}/{norm_edits} | time = {duration:.1f}s")
            method_metrics["score"] = score

    # Convert edit lists to numpy arrays for statistical tests.
    for k, v in char_edits_all.items():
        char_edits_all[k] = np.array(v)

    # Run statistical tests
    print("\nStatistical significance tests (character edits):")
    for k, v in char_edits_all.items():
        if k == "Error":
            continue
        p_value = paired_approximate_permutation_test(char_edits_all["Error"], v)
        p_value_formatted = Decimal(p_value).quantize(Decimal("0.0001")).normalize()
        print(f"p-value of Error vs {k}: {p_value_formatted}")

    # Dump metrics as json.
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / f"{model_name}_{dataset_name}_{subset_name}_{language_code}.json", "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
