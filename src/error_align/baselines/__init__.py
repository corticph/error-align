
try:
    from error_align.baselines.optimal_word_alignment import OptimalWordAlign as OptimalWordAlign
    from error_align.baselines.power_alignment import PowerAlign as PowerAlign
    from error_align.baselines.rapidfuzz_word_alignment import RapidFuzzWordAlign as RapidFuzzWordAlign
except ImportError:
    raise ImportError(
        "Baselines module could not be imported. Please ensure all dependencies are installed. "
        "You can install the evaluation dependencies using: pip install error-align[evaluation]"
    )
