from error_align.error_align import ErrorAlign, Path
from error_align.utils import Alignment, basic_normalizer, basic_tokenizer


def error_align(
    ref: str,
    hyp: str,
    tokenizer: callable = basic_tokenizer,
    normalizer: callable = basic_normalizer,
    beam_size: int = 100,
    pbar: bool = False,
    return_path: bool = False,
) -> list[Alignment] | Path:
    """Perform error alignment between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
        normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        pbar (bool): Whether to display a progress bar. Defaults to False.
        return_path (bool): Whether to return the path object or just the alignments. Defaults to False.

    Returns:
        list[tuple[str, str, OpType]]: A list of tuples containing aligned reference token,
                                        hypothesis token, and the operation type.

    """
    return ErrorAlign(
        ref,
        hyp,
        tokenizer=tokenizer,
        normalizer=normalizer,
    ).align(
        beam_size=beam_size,
        pbar=pbar,
        return_path=return_path,
    )
