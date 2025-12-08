from error_align.error_align import ErrorAlign
from error_align.utils import Alignment, basic_normalizer, basic_tokenizer


def error_align(
    ref: str,
    hyp: str,
    tokenizer: callable = basic_tokenizer,
    normalizer: callable = basic_normalizer,
    beam_size: int = 100,
    word_level_pass: bool = True,
) -> list[Alignment]:
    """Perform error alignment between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
        normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        beam_size (int): The beam size for beam search alignment. Defaults to 100.
        word_level_pass (bool): Use an initial word-level pass to identify unambiguous matches. Defaults to True.
            Note that this is not described in the original paper.

    Returns:
        list[tuple[str, str, OpType]]: A list of tuples containing aligned reference token,
                                        hypothesis token, and the operation type.

    """
    return ErrorAlign(
        ref,
        hyp,
        tokenizer=tokenizer,
        normalizer=normalizer,
        word_level_pass=word_level_pass,
    ).align(
        beam_size=beam_size,
    )
