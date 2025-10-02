import unicodedata

import regex as re
from num2words import num2words

from error_align.utils import NUMERIC_TOKEN, basic_normalizer, basic_tokenizer


def strip_accents(text: str) -> str:
    """Strip accents from the text."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def normalize_evaluation_segment(segment: str) -> str:
    """Normalize a segment by removing accents and converting to lowercase.

    Args:
        segment (str): The segment to normalize.

    Returns:
        str: The normalized segment.
    """
    return re.sub(r"[^a-z0-9]", "", strip_accents(segment.lower()))


def convert_numbers_to_words(text: str, lang: str = "en") -> str:
    """Convert numeric tokens in the text to their word representation.

    Args:
        text (str): The input text containing numeric tokens.
        lang (str): The language to use for conversion (default is "en").

    Returns:
        str: The text with numeric tokens converted to words.
    """
    if bool(re.match(NUMERIC_TOKEN, text)):
        if lang == "en":
            # NOTE: num2words doesn't support thousands separators
            text_ = text.replace(",", "")
        else:

            text_ = text.replace(".", "")
            text_ = text_.replace(",", ".")
        try:
            return num2words(text_, lang=lang)
        except Exception:
            pass

    return text


def clean_text(text: str, lang: str = "en") -> dict:
    """
    Cleans the text by removing examples with empty transcriptions.

    Args:
        example: The example to clean.

    Returns:
        A cleaned version of the example with empty transcriptions removed.
    """
    # Remove all tags, e.g., <unk>.
    text = re.sub(r"<[^>]+>", "", text)

    # Re-contract apostrophes.
    text = re.sub(r"(\w) '(\w)", r"\1'\2", text)

    # Get normalized tokens.
    normalized_tokens = [basic_normalizer(token.group()) for token in basic_tokenizer(text)]

    # Convert numbers to words.
    normalized_tokens = [convert_numbers_to_words(token, lang=lang) for token in normalized_tokens]

    return " ".join(normalized_tokens)


def clean_example(example: dict, lang="en") -> dict:
    """
    Cleans the example by removing examples with empty transcriptions.

    Args:
        example: The example to clean.
        lang: The language to use for cleaning.

    Returns:
        A cleaned version of the example with empty transcriptions removed.
    """
    if "ref" in example:
        example["ref"] = clean_text(example["ref"], lang=lang)
    if "hyp" in example:
        example["hyp"] = clean_text(example["hyp"], lang=lang)
    return example
