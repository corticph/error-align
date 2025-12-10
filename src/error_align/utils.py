from dataclasses import dataclass
from enum import IntEnum
from itertools import chain, combinations

import regex as re
from unidecode import unidecode


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3


@dataclass
class Alignment:
    """Class representing an operation with its type and cost."""

    op_type: OpType
    ref_slice: slice | None = None
    hyp_slice: slice | None = None
    ref: str | None = None
    hyp: str | None = None
    left_compound: bool = False
    right_compound: bool = False

    def __post_init__(self):
        if self.op_type == OpType.MATCH:
            if self.ref is None or self.hyp is None:
                raise ValueError("MATCH operation must have non-empty ref or hyp.")
            if self.left_compound or self.right_compound:
                raise ValueError("MATCH operation cannot have compound markers.")
        elif self.op_type == OpType.INSERT:
            if self.hyp is None or self.ref is not None:
                raise ValueError("INSERT operation must have non-empty hyp and empty ref.")
        elif self.op_type == OpType.DELETE:
            if self.hyp is not None or self.ref is None:
                raise ValueError("DELETE operation must have non-empty ref and empty hyp.")
        elif self.op_type == OpType.SUBSTITUTE:
            if self.ref is None or self.hyp is None:
                raise ValueError("SUBSTITUTE operation must have both ref and hyp.")

    @property
    def hyp_with_compound_markers(self) -> str:
        """Return the hypothesis with compound markers if applicable."""
        if self.hyp is None:
            return None
        return f'{"-" if self.left_compound else ""}"{self.hyp}"{"-" if self.right_compound else ""}'

    def __repr__(self) -> str:
        if self.op_type == OpType.DELETE:
            return f'Alignment({self.op_type.name}: "{self.ref}")'
        if self.op_type == OpType.INSERT:
            return f"Alignment({self.op_type.name}: {self.hyp_with_compound_markers})"
        if self.op_type == OpType.SUBSTITUTE:
            return f'Alignment({self.op_type.name}: {self.hyp_with_compound_markers} -> "{self.ref}")'
        return f'Alignment({self.op_type.name}: "{self.hyp}" == "{self.ref}")'


def op_type_powerset() -> chain:
    """Generate all possible combinations of operation types, except the empty set.

    Returns:
        Generator: All possible combinations of operation types.

    """
    op_types = list(OpType)
    op_combinations = [combinations(op_types, r) for r in range(1, len(op_types) + 1)]
    return chain.from_iterable(op_combinations)


START_DELIMITER = "<"
END_DELIMITER = ">"
DELIMITERS = {START_DELIMITER, END_DELIMITER}

OP_TYPE_MAP = {op_type.value: op_type for op_type in OpType}
OP_TYPE_COMBO_MAP = {i: op_types for i, op_types in enumerate(op_type_powerset())}
OP_TYPE_COMBO_MAP_INV = {v: k for k, v in OP_TYPE_COMBO_MAP.items()}

NUMERIC_TOKEN = r"\p{N}+([,.]\p{N}+)*(?=\s|$)"
STANDARD_TOKEN = r"[\p{L}\p{N}]+(['][\p{L}\p{N}]+)*'?"


def is_vowel(c: str) -> bool:
    """Check if the normalized character is a vowel.

    Args:
        c (str): The character to check.

    Returns:
        bool: True if the character is a vowel, False otherwise.

    """
    assert len(c) == 1, "Input must be a single character."
    return unidecode(c)[0] in "aeiouy"


def is_consonant(c: str) -> bool:
    """Check if the normalized character is a consonant.

    Args:
        c (str): The character to check.

    Returns:
        bool: True if the character is a consonant, False otherwise.

    """
    assert len(c) == 1, "Input must be a single character."
    return unidecode(c)[0] in "bcdfghjklmnpqrstvwxyz"


def categorize_char(c: str) -> int:
    """Categorize a character as 'vowel', 'consonant', or 'unvoiced'.

    Args:
        c (str): The character to categorize.

    Returns:
        str: The category of the character.

    """
    if c in DELIMITERS:
        return 0
    if is_consonant(c):
        return 1
    if is_vowel(c):
        return 2
    return 3  # NOTE: Unvoiced characters (only apostrophes are expected by default).


def basic_tokenizer(text: str) -> list:
    """Default tokenizer that splits text into words based on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokens (words).

    """
    return list(re.finditer(rf"({NUMERIC_TOKEN})|({STANDARD_TOKEN})", text, re.UNICODE | re.VERBOSE))


def basic_normalizer(text: str) -> str:
    """Default normalizer that only converts text to lowercase.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.

    """
    return text.lower()


def ensure_length_preservation(normalizer: callable) -> callable:
    """Decorator to ensure that the normalizer preserves the length of the input text.

    Args:
        normalizer (callable): The normalizer function to wrap.

    Returns:
        callable: The wrapped normalizer function that preserves length.

    """

    def wrapper(text: str, *args: list, **kwargs: dict) -> str:
        normalized = normalizer(text, *args, **kwargs)
        if len(normalized) != len(text):
            raise ValueError("Normalizer must preserve length.")
        return normalized

    return wrapper


def unpack_regex_match(tokenizer: callable) -> callable:
    """Unpack a regex Match object to extract the matched string.

    Args:
        tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.

    Returns:
        callable: A function that unpacks a list of Match objects into tuples of (matched string, span).

    """

    def wrapper(text: str, *args: list, **kwargs: dict) -> list[tuple[str, tuple[int, int]]]:
        matches = tokenizer(text, *args, **kwargs)
        return [(match.group(), match.span()) for match in matches]

    return wrapper


def translate_slice(segment_slice: slice, index_map: list[int]) -> None | slice:
    """Translate a slice from the alignment sequence back to the original sequence.

    Args:
        segment_slice (slice): The slice in the alignment sequence.
        index_map (list[int]): The mapping from alignment indices to original sequence indices.

    Returns:
        None | slice: The translated slice in the original sequence, or None if no valid indices.

    """
    slice_indices = index_map[segment_slice]
    slice_indices = list(filter(lambda x: x >= 0, slice_indices))
    if len(slice_indices) == 0:
        return None
    start, end = int(slice_indices[0]), int(slice_indices[-1] + 1)
    return slice(start, end)
