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
            return f'Alignment({self.op_type.name}: {self.hyp_with_compound_markers})'
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


def get_manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points a and b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
