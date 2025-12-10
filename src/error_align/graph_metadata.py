from dataclasses import dataclass, field
from functools import lru_cache

from error_align.backtrace_graph import BacktraceGraph
from error_align.core import compute_error_align_distance_matrix
from error_align.utils import categorize_char

TokenWithSpan = tuple[str, tuple[int, int]]  # (token_str, (start_idx, end_idx))


@dataclass
class GraphMetadata:
    ref_raw: str
    hyp_raw: str
    ref_token_matches: list[TokenWithSpan]
    hyp_token_matches: list[TokenWithSpan]
    ref_norm: list[str]
    hyp_norm: list[str]


@dataclass
class SubgraphMetadata:
    """Data class to hold information needed for beam search alignment.

    This data class encapsulates all necessary information about a subgraph
    derived from the reference and hypothesis texts, including their tokenized
    and normalized forms, as well as derived attributes used during
    the alignment process.

    It works as a reference for the `Path` class during beam search alignment.

    Attributes:
        ref_raw (str): The full raw reference text.
        hyp_raw (str): The full raw hypothesis text.
        ref_token_matches (list[tuple[str, tuple[int, int]]]): List of (token, (start, end)) tuples.
        hyp_token_matches (list[tuple[str, tuple[int, int]]]): List of (token, (start, end)) tuples.
        ref_norm (list[str]): List of normalized reference tokens.
        hyp_norm (list[str]): List of normalized hypothesis tokens.
        ref (str): The embedded reference text with delimiters.
        hyp (str): The embedded hypothesis text with delimiters.
        ref_max_idx (int): The maximum index in the reference text.
        hyp_max_idx (int): The maximum index in the hypothesis text.
        ref_char_types (list[int]): List of character types for the reference text.
        hyp_char_types (list[int]): List of character types for the hypothesis text.
        ref_idx_map (list[int]): Index map for the reference text.
        hyp_idx_map (list[int]): Index map for the hypothesis text.
        backtrace_graph (BacktraceGraph): The backtrace graph for the subgraph.
        backtrace_node_set (set[tuple[int, int]]): Set of nodes in the backtrace graph.
        unambiguous_matches (set[tuple[int, int]]): Set of end node indices for unambiguous token span matches.
    """

    # Init arguments.
    ref_raw: str
    hyp_raw: str
    ref_token_matches: list[tuple[str, tuple[int, int]]]
    hyp_token_matches: list[tuple[str, tuple[int, int]]]
    ref_norm: list[str]
    hyp_norm: list[str]

    # NOTE: The *_raw variables corresponds to the full input, even if only a subgraph is being aligned.
    # The *_token_matches are computed on the full input so their indices correspond to the full input as well,
    # even if only a subset of the tokens is being aligned.

    # Derived attributes.
    ref: str = field(init=False)
    hyp: str = field(init=False)
    ref_max_idx: int = field(init=False)
    hyp_max_idx: int = field(init=False)
    ref_char_types: list[int] = field(init=False)
    hyp_char_types: list[int] = field(init=False)
    ref_idx_map: list[int] = field(init=False)
    hyp_idx_map: list[int] = field(init=False)
    backtrace_node_set: set[tuple[int, int]] = field(init=False)
    unambiguous_matches: set[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        # Process reference and hypothesis texts and compute derived attributes.
        self.ref = _embed_tokens(self.ref_norm)
        self.hyp = _embed_tokens(self.hyp_norm)
        self.ref_max_idx = len(self.ref) - 1
        self.hyp_max_idx = len(self.hyp) - 1
        self.ref_char_types = _get_char_types(self.ref)
        self.hyp_char_types = _get_char_types(self.hyp)
        self.ref_idx_map = _create_index_map(self.ref_token_matches)
        self.hyp_idx_map = _create_index_map(self.hyp_token_matches)

        # First pass: Compute backtrace graph.
        _, backtrace_matrix = compute_error_align_distance_matrix(self.ref, self.hyp, backtrace=True)
        backtrace_graph = BacktraceGraph(backtrace_matrix)
        # NOTE: Used for backtrace deviation penalty during beam search.
        self.backtrace_node_set = backtrace_graph.get_node_set()
        # NOTE: Used for beam pruning during beam search.
        self.unambiguous_matches = backtrace_graph.get_unambiguous_token_span_matches(self.ref)


def _embed_tokens(text_tokens: list[str]) -> str:
    """Embed tokens with delimiters."""
    return "".join([f"<{t}>" for t in text_tokens])


@lru_cache(maxsize=None)
def _categorize_char_cached(c: str) -> int:
    """Cached version of categorize_char for performance."""
    return categorize_char(c)


def _get_char_types(text: str) -> list[int]:
    """Get character types (0-3) for each character in the text."""
    return [_categorize_char_cached(c) for c in text]


def _create_index_map(text_tokens: list[TokenWithSpan]) -> list[int]:
    """Create an index map for the given tokens.

    The 'index_map' is used to map each aligned character back to its original position in the input text.

    NOTE: -1 is used for delimiter (<>) and indicates no match in the source sequence.
    """
    index_map = []
    for _, span in text_tokens:
        index_map.append(-1)  # Start delimiter
        index_map.extend(range(*span))
        index_map.append(-1)  # End delimiter
    return index_map
