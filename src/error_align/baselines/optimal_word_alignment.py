from rapidfuzz.distance import Levenshtein

from error_align.backtrace_graph import BacktraceGraph
from error_align.edit_distance import compute_distance_matrix
from error_align.utils import (
    Alignment,
    OpType,
    basic_normalizer,
    basic_tokenizer,
)


def _get_optimal_word_alignment_values(ref_token: str, hyp_token: str):
    """Compute the optimal word alignment values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.
    """
    if hyp_token == ref_token:
        diag_cost = 0
    else:
        diag_cost = Levenshtein.distance(ref_token, hyp_token, weights=(1, 1, 2))
        diag_cost += abs(len(ref_token) - len(hyp_token))

    return len(hyp_token), len(ref_token), diag_cost

class OptimalWordAlign:
    """Optimal word-level alignment based on global-to-local edits (GLE) metric."""

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
    ):
        """Initialize the optimal word-level alignment with reference and hypothesis texts.

        Args:
            ref (str): The reference sequence/transcript.
            hyp (str): The hypothesis sequence/transcript.
            tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
            normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        """
        if not isinstance(ref, str):
            raise TypeError("Reference sequence must be a string.")
        if not isinstance(hyp, str):
            raise TypeError("Hypothesis sequence must be a string.")

        self.ref = ref
        self.hyp = hyp
        self._ref_token_matches = tokenizer(ref)
        self._hyp_token_matches = tokenizer(hyp)
        self._ref = [normalizer(r.group()) for r in self._ref_token_matches]
        self._hyp = [normalizer(h.group()) for h in self._hyp_token_matches]
        self._ref_max_idx = len(self._ref) - 1
        self._hyp_max_idx = len(self._hyp) - 1
        self.end_index = (self._hyp_max_idx, self._ref_max_idx)

        # Extract backtrace graph.
        _, B = compute_distance_matrix(
            self._ref,
            self._hyp,
            _get_optimal_word_alignment_values,
            backtrace=True,
            dtype=float,
        )
        self._backtrace_graph = BacktraceGraph(B)

    def align(self) -> list[Alignment]:
        """Extract an arbitrary path from the backtrace graph -- all paths are equal wrt. to the GLE metric.

        Returns:
            list[Alignment]: A list of Alignment objects.
        """
        path = self._backtrace_graph.get_path()
        alignments = []
        for op_type, node in path:
            if op_type == OpType.MATCH or op_type == OpType.SUBSTITUTE:
                ref_match = self._ref_token_matches[node.ref_idx - 1]
                hyp_match = self._hyp_token_matches[node.hyp_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    ref_slice=slice(*ref_match.span()),
                    hyp_slice=slice(*hyp_match.span()),
                    ref=ref_match.group(),
                    hyp=hyp_match.group(),
                )
            elif op_type == OpType.DELETE:
                ref_match = self._ref_token_matches[node.ref_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    ref_slice=slice(*ref_match.span()),
                    ref=ref_match.group(),
                )
            elif op_type == OpType.INSERT:
                hyp_match = self._hyp_token_matches[node.hyp_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    hyp_slice=slice(*hyp_match.span()),
                    hyp=hyp_match.group(),
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            alignments.append(alignment)

        return alignments
