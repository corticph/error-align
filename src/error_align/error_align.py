from collections import Counter  # For measuring performance: Delete
from time import time  # For measuring performance: Delete
from typing import Union

from error_align.backtrace_graph import BacktraceGraph
from error_align.core import compute_levenshtein_distance_matrix, error_align_beam_search
from error_align.path_to_alignment import _get_alignments
from error_align.subgraph_metadata import SubgraphMetadata
from error_align.utils import (
    Alignment,
    OpType,
    basic_normalizer,
    basic_tokenizer,
    ensure_length_preservation,
    unpack_regex_match,
)

SubspanDescriptor = Union[SubgraphMetadata, range, tuple[int, int]]


lp = Counter()


class ErrorAlign:
    """Error alignment class that performs a two-pass alignment process."""

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
        word_level_pass: bool = True,
    ):
        """Initialize the error alignment with reference and hypothesis texts.

        The first pass (backtrace graph extraction) is performed during initialization.

        The second pass (beam search) is performed in the `align` method.

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

        self.ref_raw = ref
        self.hyp_raw = hyp
        self.word_level_pass = word_level_pass

        # Inclusive tokenization: Track the token position in the original text.
        tokenizer = unpack_regex_match(tokenizer)
        self._ref_token_matches = tokenizer(ref)
        self._hyp_token_matches = tokenizer(hyp)

        # Length-preserving normalization: Ensure that the normalizer preserves token length.
        normalizer = ensure_length_preservation(normalizer)
        self._ref_norm = [normalizer(r) for r, _ in self._ref_token_matches]
        self._hyp_norm = [normalizer(h) for h, _ in self._hyp_token_matches]
        self._identical_inputs = self._ref_norm == self._hyp_norm

        if self._identical_inputs:
            self._src = None
        elif word_level_pass:
            self._src = self._prepare_subspans_with_word_level_pass()
        else:
            self._src = SubgraphMetadata(
                ref_raw=self.ref_raw,
                hyp_raw=self.hyp_raw,
                ref_token_matches=self._ref_token_matches,
                hyp_token_matches=self._hyp_token_matches,
                ref_norm=self._ref_norm,
                hyp_norm=self._hyp_norm,
            )

    def align(self, beam_size: int = 100) -> list[Alignment]:
        """Perform beam search to align reference and hypothesis texts.

        Args:
            beam_size (int): The size of the beam for beam search. Defaults to 100.

        Returns:
            list[Alignment]: A list of Alignment objects.

        """
        if self._identical_inputs:
            return self._align_identical_inputs()
        elif self.word_level_pass:
            return self._align_post_word_level(self._src, beam_size=beam_size)
        else:
            return self._align_beam_search(self._src, beam_size=beam_size)

    def _align_identical_inputs(self) -> list[Alignment]:
        """Return alignments for identical reference and hypothesis pairs."""
        assert self._identical_inputs, "Inputs are not identical."

        alignments = []
        for i in range(len(self._ref_token_matches)):
            alignment = self._get_match_alignment_from_token_indices(i, i)
            alignments.append(alignment)
        return alignments

    def _align_beam_search(self, src: SubgraphMetadata, beam_size: int) -> list[Alignment]:
        """Perform beam search alignment for the given source."""
        start = time()
        path = error_align_beam_search(src=src, beam_size=beam_size)
        lp["beam_search_time"] += time() - start
        return _get_alignments(path)

    def _align_post_word_level(self, src: list[tuple[OpType, SubspanDescriptor]], beam_size: int) -> list[Alignment]:
        """Perform alignment after a word-level pass."""
        alignments = []
        for op_type, src_ in src:
            if op_type == OpType.MATCH:
                alignment = self._get_match_alignment_from_token_indices(*src_)
                alignments.append(alignment)
            elif op_type in (OpType.INSERT, OpType.DELETE):
                alignment_ = [self._get_insert_or_delete_alignment_from_token_index(op_type, i) for i in src_]
                alignments.extend(alignment_)
            else:
                alignments_ = self._align_beam_search(src=src_, beam_size=beam_size)
                alignments.extend(alignments_)

        return alignments

    def _get_match_alignment_from_token_indices(self, hyp_idx: int, ref_idx: int) -> Alignment:
        """Get a MATCH alignment for the given token indices."""
        ref_token_match = self._ref_token_matches[ref_idx]
        hyp_token_match = self._hyp_token_matches[hyp_idx]
        ref_slice = slice(*ref_token_match[1])
        hyp_slice = slice(*hyp_token_match[1])
        alignment = Alignment(
            op_type=OpType.MATCH,
            ref_slice=ref_slice,
            hyp_slice=hyp_slice,
            ref=self.ref_raw[ref_slice],
            hyp=self.hyp_raw[hyp_slice],
        )
        return alignment

    def _get_insert_or_delete_alignment_from_token_index(
        self,
        op_type: Union[OpType.INSERT, OpType.DELETE],
        token_idx: int,
    ) -> Alignment:
        """Get an INSERT or DELETE alignment for the given token index."""
        if op_type == OpType.INSERT:
            token_match = self._hyp_token_matches[token_idx]
            slice_ = slice(*token_match[1])
            token = self.hyp_raw[slice_]
            return Alignment(
                op_type=op_type,
                hyp_slice=slice_,
                hyp=token,
            )
        elif op_type == OpType.DELETE:
            token_match = self._ref_token_matches[token_idx]
            slice_ = slice(*token_match[1])
            token = self.ref_raw[slice_]
            return Alignment(
                op_type=op_type,
                ref_slice=slice_,
                ref=token,
            )
        else:
            raise ValueError(f"Invalid operation type for insert/delete alignment: {op_type}")

    def _prepare_subspans_with_word_level_pass(self) -> list[tuple[OpType, SubspanDescriptor]]:
        """Perform a word-level alignment pass to identify unambiguous matches."""

        # Extract the word-level backtrace graph.
        _, backtrace_matrix = compute_levenshtein_distance_matrix(self._ref_norm, self._hyp_norm, backtrace=True)
        backtrace_graph = BacktraceGraph(backtrace_matrix)
        match_indices = backtrace_graph.get_unambiguous_node_matches()
        match_indices = match_indices + [(len(self._hyp_norm), len(self._ref_norm))]

        # Iterate over the unambiguous matches to extract subspans (i.e., the span of words between two matches).
        hyp_start, ref_start = (0, 0)
        subspans = []
        end_index = len(match_indices) - 1
        for i, (hyp_end, ref_end) in enumerate(match_indices):
            ref_is_empty = ref_start == ref_end
            hyp_is_empty = hyp_start == hyp_end

            # NOTE: Subspans where ref xor hyp is empty are guaranteed to be all INSERT or DELETE ops.
            if ref_is_empty and hyp_is_empty:
                pass
            elif not ref_is_empty and not hyp_is_empty:
                src = self._get_subgraph_metadata(ref_start, ref_end, hyp_start, hyp_end)
                subspans.append((OpType.SUBSTITUTE, src))
            elif ref_is_empty:
                subspans.append((OpType.INSERT, range(hyp_start, hyp_end)))
            elif hyp_is_empty:
                subspans.append((OpType.DELETE, range(ref_start, ref_end)))

            if i < end_index:
                subspans.append((OpType.MATCH, (hyp_end, ref_end)))
            ref_start, hyp_start = (ref_end + 1, hyp_end + 1)

        return subspans

    def _get_subgraph_metadata(self, ref_start, ref_end, hyp_start, hyp_end) -> SubgraphMetadata:
        """Extract subgraph metadata for the given reference and hypothesis slices."""
        return SubgraphMetadata(
            ref_raw=self.ref_raw,
            hyp_raw=self.hyp_raw,
            ref_token_matches=self._ref_token_matches[ref_start:ref_end],
            hyp_token_matches=self._hyp_token_matches[hyp_start:hyp_end],
            ref_norm=self._ref_norm[ref_start:ref_end],
            hyp_norm=self._hyp_norm[hyp_start:hyp_end],
        )

    def __repr__(self):
        ref_preview = self.ref_raw if len(self.ref_raw) < 20 else self.ref_raw[:17] + "..."
        hyp_preview = self.hyp_raw if len(self.hyp_raw) < 20 else self.hyp_raw[:17] + "..."
        return f'ErrorAlign(ref="{ref_preview}", hyp="{hyp_preview}")'
