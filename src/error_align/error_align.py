from dataclasses import dataclass, field
from functools import lru_cache
from typing import Union

import regex as re

from error_align.backtrace_graph import BacktraceGraph
from error_align.edit_distance import compute_error_align_distance_matrix, compute_levenshtein_distance_matrix
from error_align.utils import (
    END_DELIMITER,
    START_DELIMITER,
    Alignment,
    OpType,
    basic_normalizer,
    basic_tokenizer,
    categorize_char,
    ensure_length_preservation,
)


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


def _create_index_map(text_tokens: list[re.Match]) -> list[int]:
    """Create an index map for the given tokens.

    The 'index_map' is used to map each aligned character back to its original position in the input text.

    NOTE: -1 is used for delimiter (<>) and indicates no match in the source sequence.
    """
    index_map = []
    for match in text_tokens:
        index_map.append(-1)  # Start delimiter
        index_map.extend(range(*match.span()))
        index_map.append(-1)  # End delimiter
    return index_map


@dataclass
class SubgraphMetadata:
    """Data class to hold information needed for beam search alignment.

    This data class encapsulates all necessary information about a subgraph
    derived from the reference and hypothesis texts, including their tokenized
    and normalized forms, as well as various derived attributes used during
    the alignment process.

    It works as a reference for the `Path` class during beam search alignment.

    Attributes:
        ref_raw (str): The full raw reference text.
        hyp_raw (str): The full raw hypothesis text.
        ref_token_matches (list[re.Match]): List of regex Match objects for reference tokens.
        hyp_token_matches (list[re.Match]): List of regex Match objects for hypothesis tokens.
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
    ref_token_matches: list[re.Match]
    hyp_token_matches: list[re.Match]
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
    backtrace_graph: BacktraceGraph = field(init=False)
    backtrace_node_set: set[tuple[int, int]] = field(init=False)
    unambiguous_matches: set[tuple[int, int]] = field(init=False)

    def __repr__(self):
        ref_preview = self.ref if len(self.ref) < 20 else self.ref[:17] + "..."
        hyp_preview = self.hyp if len(self.hyp) < 20 else self.hyp[:17] + "..."
        return f'SubgraphMetadata(ref="{ref_preview}", hyp="{hyp_preview}")'

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
        self.backtrace_graph = BacktraceGraph(backtrace_matrix)
        # NOTE: Used for backtrace deviation penalty during beam search.
        self.backtrace_node_set = self.backtrace_graph.get_node_set()
        # NOTE: Used for beam pruning during beam search.
        self.unambiguous_matches = self.backtrace_graph.get_unambiguous_token_span_matches(self.ref)


class ErrorAlign:
    """Error alignment class that performs a two-pass alignment process."""

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
        word_level_pass: bool = False,
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
        self._ref_token_matches = tokenizer(ref)
        self._hyp_token_matches = tokenizer(hyp)

        # Length-preserving normalization: Ensure that the normalizer preserves token length.
        normalizer = ensure_length_preservation(normalizer)
        self._ref_norm = [normalizer(r.group()) for r in self._ref_token_matches]
        self._hyp_norm = [normalizer(h.group()) for h in self._hyp_token_matches]
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

    def align(self, beam_size: int = 100) -> Union[list[Alignment], "Path"]:
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
            return self._beam_search_alignment(self._src, beam_size=beam_size)

    def _beam_search_alignment(
        self,
        src: SubgraphMetadata,
        beam_size: int = 100,
    ) -> Union[list[Alignment], "Path"]:
        """Perform beam search to align reference and hypothesis texts for a given source."""

        # Initialize the beam with a single path starting at the root node.
        start_path = Path(src)
        beam = {start_path.pid: start_path}
        prune_map = dict()
        ended = []

        # Expand candidate paths until all have reached the terminal node.
        while len(beam) > 0:
            new_beam = {}

            # Expand each path in the current beam.
            for path in beam.values():
                if path.at_end:
                    ended.append(path)
                    continue

                # Transition to all child nodes.
                for new_path in path.expand():
                    new_path_cost = new_path.cost
                    new_path_pid = new_path.pid
                    if new_path_pid in prune_map:
                        if new_path_cost > prune_map[new_path_pid]:
                            continue
                    prune_map[new_path_pid] = new_path_cost

                    if new_path_pid not in new_beam or new_path_cost < new_beam[new_path_pid].cost:
                        new_beam[new_path_pid] = new_path

            # Update the beam with the newly expanded paths.
            new_beam = list(new_beam.values())
            new_beam.sort(key=lambda p: p.norm_cost)
            beam = new_beam[:beam_size]

            # Keep only the best path if, it matches the segment.
            if len(beam) > 0 and beam[0]._at_unambiguous_match_node:
                beam = beam[:1]
                prune_map = dict()
            beam = {p.pid: p for p in beam}  # Convert to dict for diversity check.

        # Return the best path or its alignments.
        ended.sort(key=lambda p: p.cost)
        return ended[0].get_alignments() if len(ended) > 0 else []

    def _align_identical_inputs(self) -> list[Alignment]:
        """Return alignments for identical reference and hypothesis pairs."""
        assert self._identical_inputs, "Inputs are not identical."

        alignments = []
        for i in range(len(self._ref_token_matches)):
            alignment = self._get_match_alignment_from_token_indices(i, i)
            alignments.append(alignment)
        return alignments

    def _align_post_word_level(
        self,
        src: list[tuple[OpType, Union[SubgraphMetadata, range, tuple[int, int]]]],
        beam_size: int,
    ) -> list[Alignment]:
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
                alignments_ = self._beam_search_alignment(src=src_, beam_size=beam_size)
                alignments.extend(alignments_)

        return alignments

    def _get_match_alignment_from_token_indices(self, hyp_idx: int, ref_idx: int) -> Alignment:
        """Get a MATCH alignment for the given token indices."""
        ref_token_match = self._ref_token_matches[ref_idx]
        hyp_token_match = self._hyp_token_matches[hyp_idx]
        ref_slice = slice(*ref_token_match.span())
        hyp_slice = slice(*hyp_token_match.span())
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
            slice_ = slice(*token_match.span())
            token = self.hyp_raw[slice_]
            return Alignment(
                op_type=op_type,
                hyp_slice=slice_,
                hyp=token,
            )
        elif op_type == OpType.DELETE:
            token_match = self._ref_token_matches[token_idx]
            slice_ = slice(*token_match.span())
            token = self.ref_raw[slice_]
            return Alignment(
                op_type=op_type,
                ref_slice=slice_,
                ref=token,
            )
        else:
            raise ValueError(f"Invalid operation type for insert/delete alignment: {op_type}")

    def _prepare_subspans_with_word_level_pass(
        self,
    ) -> list[tuple[OpType, Union[SubgraphMetadata, range, tuple[int, int]]]]:
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


class Path:
    """Class to represent a graph path."""

    __slots__ = (
        "src",
        "ref_idx",
        "hyp_idx",
        "last_ref_idx",
        "last_hyp_idx",
        "_closed_cost",
        "_open_cost",
        "_at_unambiguous_match_node",
        "_end_indices",
        "_alignments",
        "_alignments_index",
    )

    def __init__(self, src: SubgraphMetadata):
        """Initialize the Path class with a given path."""
        self.src = src
        self.ref_idx = -1
        self.hyp_idx = -1
        self.last_hyp_idx = -1
        self.last_ref_idx = -1
        self._closed_cost = 0
        self._open_cost = 0
        self._at_unambiguous_match_node = False
        self._end_indices = tuple()
        self._alignments = None
        self._alignments_index = None

    @property
    def pid(self):
        """Get the ID of the path used for pruning."""
        return hash((self.hyp_idx, self.ref_idx, self.last_hyp_idx, self.last_ref_idx))

    @property
    def cost(self):
        """Get the cost of the path."""
        return self._closed_cost + self._open_cost + self._substitution_penalty(self.hyp_idx, self.ref_idx)

    @property
    def norm_cost(self):
        """Get the normalized cost of the path."""
        cost = self.cost
        if cost == 0:
            return 0
        return cost / (self.ref_idx + self.hyp_idx + 3)  # NOTE: +3 to avoid zero division. Root = (-1,-1).

    @property
    def index(self):
        """Get the current node index of the path."""
        return (self.hyp_idx, self.ref_idx)

    @property
    def at_end(self):
        """Check if the path has reached the terminal node."""
        return self.hyp_idx == self.src.hyp_max_idx and self.ref_idx == self.src.ref_max_idx

    def get_alignments(self) -> list[Alignment]:
        """Get the alignments of the path."""

        # Return cached alignments if available and the path has not changed.
        if self._alignments is not None and self._alignments_index == self.index:
            return self._alignments

        # Compute alignments from the segment end indices.
        self._alignments_index = self.index
        alignments = []
        start_hyp, start_ref = (0, 0)
        for end_hyp, end_ref, score in self._end_indices:
            end_hyp, end_ref = end_hyp + 1, end_ref + 1

            if start_hyp == end_hyp:
                alignment = self._get_delete_alignment(start_ref, end_ref)
                alignments.append(alignment)
            elif start_ref == end_ref:
                alignment = self._get_insert_alignment(start_hyp, end_hyp)
                alignments.append(alignment)
            else:
                alignment = self._get_match_or_substitution_alignment(start_hyp, end_hyp, start_ref, end_ref, score)
                alignments.append(alignment)

            start_hyp, start_ref = end_hyp, end_ref

        # Cache the computed alignments.
        self._alignments = alignments

        return alignments

    def expand(self):
        """Expand the path by transitioning to child nodes.

        Yields:
            Path: The expanded child paths.

        """
        # Add delete operation.
        delete_path = self._add_delete()
        if delete_path is not None:
            yield delete_path

        # Add insert operation.
        insert_path = self._add_insert()
        if insert_path is not None:
            yield insert_path

        # Add substitution or match operation.
        sub_or_match_path = self._add_substitution_or_match()
        if sub_or_match_path is not None:
            yield sub_or_match_path

    def _get_delete_alignment(self, start_ref_idx: int, end_ref_idx: int) -> Alignment:
        """Get a DELETE alignment for a given reference slice."""
        ref_slice = slice(start_ref_idx, end_ref_idx)
        ref_slice = self._translate_slice(ref_slice, self.src.ref_idx_map)
        return Alignment(
            op_type=OpType.DELETE,
            ref_slice=ref_slice,
            ref=self.src.ref_raw[ref_slice],
        )

    def _get_insert_alignment(self, start_hyp_idx: int, end_hyp_idx: int) -> Alignment:
        """Get an INSERT alignment for a given hypothesis slice."""
        hyp_slice = slice(start_hyp_idx, end_hyp_idx)
        hyp_slice = self._translate_slice(hyp_slice, self.src.hyp_idx_map)
        return Alignment(
            op_type=OpType.INSERT,
            hyp_slice=hyp_slice,
            hyp=self.src.hyp_raw[hyp_slice],
        )

    def _get_match_or_substitution_alignment(
        self,
        start_hyp_idx: int,
        end_hyp_idx: int,
        start_ref_idx: int,
        end_ref_idx: int,
        score: int,
    ) -> Alignment:
        """Get a MATCH or SUBSTITUTE alignment for given hypothesis and reference slices."""
        hyp_slice = slice(start_hyp_idx, end_hyp_idx)
        ref_slice = slice(start_ref_idx, end_ref_idx)
        hyp_slice = self._translate_slice(hyp_slice, self.src.hyp_idx_map)
        ref_slice = self._translate_slice(ref_slice, self.src.ref_idx_map)
        is_match_segment = score == 0
        op_type = OpType.MATCH if is_match_segment else OpType.SUBSTITUTE
        return Alignment(
            op_type=op_type,
            ref_slice=ref_slice,
            hyp_slice=hyp_slice,
            ref=self.src.ref_raw[ref_slice],
            hyp=self.src.hyp_raw[hyp_slice],
            left_compound=self.src.hyp_idx_map[start_hyp_idx] >= 0,
            right_compound=self.src.hyp_idx_map[end_hyp_idx - 1] >= 0,
        )

    def _transition_to_child_node(self, ref_step: int, hyp_step: int):
        """Transition to a child node by creating a new Path instance."""
        new_path = Path.__new__(Path)  # NOTE: Bypass __init__ for shallow copy.
        new_path.src = self.src
        new_path.ref_idx = self.ref_idx + ref_step
        new_path.hyp_idx = self.hyp_idx + hyp_step
        new_path.last_hyp_idx = self.last_hyp_idx
        new_path.last_ref_idx = self.last_ref_idx
        new_path._closed_cost = self._closed_cost
        new_path._open_cost = self._open_cost
        new_path._at_unambiguous_match_node = False
        new_path._end_indices = self._end_indices
        new_path._alignments = None
        new_path._alignments_index = None

        return new_path

    def _reset_segment_variables(self, hyp_idx: int, ref_idx: int) -> None:
        """Apply updates when segment end is detected."""
        self._closed_cost += self._open_cost
        self._closed_cost += self._substitution_penalty(hyp_idx, ref_idx)
        self.last_hyp_idx = hyp_idx
        self.last_ref_idx = ref_idx
        self._open_cost = 0

    def _end_insertion_segment(self, hyp_idx: int, ref_idx: int) -> None:
        """End the current segment, if criteria for an insertion are met."""
        hyp_slice = slice(self.last_hyp_idx + 1, hyp_idx + 1)
        hyp_slice = self._translate_slice(hyp_slice, self.src.hyp_idx_map)
        ref_is_empty = ref_idx == self.last_ref_idx
        if hyp_slice is not None and ref_is_empty:
            self._end_indices += ((hyp_idx, ref_idx, self._open_cost),)
            self._reset_segment_variables(hyp_idx, ref_idx)

    def _end_segment(self) -> Union[None, "Path"]:
        """End the current segment, if criteria for an insertion, a substitution, or a match are met."""
        hyp_slice = slice(self.last_hyp_idx + 1, self.hyp_idx + 1)
        hyp_slice = self._translate_slice(hyp_slice, self.src.hyp_idx_map)
        ref_slice = slice(self.last_ref_idx + 1, self.ref_idx + 1)
        ref_slice = self._translate_slice(ref_slice, self.src.ref_idx_map)

        assert ref_slice is not None

        hyp_is_empty = self.hyp_idx == self.last_hyp_idx
        if hyp_is_empty:
            self._end_indices += ((self.hyp_idx, self.ref_idx, self._open_cost),)
        else:
            # TODO: Handle edge case where hyp has only covered delimiters.
            if hyp_slice is None:
                return None

            is_match_segment = self._open_cost == 0
            self._at_unambiguous_match_node = is_match_segment and self.index in self.src.unambiguous_matches
            self._end_indices += ((self.hyp_idx, self.ref_idx, self._open_cost),)

        # Update the path score and reset segments attributes.
        self._reset_segment_variables(self.hyp_idx, self.ref_idx)
        return self

    def _in_backtrace_node_set(self, index) -> bool:
        """Check if the given operation is an optimal transition at the current index."""
        return index in self.src.backtrace_node_set

    def _add_delete(self) -> Union[None, "Path"]:
        """Expand the path by adding a delete operation."""
        # Ensure we are not at the end of the hypothesis sequence.
        if self.hyp_idx >= self.src.hyp_max_idx:
            return None

        # Transition and update costs.
        new_path = self._transition_to_child_node(ref_step=0, hyp_step=1)
        is_backtrace = self._in_backtrace_node_set(self.index)
        is_delimiter = self.src.hyp_char_types[new_path.hyp_idx] == 0  # NOTE: 0 indicates delimiter.
        new_path._open_cost += 1 if is_delimiter else 2
        new_path._open_cost += 0 if is_backtrace or is_delimiter else 1

        # Check for end-of-segment criteria.
        if self.src.hyp[new_path.hyp_idx] == END_DELIMITER:
            new_path._end_insertion_segment(new_path.hyp_idx, new_path.ref_idx)

        return new_path

    def _add_insert(self) -> Union[None, "Path"]:
        """Expand the path by adding an insert operation."""
        # Ensure we are not at the end of the reference sequence.
        if self.ref_idx >= self.src.ref_max_idx:
            return None

        # Transition and check for end-of-segment criteria.
        new_path = self._transition_to_child_node(ref_step=1, hyp_step=0)
        if self.src.ref[new_path.ref_idx] == START_DELIMITER:
            new_path._end_insertion_segment(self.hyp_idx, self.ref_idx)

        # Update costs.
        is_backtrace = self._in_backtrace_node_set(self.index)
        is_delimiter = self.src.ref_char_types[new_path.ref_idx] == 0  # NOTE: 0 indicates delimiter.
        new_path._open_cost += 1 if is_delimiter else 2
        new_path._open_cost += 0 if is_backtrace or is_delimiter else 1

        # Check for end-of-segment criteria.
        if self.src.ref[new_path.ref_idx] == END_DELIMITER:
            new_path = new_path._end_segment()

        return new_path

    def _add_substitution_or_match(self) -> Union[None, "Path"]:
        """Expand the given path by adding a substitution or match operation."""
        # Ensure we are not at the end of either sequence.
        if self.ref_idx >= self.src.ref_max_idx or self.hyp_idx >= self.src.hyp_max_idx:
            return None

        # Transition and ensure that the transition is allowed.
        new_path = self._transition_to_child_node(ref_step=1, hyp_step=1)
        is_match = self.src.ref[new_path.ref_idx] == self.src.hyp[new_path.hyp_idx]
        if not is_match:
            ref_is_delimiter = self.src.ref_char_types[new_path.ref_idx] == 0  # NOTE: 0 indicates delimiter
            hyp_is_delimiter = self.src.hyp_char_types[new_path.hyp_idx] == 0  # NOTE: 0 indicates delimiter
            if ref_is_delimiter or hyp_is_delimiter:
                return None

        # Check for end-of-segment criteria.
        if self.src.ref[new_path.ref_idx] == START_DELIMITER:
            new_path._end_insertion_segment(self.hyp_idx, self.ref_idx)

        # Update costs, if not a match.
        if not is_match:
            is_backtrace = self._in_backtrace_node_set(self.index)
            is_letter_type_match = (
                self.src.ref_char_types[new_path.ref_idx] == self.src.hyp_char_types[new_path.hyp_idx]
            )
            new_path._open_cost += 2 if is_letter_type_match else 3
            new_path._open_cost += 0 if is_backtrace else 1

        # Check for end-of-segment criteria.
        if self.src.ref[new_path.ref_idx] == END_DELIMITER:
            new_path = new_path._end_segment()

        return new_path

    def _translate_slice(self, segment_slice: slice, index_map: list[int]) -> None | slice:
        """Translate a slice from the alignment sequence back to the original sequence."""
        slice_indices = index_map[segment_slice]
        slice_indices = list(filter(lambda x: x >= 0, slice_indices))
        if len(slice_indices) == 0:
            return None
        start, end = int(slice_indices[0]), int(slice_indices[-1] + 1)
        return slice(start, end)

    def _substitution_penalty(self, hyp_idx: int, ref_idx: int) -> int:
        """Get the substitution penalty given an index."""
        # NOTE: Since *_idx is guaranteed to be equal to or higher than last_*_idx, we only need to check for equality.
        if ref_idx == self.last_ref_idx or hyp_idx == self.last_hyp_idx:
            return 0
        return self._open_cost

    def __repr__(self):
        return f"Path(({self.ref_idx}, {self.hyp_idx}), score={self.cost})"
