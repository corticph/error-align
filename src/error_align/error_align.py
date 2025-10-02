from collections import defaultdict
from typing import Union

import regex as re
from tqdm import tqdm

from error_align.backtrace_graph import BacktraceGraph
from error_align.edit_distance import compute_error_align_distance_matrix
from error_align.utils import (
    END_DELIMITER,
    START_DELIMITER,
    Alignment,
    OpType,
    basic_normalizer,
    basic_tokenizer,
    categorize_char,
    ensure_length_preservation,
    get_manhattan_distance,
)


class ErrorAlign:
    """Error alignment class that performs a two-pass alignment process."""

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
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

        self.ref = ref
        self.hyp = hyp

        # Inclusive tokenization: Track the token position in the original text.
        self._ref_token_matches = tokenizer(ref)
        self._hyp_token_matches = tokenizer(hyp)

        # Length-preserving normalization: Ensure that the normalizer preserves token length.
        normalizer = ensure_length_preservation(normalizer)
        self._ref = "".join([f"<{normalizer(r.group())}>" for r in self._ref_token_matches])
        self._hyp = "".join([f"<{normalizer(h.group())}>" for h in self._hyp_token_matches])

        # Categorize characters.
        self._ref_char_types = list(map(categorize_char, self._ref))
        self._hyp_char_types = list(map(categorize_char, self._hyp))

        # Initialize graph attributes.
        self._identical_inputs = self._ref == self._hyp
        self._ref_max_idx = len(self._ref) - 1
        self._hyp_max_idx = len(self._hyp) - 1
        self.end_index = (self._hyp_max_idx, self._ref_max_idx)

        # Create index maps for reference and hypothesis sequences.
        self._ref_index_map = self._create_index_map(self._ref_token_matches)
        self._hyp_index_map = self._create_index_map(self._hyp_token_matches)

        # First pass: Extract backtrace graph.
        if not self._identical_inputs:
            _, backtrace_matrix = compute_error_align_distance_matrix(self._ref, self._hyp, backtrace=True)
            self._backtrace_graph = BacktraceGraph(backtrace_matrix)
            self._backtrace_node_set = self._backtrace_graph.get_node_set()
            self._unambiguous_matches = self._backtrace_graph.get_unambiguous_matches(self._ref)
        else:
            self._backtrace_graph = None
            self._backtrace_node_set = None
            self._unambiguous_matches = None

    def __repr__(self):
        ref_preview = self.ref if len(self.ref) < 20 else self.ref[:17] + "..."
        hyp_preview = self.hyp if len(self.hyp) < 20 else self.hyp[:17] + "..."
        return f'ErrorAlign(ref="{ref_preview}", hyp="{hyp_preview}")'

    def align(
        self,
        beam_size: int = 100,
        pbar: bool = False,
        return_path: bool = False,
    ) -> Union[list[Alignment], "Path"]:
        """Perform beam search to align reference and hypothesis texts.

        Args:
            beam_size (int): The size of the beam for beam search. Defaults to 100.
            pbar (bool): Whether to display a progress bar. Defaults to False.
            return_path (bool): Whether to return the path object or just the alignments. Defaults to False.

        Returns:
            list[Alignment]: A list of Alignment objects.

        """
        # Skip beam search if inputs are identical.
        if self._identical_inputs:
            return self._identical_input_alignments()

        # Initialize the beam with a single path starting at the root node.
        start_path = Path(self)
        beam = {start_path.pid: start_path}
        prune_map = defaultdict(lambda: float("inf"))
        ended = []

        # Setup progress bar, if enabled.
        if pbar:
            total_mdist = self._ref_max_idx + self._hyp_max_idx + 2
            progress_bar = tqdm(total=total_mdist, desc="Aligning transcripts")

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
                    if new_path.pid in prune_map:
                        if new_path.cost > prune_map[new_path.pid]:
                            continue
                    prune_map[new_path.pid] = new_path.cost

                    if new_path.pid not in new_beam or new_path.cost < new_beam[new_path.pid].cost:
                        new_beam[new_path.pid] = new_path

            # Update the beam with the newly expanded paths.
            new_beam = list(new_beam.values())
            new_beam.sort(key=lambda p: p.norm_cost)
            beam = new_beam[:beam_size]

            # Keep only the best path if, it matches the segment.
            if len(beam) > 0 and beam[0]._at_unambiguous_match_node:
                beam = beam[:1]
                prune_map = defaultdict(lambda: float("inf"))
            beam = {p.pid: p for p in beam}  # Convert to dict for diversity check.

            # Update progress bar, if enabled.
            try:
                worst_path = next(reversed(beam.values()))
                mdist = get_manhattan_distance(worst_path.index, self.end_index)
                if pbar:
                    progress_bar.n = total_mdist - mdist
                    progress_bar.refresh()
            except StopIteration:
                if pbar:
                    progress_bar.n = total_mdist
                    progress_bar.refresh()

        # Return the best path or its alignments.
        ended.sort(key=lambda p: p.cost)
        if return_path:
            return ended[0] if len(ended) > 0 else None
        return ended[0].alignments if len(ended) > 0 else []

    def _create_index_map(self, text_tokens: list[re.Match]) -> list[int]:
        """Create an index map for the given tokens.

        The 'index_map' is used to map each aligned character back to its original position in the input text.

        NOTE: -1 is used for delimiter (<>) and indicates no match in the source sequence.
        """
        index_map = []
        for match in text_tokens:
            index_map.extend([-1])  # Start delimiter
            index_map.extend(list(range(*match.span())))
            index_map.extend([-1])  # End delimiter
        return index_map

    def _identical_input_alignments(self) -> list[Alignment]:
        """Return alignments for identical reference and hypothesis pairs."""
        assert self._identical_inputs, "Inputs are not identical."

        alignments = []
        for ref_match, hyp_match in zip(self._ref_token_matches, self._hyp_token_matches, strict=False):
            ref_slice = slice(*ref_match.span())
            hyp_slice = slice(*hyp_match.span())
            ref_token = self.ref[ref_slice]
            hyp_token = self.hyp[hyp_slice]
            alignment = Alignment(
                op_type=OpType.MATCH,
                ref_slice=ref_slice,
                hyp_slice=hyp_slice,
                ref=ref_token,
                hyp=hyp_token,
            )
            alignments.append(alignment)
        return alignments


class Path:
    """Class to represent a graph path."""

    def __init__(self, src: ErrorAlign):
        """Initialize the Path class with a given path."""
        self.src = src
        self.ref_idx = -1
        self.hyp_idx = -1
        self._closed_cost = 0
        self._open_cost = 0
        self._at_unambiguous_match_node = False
        self._last_end_index = (-1, -1)
        self._end_indices = tuple()
        self._alignments = None
        self._alignments_index = None

    def __repr__(self):
        return f"Path(({self.ref_idx}, {self.hyp_idx}), score={self.cost})"

    @property
    def alignments(self) -> list[Alignment]:
        """Get the alignments of the path."""
        # Return cached alignments if available and the path has not changed.
        if self._alignments is not None and self._alignments_index == self.index:
            return self._alignments

        self._alignments_index = self.index
        alignments = []
        start_hyp, start_ref = (0, 0)
        for (end_hyp, end_ref), score in self._end_indices:
            end_hyp, end_ref = end_hyp + 1, end_ref + 1

            # Construct DELETE alignment.
            if start_hyp == end_hyp:
                assert start_ref < end_ref
                ref_slice = slice(start_ref, end_ref)
                ref_slice = self._translate_slice(ref_slice, self.src._ref_index_map)
                assert ref_slice is not None
                alignment = Alignment(
                    op_type=OpType.DELETE,
                    ref_slice=ref_slice,
                    ref=self.src.ref[ref_slice],
                )
                alignments.append(alignment)

            # Construct INSERT alignment.
            elif start_ref == end_ref:
                assert start_hyp < end_hyp
                hyp_slice = slice(start_hyp, end_hyp)
                hyp_slice = self._translate_slice(hyp_slice, self.src._hyp_index_map)
                assert hyp_slice is not None
                alignment = Alignment(
                    op_type=OpType.INSERT,
                    hyp_slice=hyp_slice,
                    hyp=self.src.hyp[hyp_slice],
                    left_compound=self.src._hyp_index_map[start_hyp] >= 0,
                    right_compound=self.src._hyp_index_map[end_hyp - 1] >= 0,
                )
                alignments.append(alignment)

            # Construct SUBSTITUTE or MATCH alignment.
            else:
                assert start_hyp < end_hyp and start_ref < end_ref
                hyp_slice = slice(start_hyp, end_hyp)
                ref_slice = slice(start_ref, end_ref)
                hyp_slice = self._translate_slice(hyp_slice, self.src._hyp_index_map)
                ref_slice = self._translate_slice(ref_slice, self.src._ref_index_map)
                assert hyp_slice is not None and ref_slice is not None
                is_match_segment = score == 0
                op_type = OpType.MATCH if is_match_segment else OpType.SUBSTITUTE
                alignment = Alignment(
                    op_type=op_type,
                    ref_slice=ref_slice,
                    hyp_slice=hyp_slice,
                    ref=self.src.ref[ref_slice],
                    hyp=self.src.hyp[hyp_slice],
                    left_compound=self.src._hyp_index_map[start_hyp] >= 0,
                    right_compound=self.src._hyp_index_map[end_hyp - 1] >= 0,
                )
                alignments.append(alignment)

            start_hyp, start_ref = end_hyp, end_ref

        # Cache the computed alignments.
        self._alignments = alignments

        return alignments

    @property
    def pid(self):
        """Get the ID of the path used for pruning."""
        return hash((self.index, self._last_end_index))

    @property
    def cost(self):
        """Get the cost of the path."""
        return self._closed_cost + self._open_cost + self._substitution_penalty()

    @property
    def norm_cost(self):
        """Get the normalized cost of the path."""
        if self.cost == 0:
            return 0
        return self.cost / (self.ref_idx + self.hyp_idx + 3)  # NOTE: +3 to avoid zero division. Root = (-1,-1).

    @property
    def index(self):
        """Get the current node index of the path."""
        return (self.hyp_idx, self.ref_idx)

    @property
    def at_end(self):
        """Check if the path has reached the terminal node."""
        return self.index == self.src.end_index

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

    def _transition_and_shallow_copy(self, ref_step: int, hyp_step: int):
        """Create a shallow copy of the path."""
        new_path = Path(self.src)
        new_path.ref_idx = self.ref_idx + ref_step
        new_path.hyp_idx = self.hyp_idx + hyp_step
        new_path._closed_cost = self._closed_cost
        new_path._open_cost = self._open_cost
        new_path._at_unambiguous_match_node = False
        new_path._last_end_index = self._last_end_index
        new_path._end_indices = self._end_indices

        return new_path

    def _reset_segment_variables(self, index: tuple[int, int]) -> None:
        """Apply updates when segment end is detected."""
        self._closed_cost += self._open_cost
        self._closed_cost += self._substitution_penalty(index)
        self._last_end_index = index
        self._open_cost = 0

    def _end_insertion_segment(self, index: tuple[int, int]) -> None:
        """End the current segment, if criteria for an insertion are met."""
        hyp_slice = slice(self._last_end_index[0] + 1, index[0] + 1)
        hyp_slice = self._translate_slice(hyp_slice, self.src._hyp_index_map)
        ref_is_empty = index[1] == self._last_end_index[1]
        if hyp_slice is not None and ref_is_empty:
            self._end_indices += ((index, self._open_cost),)
            self._reset_segment_variables(index)

    def _end_segment(self) -> Union[None, "Path"]:
        """End the current segment, if criteria for an insertion, a substitution, or a match are met."""
        hyp_slice = slice(self._last_end_index[0] + 1, self.index[0] + 1)
        hyp_slice = self._translate_slice(hyp_slice, self.src._hyp_index_map)
        ref_slice = slice(self._last_end_index[1] + 1, self.index[1] + 1)
        ref_slice = self._translate_slice(ref_slice, self.src._ref_index_map)

        assert ref_slice is not None

        hyp_is_empty = self.index[0] == self._last_end_index[0]
        if hyp_is_empty:
            self._end_indices += ((self.index, self._open_cost),)
        else:
            # TODO: Handle edge case where hyp has only covered delimiters.
            if hyp_slice is None:
                return None

            is_match_segment = self._open_cost == 0
            self._at_unambiguous_match_node = is_match_segment and self.index in self.src._unambiguous_matches
            self._end_indices += ((self.index, self._open_cost),)

        # Update the path score and reset segments attributes.
        self._reset_segment_variables(self.index)
        return self

    def _in_backtrace_node_set(self, index) -> bool:
        """Check if the given operation is an optimal transition at the current index."""
        return index in self.src._backtrace_node_set

    def _add_delete(self) -> Union[None, "Path"]:
        """Expand the path by adding a delete operation."""
        # Ensure we are not at the end of the hypothesis sequence.
        if self.hyp_idx >= self.src._hyp_max_idx:
            return None

        # Transition and update costs.
        new_path = self._transition_and_shallow_copy(ref_step=0, hyp_step=1)
        is_backtrace = self._in_backtrace_node_set(self.index)
        is_delimiter = self.src._hyp_char_types[new_path.hyp_idx] == 0  # NOTE: 0 indicates delimiter.
        new_path._open_cost += 1 if is_delimiter else 2
        new_path._open_cost += 0 if is_backtrace or is_delimiter else 1

        # Check for end-of-segment criteria.
        if self.src._hyp[new_path.hyp_idx] == END_DELIMITER:
            new_path._end_insertion_segment(new_path.index)

        return new_path

    def _add_insert(self) -> Union[None, "Path"]:
        """Expand the path by adding an insert operation."""
        # Ensure we are not at the end of the reference sequence.
        if self.ref_idx >= self.src._ref_max_idx:
            return None

        # Transition and check for end-of-segment criteria.
        new_path = self._transition_and_shallow_copy(ref_step=1, hyp_step=0)
        if self.src._ref[new_path.ref_idx] == START_DELIMITER:
            new_path._end_insertion_segment(self.index)

        # Update costs.
        is_backtrace = self._in_backtrace_node_set(self.index)
        is_delimiter = self.src._ref_char_types[new_path.ref_idx] == 0  # NOTE: 0 indicates delimiter.
        new_path._open_cost += 1 if is_delimiter else 2
        new_path._open_cost += 0 if is_backtrace or is_delimiter else 1

        # Check for end-of-segment criteria.
        if self.src._ref[new_path.ref_idx] == END_DELIMITER:
            new_path = new_path._end_segment()

        return new_path

    def _add_substitution_or_match(self) -> Union[None, "Path"]:
        """Expand the given path by adding a substitution or match operation."""
        # Ensure we are not at the end of either sequence.
        if self.ref_idx >= self.src._ref_max_idx or self.hyp_idx >= self.src._hyp_max_idx:
            return None

        # Transition and ensure that the transition is allowed.
        new_path = self._transition_and_shallow_copy(ref_step=1, hyp_step=1)
        is_match = self.src._ref[new_path.ref_idx] == self.src._hyp[new_path.hyp_idx]
        if not is_match:
            ref_is_delimiter = self.src._ref_char_types[new_path.ref_idx] == 0  # NOTE: 0 indicates delimiter
            hyp_is_delimiter = self.src._hyp_char_types[new_path.hyp_idx] == 0  # NOTE: 0 indicates delimiter
            if ref_is_delimiter or hyp_is_delimiter:
                return None

        # Check for end-of-segment criteria.
        if self.src._ref[new_path.ref_idx] == START_DELIMITER:
            new_path._end_insertion_segment(self.index)

        # Update costs, if not a match.
        if not is_match:
            is_backtrace = self._in_backtrace_node_set(self.index)
            is_letter_type_match = (
                self.src._ref_char_types[new_path.ref_idx] == self.src._hyp_char_types[new_path.hyp_idx]
            )
            new_path._open_cost += 2 if is_letter_type_match else 3
            new_path._open_cost += 0 if is_backtrace else 1

        # Check for end-of-segment criteria.
        if self.src._ref[new_path.ref_idx] == END_DELIMITER:
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

    def _substitution_penalty(self, index: tuple[int, int] | None = None) -> int:
        """Get the substitution penalty given an index."""
        index = index or self.index
        ref_is_not_empty = index[1] > self._last_end_index[1]
        hyp_is_not_empty = index[0] > self._last_end_index[0]
        if ref_is_not_empty and hyp_is_not_empty:
            return self._open_cost
        return 0
