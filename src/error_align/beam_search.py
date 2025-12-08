from __future__ import annotations

from typing import TYPE_CHECKING, Union

from error_align.utils import END_DELIMITER, START_DELIMITER, translate_slice

if TYPE_CHECKING:
    from error_align.subgraph_metadata import SubgraphMetadata

# def translate_slice(segment_slice: slice, index_map: list[int]) -> None | slice:
#     """Translate a slice from the alignment sequence back to the original sequence.

#     Args:
#         segment_slice (slice): The slice in the alignment sequence.
#         index_map (list[int]): The mapping from alignment indices to original sequence indices.

#     Returns:
#         None | slice: The translated slice in the original sequence, or None if no valid indices.

#     """
#     slice_indices = index_map[segment_slice]
#     slice_indices = list(filter(lambda x: x >= 0, slice_indices))
#     if len(slice_indices) == 0:
#         return None
#     start, end = int(slice_indices[0]), int(slice_indices[-1] + 1)
#     return slice(start, end)


# START_DELIMITER = "<"
# END_DELIMITER = ">"


# ============================================================
# PATH CLASS
# ============================================================


MASK = (1 << 64) - 1
B = 146527


def update(h, t):
    return (h * B + t) & MASK


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
        "_hash_id",
    )

    def __init__(self, src: SubgraphMetadata):
        """Initialize the Path class with a given path."""
        self.src = src
        self.ref_idx: int = -1
        self.hyp_idx: int = -1
        self.last_hyp_idx: int = -1
        self.last_ref_idx: int = -1
        self._closed_cost: float = 0
        self._open_cost: float = 0
        self._at_unambiguous_match_node = False
        self._end_indices = tuple()
        self._hash_id = 0

    @property
    def pid(self) -> int:
        """Get the ID of the path used for pruning."""
        return hash((self.hyp_idx, self.ref_idx, self.last_hyp_idx, self.last_ref_idx))

    @property
    def cost(self) -> float:
        """Get the cost of the path."""
        is_sub = _is_substitution(self.hyp_idx, self.ref_idx, self.last_hyp_idx, self.last_ref_idx)
        return self._closed_cost + self._open_cost + (self._open_cost if is_sub else 0)

    @property
    def norm_cost(self) -> float:
        """Get the normalized cost of the path."""
        cost = self.cost
        if cost == 0:
            return 0
        return cost / (self.ref_idx + self.hyp_idx + 3)  # NOTE: +3 to avoid zero division. Root = (-1,-1).

    @property
    def index(self) -> tuple[int, int]:
        """Get the current node index of the path."""
        return (self.hyp_idx, self.ref_idx)

    @property
    def at_end(self) -> bool:
        """Check if the path has reached the terminal node."""
        return self.hyp_idx == self.src.hyp_max_idx and self.ref_idx == self.src.ref_max_idx

    def __repr__(self) -> str:
        return f"Path(({self.ref_idx}, {self.hyp_idx}), score={self.cost})"


# ============================================================
# PATH EXPANSION
# ============================================================


def expand(parent: Path):
    """Expand the path by transitioning to child nodes.

    Yields:
        Path: The expanded child paths.
    """
    # Add delete operation.
    delete_path = _add_delete(parent)
    if delete_path is not None:
        yield delete_path

    # Add insert operation.
    insert_path = _add_insert(parent)
    if insert_path is not None:
        yield insert_path

    # Add substitution or match operation.
    sub_or_match_path = _add_substitution_or_match(parent)
    if sub_or_match_path is not None:
        yield sub_or_match_path


def _add_substitution_or_match(parent: Path) -> Union[None, Path]:
    """Expand the given path by adding a substitution or match operation."""
    # Ensure we are not at the end of either sequence.
    if parent.ref_idx >= parent.src.ref_max_idx or parent.hyp_idx >= parent.src.hyp_max_idx:
        return None

    # Transition and ensure that the transition is allowed.
    child = _transition_to_child_node(parent, ref_step=1, hyp_step=1)
    is_match = parent.src.ref[child.ref_idx] == parent.src.hyp[child.hyp_idx]
    if not is_match:
        ref_is_delimiter = parent.src.ref_char_types[child.ref_idx] == 0  # NOTE: 0 indicates delimiter
        hyp_is_delimiter = parent.src.hyp_char_types[child.hyp_idx] == 0  # NOTE: 0 indicates delimiter
        if ref_is_delimiter or hyp_is_delimiter:
            return None

    # Check for end-of-segment criteria.
    if parent.src.ref[child.ref_idx] == START_DELIMITER:
        _end_insertion_segment(child, parent.hyp_idx, parent.ref_idx)

    # Update costs, if not a match.
    if not is_match:
        is_backtrace = parent.index in parent.src.backtrace_node_set
        is_letter_type_match = parent.src.ref_char_types[child.ref_idx] == parent.src.hyp_char_types[child.hyp_idx]
        child._open_cost += 2 if is_letter_type_match else 3
        child._open_cost += 0 if is_backtrace else 1

    # Check for end-of-segment criteria.
    if child.src.ref[child.ref_idx] == END_DELIMITER:
        child = _end_segment(child)

    return child


def _add_insert(parent: Path) -> Union[None, Path]:
    """Expand the path by adding an insert operation."""
    # Ensure we are not at the end of the reference sequence.
    if parent.ref_idx >= parent.src.ref_max_idx:
        return None

    # Transition and check for end-of-segment criteria.
    child = _transition_to_child_node(parent, ref_step=1, hyp_step=0)
    if parent.src.ref[child.ref_idx] == START_DELIMITER:
        _end_insertion_segment(child, parent.hyp_idx, parent.ref_idx)

    # Update costs.
    is_backtrace = parent.index in parent.src.backtrace_node_set
    is_delimiter = parent.src.ref_char_types[child.ref_idx] == 0  # NOTE: 0 indicates delimiter.
    child._open_cost += 1 if is_delimiter else 2
    child._open_cost += 0 if is_backtrace or is_delimiter else 1

    # Check for end-of-segment criteria.
    if child.src.ref[child.ref_idx] == END_DELIMITER:
        child = _end_segment(child)

    return child


def _add_delete(parent: Path) -> Union[None, Path]:
    """Expand the path by adding a delete operation."""
    # Ensure we are not at the end of the hypothesis sequence.
    if parent.hyp_idx >= parent.src.hyp_max_idx:
        return None

    # Transition and update costs.
    child = _transition_to_child_node(parent, ref_step=0, hyp_step=1)
    is_backtrace = parent.index in parent.src.backtrace_node_set
    is_delimiter = parent.src.hyp_char_types[child.hyp_idx] == 0  # NOTE: 0 indicates delimiter.
    child._open_cost += 1 if is_delimiter else 2
    child._open_cost += 0 if is_backtrace or is_delimiter else 1

    # Check for end-of-segment criteria.
    if child.src.hyp[child.hyp_idx] == END_DELIMITER:
        _end_insertion_segment(child, child.hyp_idx, child.ref_idx)

    return child


# ============================================================
# PATH EXPANSION HELPERS
# ============================================================


def _reset_segment_variables(path: Path, hyp_idx: int, ref_idx: int) -> None:
    """Apply updates when segment end is detected."""
    path._closed_cost += path._open_cost
    is_sub = _is_substitution(hyp_idx, ref_idx, path.last_hyp_idx, path.last_ref_idx)
    path._closed_cost += path._open_cost if is_sub else 0
    path.last_hyp_idx = hyp_idx
    path.last_ref_idx = ref_idx
    path._open_cost = 0


def _end_insertion_segment(path: Path, hyp_idx: int, ref_idx: int) -> None:
    """End the current segment, if criteria for an insertion are met."""
    hyp_slice = slice(path.last_hyp_idx + 1, hyp_idx + 1)
    hyp_slice = translate_slice(hyp_slice, path.src.hyp_idx_map)
    ref_is_empty = ref_idx == path.last_ref_idx
    if hyp_slice is not None and ref_is_empty:
        path._end_indices += ((hyp_idx, ref_idx, path._open_cost),)
        _reset_segment_variables(path, hyp_idx, ref_idx)


def _end_segment(path: Path) -> Union[None, "Path"]:
    """End the current segment, if criteria for an insertion, a substitution, or a match are met."""
    hyp_slice = slice(path.last_hyp_idx + 1, path.hyp_idx + 1)
    hyp_slice = translate_slice(hyp_slice, path.src.hyp_idx_map)
    ref_slice = slice(path.last_ref_idx + 1, path.ref_idx + 1)
    ref_slice = translate_slice(ref_slice, path.src.ref_idx_map)

    assert ref_slice is not None

    hyp_is_empty = path.hyp_idx == path.last_hyp_idx
    if hyp_is_empty:
        path._end_indices += ((path.hyp_idx, path.ref_idx, path._open_cost),)
    else:
        # TODO: Handle edge case where hyp has only covered delimiters.
        if hyp_slice is None:
            return None

        is_match_segment = path._open_cost == 0
        path._at_unambiguous_match_node = is_match_segment and path.index in path.src.unambiguous_matches
        path._end_indices += ((path.hyp_idx, path.ref_idx, path._open_cost),)

    # Update the path score and reset segments attributes.
    _reset_segment_variables(path, path.hyp_idx, path.ref_idx)
    return path


def _transition_to_child_node(parent: Path, ref_step: int, hyp_step: int):
    """Transition to a child node by creating a new Path instance."""
    child = Path.__new__(Path)  # NOTE: Bypass __init__ for shallow copy.
    child.src = parent.src
    child.ref_idx = parent.ref_idx + ref_step
    child.hyp_idx = parent.hyp_idx + hyp_step
    child.last_hyp_idx = parent.last_hyp_idx
    child.last_ref_idx = parent.last_ref_idx
    child._closed_cost = parent._closed_cost
    child._open_cost = parent._open_cost
    child._at_unambiguous_match_node = False
    child._end_indices = parent._end_indices
    child._hash_id = update(parent._hash_id, (ref_step * 2) + hyp_step)

    return child


def _is_substitution(hyp_idx: int, ref_idx: int, last_hyp_idx: int, last_ref_idx: int) -> int:
    """Get the substitution penalty given an index."""
    # NOTE: Since *_idx is guaranteed to be equal to or higher than last_*_idx, we only need to check for equality.
    if ref_idx == last_ref_idx or hyp_idx == last_hyp_idx:
        return False
    return True


# ============================================================
# MAIN BEAM SEARCH FUNCTION
# ============================================================


def _get_path_hash(path: Path) -> int:
    """Get a hash for the path based on its end indices."""
    x = tuple(
        [
            path.ref_idx,
            path.hyp_idx,
            path.last_ref_idx,
            path.last_hyp_idx,
            path._closed_cost,
            path._open_cost,
            int(path.cost),
            round(float(path.norm_cost), 10),
            tuple([(i, j) for i, j, _ in path._end_indices]),
        ]
    )
    return hash(x)


def cpp_path_to_py_path(cpp_path) -> Path:
    """Convert a C++ Path object to a Python Path object."""
    py_path = Path.__new__(Path)  # Bypass __init__
    py_path.src = cpp_path.src
    py_path.ref_idx = cpp_path.ref_idx
    py_path.hyp_idx = cpp_path.hyp_idx
    py_path.last_ref_idx = cpp_path.last_ref_idx
    py_path.last_hyp_idx = cpp_path.last_hyp_idx
    py_path._closed_cost = cpp_path._closed_cost
    py_path._open_cost = cpp_path._open_cost
    py_path._at_unambiguous_match_node = cpp_path._at_unambiguous_match_node
    py_path._end_indices = cpp_path._end_indices
    return py_path


def error_align_beam_search(src: SubgraphMetadata, beam_size: int = 100) -> Path:
    """Perform beam search to align reference and hypothesis texts for a given source."""
    # Initialize the beam with a single path starting at the root node.
    start_path = Path(src)
    beam = {start_path.pid: start_path}
    prune_map = dict()
    ended = []

    # Expand candidate paths until all have reached the terminal node.
    # i = 0
    while len(beam) > 0:
        new_beam = {}

        # Expand each path in the current beam.
        for path in beam.values():
            if path.at_end:
                ended.append(path)
                continue

            # # Transition to all child nodes.
            # py_children = list(expand(path))
            # cpp_children = cpp_expand_paths(path)

            # for i, child in enumerate(cpp_children):
            #     child_hash = _get_path_hash(child)
            #     try:
            #         child.cost = py_children[i].cost
            #         child.norm_cost = py_children[i].norm_cost
            #         child.pid = py_children[i].pid
            #     except AssertionError as e:
            #         # Handle the error (e.g., log it, raise a different exception, etc.)
            #         import IPython; IPython.embed(using=False); exit()

            # for new_path in cpp_children:
            for new_path in expand(path):
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
        new_beam.sort(key=lambda p: (p.norm_cost, p._hash_id))
        beam = new_beam[:beam_size]

        # Keep only the best path if, it matches the segment.
        if len(beam) > 0 and beam[0]._at_unambiguous_match_node:
            beam = beam[:1]
            prune_map = dict()
        beam = {p.pid: p for p in beam}  # Convert to dict for diversity check.

    # Return the best path or its alignments.
    if len(ended) == 0:
        return []
    # ended.sort(key=lambda p: (p.cost, _get_path_hash(p)))
    ended.sort(key=lambda p: (p.cost, p._hash_id))
    return ended[0]
