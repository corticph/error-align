from __future__ import annotations

from typing import TYPE_CHECKING, Union

from error_align.utils import END_DELIMITER, START_DELIMITER, translate_slice

if TYPE_CHECKING:
    from error_align.graph_metadata import SubgraphMetadata


INT64_MASK = (1 << 64) - 1
SORT_ID_BASE = 146527

# ============================================================
# PATH CLASS
# ============================================================


class Path:
    """Class to represent a graph path."""

    __slots__ = (
        "src",
        "ref_idx",
        "hyp_idx",
        "last_ref_idx",
        "last_hyp_idx",
        "closed_cost",
        "open_cost",
        "at_unambiguous_match_node",
        "end_indices",
        "sort_id",
    )

    def __init__(self, src: SubgraphMetadata):
        """Initialize the Path class with a given path."""
        self.src = src
        self.ref_idx: int = -1
        self.hyp_idx: int = -1
        self.last_hyp_idx: int = -1
        self.last_ref_idx: int = -1
        self.closed_cost: float = 0
        self.open_cost: float = 0
        self.at_unambiguous_match_node = False
        self.end_indices = tuple()
        self.sort_id = 0

    @property
    def prune_id(self) -> int:
        """Get the ID of the path used for pruning."""
        return hash((self.hyp_idx, self.ref_idx, self.last_hyp_idx, self.last_ref_idx))

    @property
    def cost(self) -> float:
        """Get the cost of the path."""
        is_sub = is_substitution(self.hyp_idx, self.ref_idx, self.last_hyp_idx, self.last_ref_idx)
        return self.closed_cost + self.open_cost + (self.open_cost if is_sub else 0)

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

    def update_sort_id(self, t: int) -> None:
        """Update the sort ID for path ordering. Ensures identical behavior as C++ implementation."""
        self.sort_id = (self.sort_id * SORT_ID_BASE + t) & INT64_MASK


# ============================================================
# PATH EXPANSION
# ============================================================


def expand(parent: Path):
    """Expand the path by transitioning to child nodes.

    Yields:
        Path: The expanded child paths.
    """
    # Add delete operation.
    delete_path = add_delete(parent)
    if delete_path is not None:
        yield delete_path

    # Add insert operation.
    insert_path = add_insert(parent)
    if insert_path is not None:
        yield insert_path

    # Add substitution or match operation.
    sub_or_match_path = add_substitution_or_match(parent)
    if sub_or_match_path is not None:
        yield sub_or_match_path


def add_substitution_or_match(parent: Path) -> Union[None, Path]:
    """Expand the given path by adding a substitution or match operation."""
    # Ensure we are not at the end of either sequence.
    if parent.ref_idx >= parent.src.ref_max_idx or parent.hyp_idx >= parent.src.hyp_max_idx:
        return None

    # Transition and ensure that the transition is allowed. If not, terminate.
    child = transition_to_child_node(parent, ref_step=1, hyp_step=1)
    is_match = parent.src.ref[child.ref_idx] == parent.src.hyp[child.hyp_idx]
    if not is_match:
        ref_is_delimiter = parent.src.ref_char_types[child.ref_idx] == 0  # NOTE: 0 indicates delimiter
        hyp_is_delimiter = parent.src.hyp_char_types[child.hyp_idx] == 0  # NOTE: 0 indicates delimiter
        if ref_is_delimiter or hyp_is_delimiter:
            return None

    # Check for end-of-segment criteria.
    if parent.src.ref[child.ref_idx] == START_DELIMITER:
        end_insertion_segment(child, parent.hyp_idx, parent.ref_idx)

    # Update costs, if not a match.
    if not is_match:
        is_backtrace = parent.index in parent.src.backtrace_node_set
        is_letter_type_match = parent.src.ref_char_types[child.ref_idx] == parent.src.hyp_char_types[child.hyp_idx]
        child.open_cost += 2 if is_letter_type_match else 3
        child.open_cost += 0 if is_backtrace else 1

    # Check for end-of-segment criteria.
    if child.src.ref[child.ref_idx] == END_DELIMITER:
        child = end_segment(child)

    return child


def add_insert(parent: Path) -> Union[None, Path]:
    """Expand the path by adding an insert operation."""
    # Ensure we are not at the end of the reference sequence.
    if parent.ref_idx >= parent.src.ref_max_idx:
        return None

    # Transition and check for end-of-segment criteria.
    child = transition_to_child_node(parent, ref_step=1, hyp_step=0)
    if parent.src.ref[child.ref_idx] == START_DELIMITER:
        end_insertion_segment(child, parent.hyp_idx, parent.ref_idx)

    # Update costs.
    is_backtrace = parent.index in parent.src.backtrace_node_set
    is_delimiter = parent.src.ref_char_types[child.ref_idx] == 0  # NOTE: 0 indicates delimiter.
    child.open_cost += 1 if is_delimiter else 2
    child.open_cost += 0 if is_backtrace or is_delimiter else 1

    # Check for end-of-segment criteria.
    if child.src.ref[child.ref_idx] == END_DELIMITER:
        child = end_segment(child)

    return child


def add_delete(parent: Path) -> Union[None, Path]:
    """Expand the path by adding a delete operation."""
    # Ensure we are not at the end of the hypothesis sequence.
    if parent.hyp_idx >= parent.src.hyp_max_idx:
        return None

    # Transition and update costs.
    child = transition_to_child_node(parent, ref_step=0, hyp_step=1)
    is_backtrace = parent.index in parent.src.backtrace_node_set
    is_delimiter = parent.src.hyp_char_types[child.hyp_idx] == 0  # NOTE: 0 indicates delimiter.
    child.open_cost += 1 if is_delimiter else 2
    child.open_cost += 0 if is_backtrace or is_delimiter else 1

    # Check for end-of-segment criteria.
    if child.src.hyp[child.hyp_idx] == END_DELIMITER:
        end_insertion_segment(child, child.hyp_idx, child.ref_idx)

    return child


# ============================================================
# PATH EXPANSION HELPERS
# ============================================================


def _reset_segment_variables(path: Path, hyp_idx: int, ref_idx: int) -> None:
    """Apply updates when segment end is detected."""
    path.closed_cost += path.open_cost
    is_sub = is_substitution(hyp_idx, ref_idx, path.last_hyp_idx, path.last_ref_idx)
    path.closed_cost += path.open_cost if is_sub else 0
    path.last_hyp_idx = hyp_idx
    path.last_ref_idx = ref_idx
    path.open_cost = 0


def end_insertion_segment(path: Path, hyp_idx: int, ref_idx: int) -> None:
    """End the current segment, if criteria for an insertion are met."""
    hyp_slice = slice(path.last_hyp_idx + 1, hyp_idx + 1)
    hyp_slice = translate_slice(hyp_slice, path.src.hyp_idx_map)
    ref_is_empty = ref_idx == path.last_ref_idx
    if hyp_slice is not None and ref_is_empty:
        path.end_indices += ((hyp_idx, ref_idx, path.open_cost),)
        _reset_segment_variables(path, hyp_idx, ref_idx)


def end_segment(path: Path) -> Union[None, "Path"]:
    """End the current segment, if criteria for an insertion, a substitution, or a match are met."""
    hyp_slice = slice(path.last_hyp_idx + 1, path.hyp_idx + 1)
    hyp_slice = translate_slice(hyp_slice, path.src.hyp_idx_map)
    ref_slice = slice(path.last_ref_idx + 1, path.ref_idx + 1)
    ref_slice = translate_slice(ref_slice, path.src.ref_idx_map)

    assert ref_slice is not None

    hyp_is_empty = path.hyp_idx == path.last_hyp_idx
    if hyp_is_empty:
        path.end_indices += ((path.hyp_idx, path.ref_idx, path.open_cost),)
    else:
        # TODO: Handle edge case where hyp has only covered delimiters.
        if hyp_slice is None:
            return None

        is_match_segment = path.open_cost == 0
        path.at_unambiguous_match_node = is_match_segment and path.index in path.src.unambiguous_matches
        path.end_indices += ((path.hyp_idx, path.ref_idx, path.open_cost),)

    # Update the path score and reset segments attributes.
    _reset_segment_variables(path, path.hyp_idx, path.ref_idx)
    return path


def transition_to_child_node(parent: Path, ref_step: int, hyp_step: int):
    """Transition to a child node by creating a new Path instance."""
    child = Path.__new__(Path)  # NOTE: Bypass __init__ for shallow copy.
    child.src = parent.src
    child.ref_idx = parent.ref_idx + ref_step
    child.hyp_idx = parent.hyp_idx + hyp_step
    child.last_hyp_idx = parent.last_hyp_idx
    child.last_ref_idx = parent.last_ref_idx
    child.closed_cost = parent.closed_cost
    child.open_cost = parent.open_cost
    child.at_unambiguous_match_node = False
    child.end_indices = parent.end_indices
    child.sort_id = parent.sort_id
    child.update_sort_id(ref_step + ref_step + hyp_step)
    return child


def is_substitution(hyp_idx: int, ref_idx: int, last_hyp_idx: int, last_ref_idx: int) -> int:
    """Get the substitution penalty given an index."""
    # NOTE: Since *_idx is guaranteed to be equal to or higher than last_*_idx, we only need to check for equality.
    if ref_idx == last_ref_idx or hyp_idx == last_hyp_idx:
        return False
    return True


def _cpp_path_to_py_path(cpp_path) -> Path:
    """Convert a C++ Path object to a Python Path object.

    Used for testing and debugging purposes only.

    """
    py_path = Path.__new__(Path)  # Bypass __init__
    py_path.src = cpp_path.src
    py_path.ref_idx = cpp_path.ref_idx
    py_path.hyp_idx = cpp_path.hyp_idx
    py_path.last_ref_idx = cpp_path.last_ref_idx
    py_path.last_hyp_idx = cpp_path.last_hyp_idx
    py_path.closed_cost = cpp_path.closed_cost
    py_path.open_cost = cpp_path.open_cost
    py_path.at_unambiguous_match_node = cpp_path.at_unambiguous_match_node
    py_path.end_indices = cpp_path.end_indices
    return py_path


# ============================================================
# MAIN BEAM SEARCH FUNCTION
# ============================================================


def error_align_beam_search(src: SubgraphMetadata, beam_size: int = 100) -> Path:
    """Perform beam search to align reference and hypothesis texts for a given source.

    Args:
        src (SubgraphMetadata): The source metadata for alignment.
        beam_size (int): The size of the beam for beam search. Defaults to 100

    """
    # Initialize the beam with a single path starting at the root node.
    start_path = Path(src)
    beam = [start_path]
    prune_map = dict()
    ended = []

    # Expand candidate paths until all have reached the terminal node.
    while len(beam) > 0:
        new_beam = {}

        # Expand each path in the current beam.
        for path in beam:
            if path.at_end:
                ended.append(path)
                continue

            # for new_path in cpp_children:
            for new_path in expand(path):
                new_path_cost = new_path.cost
                new_path_prune_id = new_path.prune_id
                if new_path_prune_id in prune_map:
                    if new_path_cost > prune_map[new_path_prune_id]:
                        continue
                prune_map[new_path_prune_id] = new_path_cost
                if new_path_prune_id not in new_beam or new_path_cost < new_beam[new_path_prune_id].cost:
                    new_beam[new_path_prune_id] = new_path

        # Update the beam with the newly expanded paths.
        new_beam = list(new_beam.values())
        new_beam.sort(key=lambda p: (p.norm_cost, p.sort_id))
        beam = new_beam[:beam_size]

        # Keep only the best path if, it matches the segment.
        if len(beam) > 0 and beam[0].at_unambiguous_match_node:
            beam = beam[:1]
            prune_map = dict()

    # Return the best path or its alignments.
    if len(ended) == 0:
        return []

    ended.sort(key=lambda p: (p.cost, p.sort_id))
    return ended[0]
