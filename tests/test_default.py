from typeguard import suppress_type_checks

from error_align import ErrorAlign, error_align
from error_align.edit_distance import compute_levenshtein_distance_matrix
from error_align.utils import OpType, categorize_char


def test_error_align() -> None:
    """Test error alignment for an example including all substitution types."""

    ref = "This is a substitution test deleted."
    hyp = "Inserted this is a contribution test."

    alignments = error_align(ref, hyp, pbar=True)
    expected_ops = [
        OpType.INSERT,  # Inserted
        OpType.MATCH,  # This
        OpType.MATCH,  # is
        OpType.MATCH,  # a
        OpType.SUBSTITUTE,  # contribution -> substitution
        OpType.MATCH,  # test
        OpType.DELETE,  # deleted
    ]

    for op, alignment in zip(expected_ops, alignments, strict=True):
        assert alignment.op_type == op


def test_error_align_full_match() -> None:
    """Test error alignment for full match."""

    ref = "This is a test."
    hyp = "This is a test."

    alignments = error_align(ref, hyp)

    for alignment in alignments:
        assert alignment.op_type == OpType.MATCH


def test_categorize_char() -> None:
    """Test character categorization."""

    assert categorize_char("<") == 0  # Delimiters
    assert categorize_char("b") == 1  # Consonants
    assert categorize_char("a") == 2  # Vowels
    assert categorize_char("'") == 3  # Unvoiced characters


def test_representations() -> None:
    """Test the string representation of Alignment objects."""

    # Test DELETE operation
    delete_alignment = error_align("deleted", "")[0]
    assert repr(delete_alignment) == 'Alignment(DELETE: "deleted")'

    # Test INSERT operation with compound markers
    insert_alignment = error_align("", "inserted")[0]
    assert repr(insert_alignment) == 'Alignment(INSERT: "inserted")'

    # Test SUBSTITUTE operation with compound markers
    substitute_alignment = error_align("substitution", "substitutiontesting")[0]
    assert substitute_alignment.left_compound is False
    assert substitute_alignment.right_compound is True
    assert repr(substitute_alignment) == 'Alignment(SUBSTITUTE: "substitution" -> "substitution"-)'

    # Test MATCH operation without compound markers
    match_alignment = error_align("test", "test")[0]
    assert repr(match_alignment) == 'Alignment(MATCH: "test" == "test")'

    # Test ErrorAlign class representation
    ea = ErrorAlign(ref="test", hyp="pest")
    assert repr(ea) == 'ErrorAlign(ref="test", hyp="pest")'

    # Test Path class representation
    path = ea.align(beam_size=10, return_path=True)
    assert repr(path) == f"Path(({path.ref_idx}, {path.hyp_idx}), score={path.cost})"


@suppress_type_checks
def test_input_type_checks() -> None:
    """Test input type checks for ErrorAlign class."""

    try:
        _ = ErrorAlign(ref=123, hyp="valid")  # type: ignore
    except TypeError as e:
        assert str(e) == "Reference sequence must be a string."

    try:
        _ = ErrorAlign(ref="valid", hyp=456)  # type: ignore
    except TypeError as e:
        assert str(e) == "Hypothesis sequence must be a string."


def test_backtrace_graph() -> None:
    """Test backtrace graph generation."""

    ref = "This is a test."
    hyp = "This is a pest."

    # Create ErrorAlign instance and generate backtrace graph.
    ea = ErrorAlign(ref, hyp)
    ea.align(beam_size=10)
    graph = ea._backtrace_graph

    # Check basic properties of the graph.
    assert isinstance(graph.get_path(), list)
    assert isinstance(graph.get_path(sample=True), list)
    assert graph.number_of_paths == 3
    for index in graph._iter_topological_order():
        assert isinstance(index, tuple)

    # Check specific node properties.
    node = graph.get_node(2, 2)
    assert node.number_of_ingoing_paths_via(OpType.MATCH) == 3
    assert node.number_of_outgoing_paths_via(OpType.MATCH) == 3
    assert node.number_of_ingoing_paths_via(OpType.INSERT) == 0
    assert node.number_of_outgoing_paths_via(OpType.INSERT) == 0


def test_levenshtein_distance_matrix() -> None:
    """Test Levenshtein distance matrix computation."""

    ref = "kitten"
    hyp = "sitting"

    distance_matrix = compute_levenshtein_distance_matrix(ref, hyp)

    assert distance_matrix[-1][-1] == 3  # The Levenshtein distance between "kitten" and "sitting" is 3
