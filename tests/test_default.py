from error_align._cpp_beam_search import error_align_beam_search as cpp_error_align_beam_search
from typeguard import suppress_type_checks

from error_align import error_align
from error_align.backtrace_graph import BacktraceGraph
from error_align.beam_search import error_align_beam_search as python_error_align_beam_search
from error_align.edit_distance import compute_error_align_distance_matrix, compute_levenshtein_distance_matrix
from error_align.error_align import prepare_graph_metadata
from error_align.graph_metadata import SubgraphMetadata
from error_align.utils import OpType, categorize_char


def test_error_align() -> None:
    """Test error alignment for an example including all substitution types."""

    ref = "This is a substitution test deleted."
    hyp = "Inserted this is a contribution test."

    alignments = error_align(ref, hyp)
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


def test_beam_search_cpp_vs_python() -> None:
    """Test that the C++ and Python beam search implementations produce the same results."""

    ref = "This is a substitution test deleted."
    hyp = "Inserted this is a contribution test."

    graph_metadata = prepare_graph_metadata(ref, hyp)
    subgraph_metadata = SubgraphMetadata(
        ref_raw=graph_metadata.ref_raw,
        hyp_raw=graph_metadata.hyp_raw,
        ref_token_matches=graph_metadata.ref_token_matches,
        hyp_token_matches=graph_metadata.hyp_token_matches,
        ref_norm=graph_metadata.ref_norm,
        hyp_norm=graph_metadata.hyp_norm,
    )

    path_cpp = cpp_error_align_beam_search(src=subgraph_metadata)
    path_python = python_error_align_beam_search(src=subgraph_metadata)

    assert path_cpp.open_cost == path_python.open_cost
    assert path_cpp.closed_cost == path_python.closed_cost
    assert path_cpp.cost == path_python.cost
    assert path_cpp.norm_cost == path_python.norm_cost
    assert path_cpp.sort_id == path_python.sort_id


def test_error_align_identical() -> None:
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
    assert repr(substitute_alignment) == 'Alignment(SUBSTITUTE: "substitution"- -> "substitution")'

    # Test MATCH operation without compound markers
    match_alignment = error_align("test", "test")[0]
    assert repr(match_alignment) == 'Alignment(MATCH: "test" == "test")'


@suppress_type_checks
def test_input_type_checks() -> None:
    """Test input type checks for ErrorAlign class."""

    try:
        _ = error_align(ref=123, hyp="valid")  # type: ignore
    except TypeError as e:
        assert str(e) == "Reference sequence must be a string."

    try:
        _ = error_align(ref="valid", hyp=456)  # type: ignore
    except TypeError as e:
        assert str(e) == "Hypothesis sequence must be a string."


def test_backtrace_graph() -> None:
    """Test backtrace graph generation."""

    ref = "This is a test."
    hyp = "This is a pest."

    # Create ErrorAlign instance and generate backtrace graph.
    graph_metadata = prepare_graph_metadata(ref, hyp)
    subgraph_metadata = SubgraphMetadata(
        ref_raw=graph_metadata.ref_raw,
        hyp_raw=graph_metadata.hyp_raw,
        ref_token_matches=graph_metadata.ref_token_matches,
        hyp_token_matches=graph_metadata.hyp_token_matches,
        ref_norm=graph_metadata.ref_norm,
        hyp_norm=graph_metadata.hyp_norm,
    )
    _, backtrace_matrix = compute_error_align_distance_matrix(
        subgraph_metadata.ref,
        subgraph_metadata.hyp,
        backtrace=True,
    )
    graph = BacktraceGraph(backtrace_matrix)

    # Check basic properties of the graph.
    assert isinstance(graph.get_path(), list)
    assert isinstance(graph.get_path(sample=True), list)
    for index in graph._iter_topological_order():
        assert isinstance(index, tuple)


def test_levenshtein_distance_matrix() -> None:
    """Test Levenshtein distance matrix computation."""

    ref = "kitten"
    hyp = "sitting"

    distance_matrix = compute_levenshtein_distance_matrix(ref, hyp)

    assert distance_matrix[-1][-1] == 3
