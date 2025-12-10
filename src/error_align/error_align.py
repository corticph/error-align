from error_align.backtrace_graph import BacktraceGraph
from error_align.core import compute_levenshtein_distance_matrix, error_align_beam_search
from error_align.graph_metadata import GraphMetadata, SubgraphMetadata
from error_align.path_to_alignment import get_alignments
from error_align.utils import (
    Alignment,
    OpType,
    basic_normalizer,
    basic_tokenizer,
    ensure_length_preservation,
    unpack_regex_match,
)


def error_align(
    ref: str,
    hyp: str,
    tokenizer: callable = basic_tokenizer,
    normalizer: callable = basic_normalizer,
    beam_size: int = 100,
    word_level_pass: bool = True,
):
    """Run error alignment between reference and hypothesis texts.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
        normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        beam_size (int): The beam size for beam search alignment.
        word_level_pass (bool): Whether to perform a word-level alignment pass to identify unambiguous matches.

    """
    graph_metadata = prepare_graph_metadata(
        ref=ref,
        hyp=hyp,
        tokenizer=tokenizer,
        normalizer=normalizer,
    )

    if graph_metadata.ref_norm == graph_metadata.hyp_norm:
        return align_identical_inputs(graph_metadata)
    elif not word_level_pass:
        return align_beam_search(graph_metadata, beam_size=beam_size)
    else:
        return align_with_word_level_pass(graph_metadata, beam_size=beam_size)


def prepare_graph_metadata(
    ref: str,
    hyp: str,
    tokenizer: callable = basic_tokenizer,
    normalizer: callable = basic_normalizer,
) -> GraphMetadata:
    """Prepare graph metadata from reference and hypothesis texts."""

    if not isinstance(ref, str):
        raise TypeError("Reference sequence must be a string.")
    if not isinstance(hyp, str):
        raise TypeError("Hypothesis sequence must be a string.")

    # Inclusive tokenization: Track the token position in the original text.
    tokenizer = unpack_regex_match(tokenizer)
    ref_token_matches = tokenizer(ref)
    hyp_token_matches = tokenizer(hyp)

    # Length-preserving normalization: Ensure that the normalizer preserves token length.
    normalizer = ensure_length_preservation(normalizer)
    ref_norm = [normalizer(r) for r, _ in ref_token_matches]
    hyp_norm = [normalizer(h) for h, _ in hyp_token_matches]

    # Prepare input data dataclass for ease of passing around
    return GraphMetadata(
        ref_raw=ref,
        hyp_raw=hyp,
        ref_token_matches=ref_token_matches,
        hyp_token_matches=hyp_token_matches,
        ref_norm=ref_norm,
        hyp_norm=hyp_norm,
    )


def align_identical_inputs(graph_metadata: GraphMetadata) -> list[Alignment]:
    """Return alignments for identical reference and hypothesis pairs."""
    alignments = []
    for i in range(len(graph_metadata.ref_token_matches)):
        alignment = get_match_alignment_from_token_indices(
            graph_metadata=graph_metadata,
            ref_index=i,
            hyp_index=i,
        )
        alignments.append(alignment)
    return alignments


def align_beam_search(
    graph_metadata: GraphMetadata,
    beam_size: int,
    ref_start: int | None = None,
    ref_end: int | None = None,
    hyp_start: int | None = None,
    hyp_end: int | None = None,
) -> list[Alignment]:
    """Perform beam search alignment for the given source."""
    src = SubgraphMetadata(
        ref_raw=graph_metadata.ref_raw,
        hyp_raw=graph_metadata.hyp_raw,
        ref_token_matches=graph_metadata.ref_token_matches[ref_start:ref_end],
        hyp_token_matches=graph_metadata.hyp_token_matches[hyp_start:hyp_end],
        ref_norm=graph_metadata.ref_norm[ref_start:ref_end],
        hyp_norm=graph_metadata.hyp_norm[hyp_start:hyp_end],
    )
    path = error_align_beam_search(src=src, beam_size=beam_size)
    return get_alignments(path)


def align_with_word_level_pass(
    graph_metadata: GraphMetadata,
    beam_size: int,
) -> list[Alignment]:
    """Perform a word-level alignment pass to identify unambiguous matches."""
    # Extract the word-level backtrace graph and identify unambiguous matches.
    _, backtrace_matrix = compute_levenshtein_distance_matrix(
        graph_metadata.ref_norm,
        graph_metadata.hyp_norm,
        backtrace=True,
    )
    backtrace_graph = BacktraceGraph(backtrace_matrix)
    match_indices = backtrace_graph.get_unambiguous_node_matches()
    # NOTE: We always add an artificial terminal match node to simplify subspan extraction.
    match_indices = match_indices + [(len(graph_metadata.hyp_norm), len(graph_metadata.ref_norm))]

    # Iterate over the unambiguous matches to extract subspans (i.e., the span of words between two matches).
    hyp_start, ref_start = (0, 0)
    alignments = []
    end_index = len(match_indices) - 1
    for i, (hyp_end, ref_end) in enumerate(match_indices):
        ref_is_empty = ref_start == ref_end
        hyp_is_empty = hyp_start == hyp_end

        # NOTE: Subspans where ref xor hyp is empty are guaranteed to be all INSERT or DELETE ops.
        if not ref_is_empty and not hyp_is_empty:
            alignments.extend(
                align_beam_search(
                    graph_metadata=graph_metadata,
                    beam_size=beam_size,
                    ref_start=ref_start,
                    ref_end=ref_end,
                    hyp_start=hyp_start,
                    hyp_end=hyp_end,
                )
            )
        elif ref_is_empty and not hyp_is_empty:
            for token_idx in range(hyp_start, hyp_end):
                alignments.append(
                    get_insert_alignment_from_token_index(
                        graph_metadata=graph_metadata,
                        hyp_index=token_idx,
                    )
                )
        elif hyp_is_empty and not ref_is_empty:
            for token_idx in range(ref_start, ref_end):
                alignments.append(
                    get_delete_alignment_from_token_index(
                        graph_metadata=graph_metadata,
                        ref_index=token_idx,
                    )
                )
        if i < end_index:
            alignments.append(
                get_match_alignment_from_token_indices(
                    graph_metadata=graph_metadata,
                    ref_index=ref_end,
                    hyp_index=hyp_end,
                )
            )
        ref_start, hyp_start = (ref_end + 1, hyp_end + 1)

    return alignments


def get_match_alignment_from_token_indices(
    graph_metadata: GraphMetadata,
    ref_index: int,
    hyp_index: int,
) -> Alignment:
    """Get a MATCH alignment for the given token indices."""
    ref_slice = slice(*graph_metadata.ref_token_matches[ref_index][1])
    hyp_slice = slice(*graph_metadata.hyp_token_matches[hyp_index][1])
    return Alignment(
        op_type=OpType.MATCH,
        ref_slice=ref_slice,
        hyp_slice=hyp_slice,
        ref=graph_metadata.ref_raw[ref_slice],
        hyp=graph_metadata.hyp_raw[hyp_slice],
    )


def get_insert_alignment_from_token_index(
    graph_metadata: GraphMetadata,
    hyp_index: int,
) -> Alignment:
    """Get an INSERT alignment for the given token index."""
    slice_ = slice(*graph_metadata.hyp_token_matches[hyp_index][1])
    token = graph_metadata.hyp_raw[slice_]
    return Alignment(
        op_type=OpType.INSERT,
        hyp_slice=slice_,
        hyp=token,
    )


def get_delete_alignment_from_token_index(
    graph_metadata: GraphMetadata,
    ref_index: int,
) -> Alignment:
    """Get a DELETE alignment for the given token index."""
    slice_ = slice(*graph_metadata.ref_token_matches[ref_index][1])
    token = graph_metadata.ref_raw[slice_]
    return Alignment(
        op_type=OpType.DELETE,
        ref_slice=slice_,
        ref=token,
    )
