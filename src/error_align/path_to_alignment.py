from types import SimpleNamespace
from typing import Union

from error_align.beam_search import Path
from error_align.graph_metadata import SubgraphMetadata
from error_align.utils import Alignment, OpType, translate_slice

# ============================================================
# GET ALIGNMENTS FROM SEGMENTATION INDICES
# ============================================================


def get_delete_alignment(
    start_ref_idx: int,
    end_ref_idx: int,
    subgraph_metadata: SubgraphMetadata,
) -> Alignment:
    """Get a DELETE alignment for a given reference slice."""
    ref_slice = slice(start_ref_idx, end_ref_idx)
    ref_slice = translate_slice(ref_slice, subgraph_metadata.ref_idx_map)
    return Alignment(
        op_type=OpType.DELETE,
        ref_slice=ref_slice,
        ref=subgraph_metadata.ref_raw[ref_slice],
    )


def get_insert_alignment(
    start_hyp_idx: int,
    end_hyp_idx: int,
    subgraph_metadata: SubgraphMetadata,
) -> Alignment:
    """Get an INSERT alignment for a given hypothesis slice."""
    hyp_slice = slice(start_hyp_idx, end_hyp_idx)
    hyp_slice = translate_slice(hyp_slice, subgraph_metadata.hyp_idx_map)
    return Alignment(
        op_type=OpType.INSERT,
        hyp_slice=hyp_slice,
        hyp=subgraph_metadata.hyp_raw[hyp_slice],
    )


def get_match_or_substitution_alignment(
    start_hyp_idx: int,
    end_hyp_idx: int,
    start_ref_idx: int,
    end_ref_idx: int,
    score: Union[int, float],
    subgraph_metadata: SubgraphMetadata,
) -> Alignment:
    """Get a MATCH or SUBSTITUTE alignment for given hypothesis and reference slices."""
    hyp_slice = slice(start_hyp_idx, end_hyp_idx)
    ref_slice = slice(start_ref_idx, end_ref_idx)
    hyp_slice = translate_slice(hyp_slice, subgraph_metadata.hyp_idx_map)
    ref_slice = translate_slice(ref_slice, subgraph_metadata.ref_idx_map)
    is_match_segment = score == 0
    op_type = OpType.MATCH if is_match_segment else OpType.SUBSTITUTE
    return Alignment(
        op_type=op_type,
        ref_slice=ref_slice,
        hyp_slice=hyp_slice,
        ref=subgraph_metadata.ref_raw[ref_slice],
        hyp=subgraph_metadata.hyp_raw[hyp_slice],
        left_compound=subgraph_metadata.hyp_idx_map[start_hyp_idx] >= 0,
        right_compound=subgraph_metadata.hyp_idx_map[end_hyp_idx - 1] >= 0,
    )


def get_alignments(path: Union[Path, SimpleNamespace]) -> list[Alignment]:
    """Get the alignments of the path.

    The input will be a SimpleNamespace if using the C++ beam search implementation.

    """

    subgraph_metadata = path.src
    segmentation_indices = path.end_indices

    # Compute alignments from the segment end indices.
    alignments = []
    start_hyp, start_ref = (0, 0)
    for end_hyp, end_ref, score in segmentation_indices:
        end_hyp, end_ref = end_hyp + 1, end_ref + 1

        if start_hyp == end_hyp:
            alignment = get_delete_alignment(
                start_ref,
                end_ref,
                subgraph_metadata,
            )
            alignments.append(alignment)
        elif start_ref == end_ref:
            alignment = get_insert_alignment(
                start_hyp,
                end_hyp,
                subgraph_metadata,
            )
            alignments.append(alignment)
        else:
            alignment = get_match_or_substitution_alignment(
                start_hyp,
                end_hyp,
                start_ref,
                end_ref,
                score,
                subgraph_metadata,
            )
            alignments.append(alignment)

        start_hyp, start_ref = end_hyp, end_ref

    return alignments
