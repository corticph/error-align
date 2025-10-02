from error_align.utils import DELIMITERS, OP_TYPE_COMBO_MAP_INV, OpType


def _get_levenshtein_values(ref_token: str, hyp_token: str):
    """Compute the Levenshtein values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.

    """
    if hyp_token == ref_token:
        diag_cost = 0
    else:
        diag_cost = 1

    return 1, 1, diag_cost


def _get_error_align_values(ref_token: str, hyp_token: str):
    """Compute the error alignment values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.

    """
    if hyp_token == ref_token:
        diag_cost = 0
    elif hyp_token in DELIMITERS or ref_token in DELIMITERS:
        diag_cost = 3  # NOTE: Will never be chosen as insert + delete (= 2) is equivalent and cheaper.
    else:
        diag_cost = 2

    return 1, 1, diag_cost


def compute_distance_matrix(
    ref: str | list[str],
    hyp: str | list[str],
    score_func: callable,
    backtrace: bool = False,
    dtype: type = int,
):
    """Compute the edit distance score matrix between two sequences x (hyp) and y (ref)
    using only pure Python lists.

    Args:
        ref (str or list[str]): The reference sequence/transcript.
        hyp (str or list[str]): The hypothesis sequence/transcript.
        score_func (callable): A function that takes two tokens (ref_token, hyp_token) and returns
            a tuple of (deletion_cost, insertion_cost, diagonal_cost).
        backtrace (bool): Whether to compute the backtrace matrix.
        dtype (type): The type to store scores (int, float, etc.).

    Returns:
        list[list]: The score matrix.
        list[list]: The backtrace matrix, if backtrace=True.

    """
    hyp_dim, ref_dim = len(hyp) + 1, len(ref) + 1

    # Create empty score matrix of zeros and initialize first row and column.
    score_matrix = [[dtype(0) for _ in range(ref_dim)] for _ in range(hyp_dim)]
    for j in range(ref_dim):
        score_matrix[0][j] = dtype(j)
    for i in range(hyp_dim):
        score_matrix[i][0] = dtype(i)

    # Create backtrace matrix and operation combination map and initialize first row and column.
    # Each operation combination is dynamically assigned a unique integer.
    if backtrace:
        backtrace_matrix = [[0 for _ in range(ref_dim)] for _ in range(hyp_dim)]
        backtrace_matrix[0][0] = OP_TYPE_COMBO_MAP_INV[(OpType.MATCH,)]
        for j in range(1, ref_dim):
            backtrace_matrix[0][j] = OP_TYPE_COMBO_MAP_INV[(OpType.DELETE,)]
        for i in range(1, hyp_dim):
            backtrace_matrix[i][0] = OP_TYPE_COMBO_MAP_INV[(OpType.INSERT,)]

    # Fill in the score and backtrace matrix.
    for j in range(1, ref_dim):
        for i in range(1, hyp_dim):
            ins_cost, del_cost, diag_cost = score_func(ref[j - 1], hyp[i - 1])

            ins_val = score_matrix[i - 1][j] + ins_cost
            del_val = score_matrix[i][j - 1] + del_cost
            diag_val = score_matrix[i - 1][j - 1] + diag_cost
            new_val = min(ins_val, del_val, diag_val)
            score_matrix[i][j] = dtype(new_val)

            # Track possible operations (note that the order of operations matters).
            if backtrace:
                pos_ops = tuple()
                if diag_val == new_val and diag_cost == 0:
                    pos_ops += (OpType.MATCH,)
                if ins_val == new_val:
                    pos_ops += (OpType.INSERT,)
                if del_val == new_val:
                    pos_ops += (OpType.DELETE,)
                if diag_val == new_val and diag_cost > 0:
                    pos_ops += (OpType.SUBSTITUTE,)
                backtrace_matrix[i][j] = OP_TYPE_COMBO_MAP_INV[pos_ops]

    if backtrace:
        return score_matrix, backtrace_matrix
    return score_matrix


def compute_levenshtein_distance_matrix(ref: str | list[str], hyp: str | list[str], backtrace: bool = False):
    """Compute the Levenshtein distance matrix between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.

    """
    return compute_distance_matrix(ref, hyp, _get_levenshtein_values, backtrace)


def compute_error_align_distance_matrix(ref: str | list[str], hyp: str | list[str], backtrace: bool = False):
    """Compute the error alignment distance matrix between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.

    """
    return compute_distance_matrix(ref, hyp, _get_error_align_values, backtrace)
