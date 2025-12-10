# error_align/__init__.py  (or error_align/core.py)
from importlib import import_module

USING_CPP = False

# Try native extension first
try:
    _core_edit_distance = import_module("error_align._cpp_edit_distance")
    _core_beam_search = import_module("error_align._cpp_beam_search")
    USING_CPP = True
except ModuleNotFoundError:
    # Fall back to pure Python
    _core_edit_distance = import_module("error_align.edit_distance")
    _core_beam_search = import_module("error_align.beam_search")

# Re-export unified interface
compute_error_align_distance_matrix = _core_edit_distance.compute_error_align_distance_matrix
compute_levenshtein_distance_matrix = _core_edit_distance.compute_levenshtein_distance_matrix
error_align_beam_search = _core_beam_search.error_align_beam_search

__all__ = [
    "compute_error_align_distance_matrix",
    "compute_levenshtein_distance_matrix",
    "error_align_beam_search",
    "USING_CPP",
]
