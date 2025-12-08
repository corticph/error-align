#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <sstream>

namespace py = pybind11;

// --- Enums and Constants -----------------------------------------------------

enum OpType {
    MATCH = 0,
    INSERT = 1,
    DELETE = 2,
    SUBSTITUTE = 3
};

const std::unordered_set<std::string> DELIMITERS = {"<", ">"};

// --- Custom hash for std::vector<OpType> -------------------------------------

struct VectorHash {
    std::size_t operator()(const std::vector<OpType>& v) const noexcept {
        std::size_t h = 0;
        for (auto op : v) {
            h ^= std::hash<int>()(static_cast<int>(op)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// --- Hardcoded OP_TYPE_COMBO_MAP ---------------------------------------------

static const std::unordered_map<int, std::vector<OpType>> OP_TYPE_COMBO_MAP = {
    {0, {MATCH}},
    {1, {INSERT}},
    {2, {DELETE}},
    {3, {SUBSTITUTE}},
    {4, {MATCH, INSERT}},
    {5, {MATCH, DELETE}},
    {6, {MATCH, SUBSTITUTE}},
    {7, {INSERT, DELETE}},
    {8, {INSERT, SUBSTITUTE}},
    {9, {DELETE, SUBSTITUTE}},
    {10, {MATCH, INSERT, DELETE}},
    {11, {MATCH, INSERT, SUBSTITUTE}},
    {12, {MATCH, DELETE, SUBSTITUTE}},
    {13, {INSERT, DELETE, SUBSTITUTE}},
    {14, {MATCH, INSERT, DELETE, SUBSTITUTE}},
};

// Build inverse map: vector<OpType> → int
static std::unordered_map<std::vector<OpType>, int, VectorHash> build_inverse_combo_map() {
    std::unordered_map<std::vector<OpType>, int, VectorHash> inv;
    for (const auto& kv : OP_TYPE_COMBO_MAP)
        inv[kv.second] = kv.first;
    return inv;
}

static const std::unordered_map<std::vector<OpType>, int, VectorHash> OP_TYPE_COMBO_MAP_INV =
    build_inverse_combo_map();

int op_code(const std::vector<OpType>& ops) {
    auto it = OP_TYPE_COMBO_MAP_INV.find(ops);
    if (it != OP_TYPE_COMBO_MAP_INV.end())
        return it->second;
    throw std::runtime_error("Invalid op combination");
}

// --- Tokenizer for string input ----------------------------------------------

// Split UTF-8 string into individual Unicode code points
std::vector<std::string> utf8_split(const std::string& text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_len = 0;

        if (c < 0x80)               // 1-byte ASCII
            char_len = 1;
        else if ((c >> 5) == 0x6)   // 110xxxxx -> 2 bytes
            char_len = 2;
        else if ((c >> 4) == 0xE)   // 1110xxxx -> 3 bytes
            char_len = 3;
        else if ((c >> 3) == 0x1E)  // 11110xxx -> 4 bytes
            char_len = 4;
        else
            throw std::runtime_error("Invalid UTF-8 sequence");

        if (i + char_len > text.size())
            throw std::runtime_error("Truncated UTF-8 sequence");

        chars.emplace_back(text.substr(i, char_len));
        i += char_len;
    }
    return chars;
}

// --- Cost Functions ----------------------------------------------------------

inline std::tuple<int, int, int> get_levenshtein_values(const std::string& ref_token,
                                                        const std::string& hyp_token) {
    int diag_cost = (ref_token == hyp_token) ? 0 : 1;
    return {1, 1, diag_cost};
}

inline std::tuple<int, int, int> get_error_align_values(const std::string& ref_token,
                                                        const std::string& hyp_token) {
    int diag_cost;
    if (ref_token == hyp_token) {
        diag_cost = 0;
    } else if (DELIMITERS.count(ref_token) || DELIMITERS.count(hyp_token)) {
        diag_cost = 3;  // Will never be chosen (insert+delete=2)
    } else {
        diag_cost = 2;
    }
    return {1, 1, diag_cost};
}

// --- Core Distance Matrix Computation ----------------------------------------

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
compute_distance_matrix(const std::vector<std::string>& ref,
                        const std::vector<std::string>& hyp,
                        const std::string& mode,
                        bool backtrace) {
    const int ref_dim = static_cast<int>(ref.size()) + 1;
    const int hyp_dim = static_cast<int>(hyp.size()) + 1;

    std::vector<std::vector<int>> score_matrix(hyp_dim, std::vector<int>(ref_dim, 0));
    std::vector<std::vector<int>> backtrace_matrix;

    for (int j = 0; j < ref_dim; ++j)
        score_matrix[0][j] = j;
    for (int i = 0; i < hyp_dim; ++i)
        score_matrix[i][0] = i;

    if (backtrace) {
        backtrace_matrix.assign(hyp_dim, std::vector<int>(ref_dim, 0));
        backtrace_matrix[0][0] = op_code({MATCH});
        for (int j = 1; j < ref_dim; ++j)
            backtrace_matrix[0][j] = op_code({DELETE});
        for (int i = 1; i < hyp_dim; ++i)
            backtrace_matrix[i][0] = op_code({INSERT});
    }

    for (int j = 1; j < ref_dim; ++j) {
        for (int i = 1; i < hyp_dim; ++i) {
            int ins_cost, del_cost, diag_cost;
            if (mode == "levenshtein")
                std::tie(ins_cost, del_cost, diag_cost) = get_levenshtein_values(ref[j - 1], hyp[i - 1]);
            else
                std::tie(ins_cost, del_cost, diag_cost) = get_error_align_values(ref[j - 1], hyp[i - 1]);

            int ins_val = score_matrix[i - 1][j] + ins_cost;
            int del_val = score_matrix[i][j - 1] + del_cost;
            int diag_val = score_matrix[i - 1][j - 1] + diag_cost;
            int new_val = std::min({ins_val, del_val, diag_val});
            score_matrix[i][j] = new_val;

            if (backtrace) {
                std::vector<OpType> pos_ops;
                if (diag_val == new_val && diag_cost <= 0)
                    pos_ops.push_back(MATCH);
                if (ins_val == new_val)
                    pos_ops.push_back(INSERT);
                if (del_val == new_val)
                    pos_ops.push_back(DELETE);
                if (diag_val == new_val && diag_cost > 0)
                    pos_ops.push_back(SUBSTITUTE);
                backtrace_matrix[i][j] = op_code(pos_ops);
            }
        }
    }

    return {score_matrix, backtrace_matrix};
}

// --- Unified entrypoints (handle str or list[str]) ---------------------------

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
compute_levenshtein_distance_matrix(py::object ref_obj, py::object hyp_obj, bool backtrace) {
    std::vector<std::string> ref, hyp;
    if (py::isinstance<py::str>(ref_obj))
        ref = utf8_split(ref_obj.cast<std::string>());
    else
        ref = ref_obj.cast<std::vector<std::string>>();
    if (py::isinstance<py::str>(hyp_obj))
        hyp = utf8_split(hyp_obj.cast<std::string>());
    else
        hyp = hyp_obj.cast<std::vector<std::string>>();

    return compute_distance_matrix(ref, hyp, "levenshtein", backtrace);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
compute_error_align_distance_matrix(py::object ref_obj, py::object hyp_obj, bool backtrace) {
    std::vector<std::string> ref, hyp;
    if (py::isinstance<py::str>(ref_obj))
        ref = utf8_split(ref_obj.cast<std::string>());
    else
        ref = ref_obj.cast<std::vector<std::string>>();
    if (py::isinstance<py::str>(hyp_obj))
        hyp = utf8_split(hyp_obj.cast<std::string>());
    else
        hyp = hyp_obj.cast<std::vector<std::string>>();

    return compute_distance_matrix(ref, hyp, "error_align", backtrace);
}

// --- Expose OP_TYPE_COMBO_MAP_INV -------------------------------------------

py::dict get_op_type_combo_map_inv() {
    py::dict d;
    for (const auto& kv : OP_TYPE_COMBO_MAP_INV) {
        py::tuple key_tuple(kv.first.size());
        for (size_t i = 0; i < kv.first.size(); ++i)
            key_tuple[i] = static_cast<int>(kv.first[i]);
        d[key_tuple] = kv.second;
    }
    return d;
}

// ----------------- PYBIND11 MODULE -----------------
PYBIND11_MODULE(_cpp_edit_distance, m) {
    m.doc() = "C++ accelerated Levenshtein and ErrorAlign distance matrix computations";

    m.def("compute_levenshtein_distance_matrix", &compute_levenshtein_distance_matrix,
          py::arg("ref"), py::arg("hyp"), py::arg("backtrace") = false);

    m.def("compute_error_align_distance_matrix", &compute_error_align_distance_matrix,
          py::arg("ref"), py::arg("hyp"), py::arg("backtrace") = false);

    m.def("get_op_type_combo_map_inv", &get_op_type_combo_map_inv,
          "Return the mapping of operation-type combinations to integer codes");
}
