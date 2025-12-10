#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <limits>
#include <codecvt>
#include <locale>

namespace py = pybind11;

// ---------- Helpers to read Python SubgraphMetadata ----------
std::u32string utf8_to_u32(const std::string& s) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(s);
}

static inline uint64_t pair_key(int hyp, int ref) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(hyp)) << 32) |
           static_cast<uint32_t>(ref);
}

struct SrcView {
    py::object src_obj;
    std::u32string ref;
    std::u32string hyp;
    int ref_max_idx;
    int hyp_max_idx;
    std::vector<int> ref_char_types;
    std::vector<int> hyp_char_types;
    std::vector<int> ref_idx_map;
    std::vector<int> hyp_idx_map;

    // Added optimizations
    std::unordered_set<uint64_t> backtrace_keys;
    std::unordered_set<uint64_t> unambiguous_keys;
    std::vector<int> ref_nonneg_psum;
    std::vector<int> hyp_nonneg_psum;
};

static SrcView make_src_view(const py::object& src) {
    SrcView v;
    v.src_obj = src;
    v.ref = utf8_to_u32(py::cast<std::string>(src.attr("ref")));
    v.hyp = utf8_to_u32(py::cast<std::string>(src.attr("hyp")));
    v.ref_max_idx = py::cast<int>(src.attr("ref_max_idx"));
    v.hyp_max_idx = py::cast<int>(src.attr("hyp_max_idx"));
    v.ref_char_types = py::cast<std::vector<int>>(src.attr("ref_char_types"));
    v.hyp_char_types = py::cast<std::vector<int>>(src.attr("hyp_char_types"));
    v.ref_idx_map = py::cast<std::vector<int>>(src.attr("ref_idx_map"));
    v.hyp_idx_map = py::cast<std::vector<int>>(src.attr("hyp_idx_map"));

    // Convert Python sets to C++ unordered_set for O(1) membership checks
    py::object backtrace_set = src.attr("backtrace_node_set");
    for (auto item : py::set(backtrace_set)) {
        auto pr = item.cast<std::pair<int, int>>();
        v.backtrace_keys.insert(pair_key(pr.first, pr.second));
    }

    py::object unambiguous_set = src.attr("unambiguous_matches");
    for (auto item : py::set(unambiguous_set)) {
        auto pr = item.cast<std::pair<int, int>>();
        v.unambiguous_keys.insert(pair_key(pr.first, pr.second));
    }

    // Build prefix sums for O(1) slice checks
    auto build_psum = [](const std::vector<int>& a) {
        std::vector<int> ps(a.size() + 1, 0);
        for (size_t i = 0; i < a.size(); ++i)
            ps[i + 1] = ps[i] + (a[i] >= 0);
        return ps;
    };
    v.ref_nonneg_psum = build_psum(v.ref_idx_map);
    v.hyp_nonneg_psum = build_psum(v.hyp_idx_map);

    // Sanity check
    if (v.ref_max_idx != static_cast<int>(v.ref.size()) - 1)
        throw std::runtime_error("Ref length mismatch: expected ref_max_idx = ref.size() - 1");
    if (v.hyp_max_idx != static_cast<int>(v.hyp.size()) - 1)
        throw std::runtime_error("Hyp length mismatch: expected hyp_max_idx = hyp.size() - 1");

    return v;
}

// Optimized set membership checks
static inline bool in_backtrace(const SrcView* s, int hyp, int ref) {
    return s->backtrace_keys.find(pair_key(hyp, ref)) != s->backtrace_keys.end();
}

static inline bool in_unambiguous(const SrcView* s, int hyp, int ref) {
    return s->unambiguous_keys.find(pair_key(hyp, ref)) != s->unambiguous_keys.end();
}

// ---------- Path and rest of beam search logic (unchanged) ----------
// ... (Keep your existing Path struct, expand_paths, to_python_path_like, and beam search loop exactly as they are)


// ---------- Path (C++ analog) ----------
struct Path {
    const SrcView* src;  // non-owning
    int ref_idx = -1;
    int hyp_idx = -1;
    int last_ref_idx = -1;
    int last_hyp_idx = -1;
    double closed_cost = 0.0;
    double open_cost = 0.0;
    bool at_unambiguous_match_node = false;
    std::vector<std::tuple<int,int,double>> end_indices;
    std::uint64_t sort_id = 0;

    inline bool at_end() const {
        return hyp_idx == src->hyp_max_idx && ref_idx == src->ref_max_idx;
    }

    inline std::pair<int,int> index() const { return {hyp_idx, ref_idx}; }

    inline static bool is_substitution(int hyp_i, int ref_i, int last_hyp_i, int last_ref_i) {
        // Same logic as Python: substitution if both advanced (neither equals the last)
        if (ref_i == last_ref_i || hyp_i == last_hyp_i) return false;
        return true;
    }

    inline double cost() const {
        bool is_sub = is_substitution(hyp_idx, ref_idx, last_hyp_idx, last_ref_idx);
        return closed_cost + open_cost + (is_sub ? open_cost : 0.0);
    }

    inline double norm_cost() const {
        double c = cost();
        if (c == 0.0) return 0.0;
        // +3 to avoid zero division; root=(-1,-1)
        return c / (ref_idx + hyp_idx + 3.0);
    }

    inline std::size_t prune_id() const {
        // Reproduce Python tuple hash style enough for pruning purposes.
        // Use a simple mix of the four ints.
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](long long x){
            for (int i=0;i<8;i++) {
                h ^= (std::size_t)((x >> (i*8)) & 0xff);
                h *= 1099511628211ull;
            }
        };
        mix(hyp_idx);
        mix(ref_idx);
        mix(last_hyp_idx);
        mix(last_ref_idx);
        return h;
    }
};

// ---------- Small utilities ----------
static inline bool in_set_of_pairs(const py::object& pyset, int hyp, int ref) {
    // Build a temporary tuple (hyp, ref) and use Python "in"
    py::tuple t = py::make_tuple(hyp, ref);
    return py::bool_(pyset.attr("__contains__")(t));
}

// O(1) prefix-sum slice check
static inline bool has_valid_slice_any_nonneg_psum(const std::vector<int>& psum,
                                                   int start_inclusive, int end_exclusive) {
    if (start_inclusive < 0) start_inclusive = 0;
    if (end_exclusive > (int)psum.size() - 1) end_exclusive = (int)psum.size() - 1;
    if (end_exclusive <= start_inclusive) return false;
    return (psum[end_exclusive] - psum[start_inclusive]) > 0;
}

static inline void reset_segment_variables(Path& p, int hyp_idx, int ref_idx) {
    p.closed_cost += p.open_cost;
    bool is_sub = Path::is_substitution(hyp_idx, ref_idx, p.last_hyp_idx, p.last_ref_idx);
    if (is_sub) p.closed_cost += p.open_cost;
    p.last_hyp_idx = hyp_idx;
    p.last_ref_idx = ref_idx;
    p.open_cost = 0.0;
}

static inline void end_insertion_segment(Path& p, int hyp_idx, int ref_idx) {
    // hyp slice = [last_hyp_idx+1, hyp_idx+1)
    bool hyp_slice_ok = has_valid_slice_any_nonneg_psum(p.src->hyp_nonneg_psum, p.last_hyp_idx + 1, hyp_idx + 1);
    bool ref_is_empty = (ref_idx == p.last_ref_idx);
    if (hyp_slice_ok && ref_is_empty) {
        p.end_indices.emplace_back(hyp_idx, ref_idx, p.open_cost);
        reset_segment_variables(p, hyp_idx, ref_idx);
    }
}

static inline bool end_segment(Path& p) {
    // ref_slice must be not None in Python => here, require some nonneg index in [last_ref+1, ref+1)
    bool ref_slice_ok = has_valid_slice_any_nonneg_psum(p.src->ref_nonneg_psum, p.last_ref_idx + 1, p.ref_idx + 1);
    if (!ref_slice_ok) {
        // Python used assert; we’ll keep behavior close by returning false (drop path)
        return false;
    }

    // hyp side:
    bool hyp_is_empty = (p.hyp_idx == p.last_hyp_idx);
    if (hyp_is_empty) {
        p.end_indices.emplace_back(p.hyp_idx, p.ref_idx, p.open_cost);
    } else {
        bool hyp_slice_ok = has_valid_slice_any_nonneg_psum(p.src->hyp_nonneg_psum, p.last_hyp_idx + 1, p.hyp_idx + 1);
        if (!hyp_slice_ok) {
            return false; // equivalent to returning None in Python
        }
        bool is_match_segment = (p.open_cost == 0.0);
        p.at_unambiguous_match_node =
            is_match_segment && in_unambiguous(p.src, p.hyp_idx, p.ref_idx);
        p.end_indices.emplace_back(p.hyp_idx, p.ref_idx, p.open_cost);
    }

    reset_segment_variables(p, p.hyp_idx, p.ref_idx);
    return true;
}

static const uint64_t B = 146527ULL;

uint64_t update_hash(uint64_t h, uint64_t t) {
    return h * B + t;   // wraparound = correct
}

static inline Path transition_to_child_node(const Path& parent, int ref_step, int hyp_step) {
    Path child = parent; // shallow copy (we’ll adjust fields)
    child.ref_idx = parent.ref_idx + ref_step;
    child.hyp_idx = parent.hyp_idx + hyp_step;
    child.at_unambiguous_match_node = false;
    // end_indices is copied (like Python’s tuple carry-forward)
    int transition_value = ref_step * 2 + hyp_step;
    child.sort_id = update_hash(parent.sort_id, transition_value);
    return child;
}

// returns: 0=skip, 1=ok, but no end-seg, 2=ok and child ended-seg (already handled)
static int add_substitution_or_match(const Path& parent, Path& out_child) {
    if (parent.ref_idx >= parent.src->ref_max_idx || parent.hyp_idx >= parent.src->hyp_max_idx) {
        return 0;
    }
    Path child = transition_to_child_node(parent, 1, 1);

    bool is_match = (parent.src->ref[child.ref_idx] == parent.src->hyp[child.hyp_idx]);
    if (!is_match) {
        bool ref_is_delim = (parent.src->ref_char_types[child.ref_idx] == 0);
        bool hyp_is_delim = (parent.src->hyp_char_types[child.hyp_idx] == 0);
        if (ref_is_delim || hyp_is_delim) {
            return 0;
        }
    }

    // end-of-segment criteria (insertion segment)
    if (parent.src->ref[child.ref_idx] == '<') {
        end_insertion_segment(child, parent.hyp_idx, parent.ref_idx);
    }

    if (!is_match) {
        bool is_backtrace = in_backtrace(parent.src, parent.hyp_idx, parent.ref_idx);
        bool letter_type_match = (parent.src->ref_char_types[child.ref_idx] == parent.src->hyp_char_types[child.hyp_idx]);
        child.open_cost += letter_type_match ? 2.0 : 3.0;
        child.open_cost += is_backtrace ? 0.0 : 1.0;
    }

    if (child.src->ref[child.ref_idx] == '>') {
        if (!end_segment(child)) return 0; // drop
    }

    out_child = std::move(child);
    return 1;
}

static int add_insert(const Path& parent, Path& out_child) {
    if (parent.ref_idx >= parent.src->ref_max_idx) return 0;

    Path child = transition_to_child_node(parent, 1, 0);

    if (parent.src->ref[child.ref_idx] == '<') {
        end_insertion_segment(child, parent.hyp_idx, parent.ref_idx);
    }

    bool is_backtrace = in_backtrace(parent.src, parent.hyp_idx, parent.ref_idx);
    bool is_delim = (parent.src->ref_char_types[child.ref_idx] == 0);
    child.open_cost += is_delim ? 1.0 : 2.0;
    child.open_cost += (is_backtrace || is_delim) ? 0.0 : 1.0;

    if (child.src->ref[child.ref_idx] == '>') {
        if (!end_segment(child)) return 0; // drop
    }

    out_child = std::move(child);
    return 1;
}

static int add_delete(const Path& parent, Path& out_child) {
    if (parent.hyp_idx >= parent.src->hyp_max_idx) return 0;

    Path child = transition_to_child_node(parent, 0, 1);

    bool is_backtrace = in_backtrace(parent.src, parent.hyp_idx, parent.ref_idx);
    bool is_delim = (parent.src->hyp_char_types[child.hyp_idx] == 0);
    child.open_cost += is_delim ? 1.0 : 2.0;
    child.open_cost += (is_backtrace || is_delim) ? 0.0 : 1.0;

    if (child.src->hyp[child.hyp_idx] == '>') {
        end_insertion_segment(child, child.hyp_idx, child.ref_idx);
    }

    out_child = std::move(child);
    return 1;
}

// expand() equivalent
static std::vector<Path> expand_paths(const Path& p) {
    std::vector<Path> out;
    out.reserve(3);
    Path child;

    if (add_delete(p, child)) out.push_back(child);
    if (add_insert(p, child)) out.push_back(child);
    if (add_substitution_or_match(p, child)) out.push_back(child);

    return out;
}

// Create a Python object with the same attributes as Path (no computed properties).
static py::object to_python_path_like(const Path& p) {
    py::dict d;
    d["src"] = p.src->src_obj; // keep original SubgraphMetadata
    d["ref_idx"] = p.ref_idx;
    d["hyp_idx"] = p.hyp_idx;
    d["last_ref_idx"] = p.last_ref_idx;
    d["last_hyp_idx"] = p.last_hyp_idx;
    d["closed_cost"] = p.closed_cost;
    d["open_cost"] = p.open_cost;
    d["at_unambiguous_match_node"] = p.at_unambiguous_match_node;
    d["sort_id"] = p.sort_id;

    // Add computed properties
    d["cost"] = p.cost();
    d["norm_cost"] = p.norm_cost();
    d["prune_id"] = static_cast<long long>(p.prune_id());
    d["index"] = py::make_tuple(p.hyp_idx, p.ref_idx);
    d["at_end"] = p.at_end();

    // end_indices as tuple of (hyp_idx, ref_idx, open_cost)
    py::list triples;
    for (const auto& t : p.end_indices) {
        triples.append(py::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t)));
    }
    d["end_indices"] = py::tuple(triples);

    // Return a SimpleNamespace so attributes are accessible via dot notation
    py::object ns = py::module_::import("types").attr("SimpleNamespace")(**d);
    return ns;
}

// ----------------- Main function -----------------
static py::object error_align_beam_search_cpp(py::object src_obj, int beam_size) {
    // View over SubgraphMetadata (already initialized on Python side)
    SrcView src = make_src_view(src_obj);

    // Initialize beam with root path
    Path start; start.src = &src;
    std::vector<Path> beam;
    beam.reserve(128);
    beam.push_back(start);

    // Prune map: prune_id -> best cost seen so far
    std::unordered_map<std::size_t, double> prune_map;
    prune_map.reserve(4096);
    std::vector<Path> ended;
    ended.reserve(128);

    while (!beam.empty()) {
        std::unordered_map<std::size_t, Path> new_beam;
        new_beam.reserve(beam.size()*3 + 8);

        for (const Path& path : beam) {
            if (path.at_end()) {
                ended.push_back(path);
                continue;
            }
            for (auto& new_path : expand_paths(path)) {
                double c = new_path.cost();
                std::size_t id = new_path.prune_id();

                auto itp = prune_map.find(id);
                if (itp != prune_map.end() && c > itp->second) {
                    continue;
                }
                prune_map[id] = c;

                auto it = new_beam.find(id);
                if (it == new_beam.end() || c < it->second.cost()) {
                    // new_beam[id] = std::move(new_path);
                    new_beam[id] = new_path;
                }
            }
        }

        // sort by norm_cost with deterministic tiebreaker
        beam.clear();
        beam.reserve(new_beam.size());
        for (auto& kv : new_beam) {
            // beam.push_back(std::move(kv.second));
            beam.push_back(kv.second);
        }
        std::sort(beam.begin(), beam.end(),
            [](const Path& a, const Path& b) {

                double an = a.norm_cost();
                double bn = b.norm_cost();

                if (an < bn) return true;
                if (bn < an) return false;

                // Tie → use deterministic hash_id tiebreaker
                return a.sort_id < b.sort_id;
            }
        );

        // prune to beam size
        if ((int)beam.size() > beam_size) beam.resize(beam_size);

        // prune to only best if at unambiguous match node
        if (!beam.empty() && beam[0].at_unambiguous_match_node) {
            beam.resize(1);
            prune_map.clear();
        }
    }

    if (ended.empty()) {
        return py::list(); // same as Python: return []
    }
    std::sort(ended.begin(), ended.end(),
        [](const Path& a, const Path& b) {

            double ac = a.cost();
            double bc = b.cost();

            if (ac < bc) return true;
            if (bc < ac) return false;

            // Tie → use deterministic hash tiebreaker
            return a.sort_id < b.sort_id;
        }
    );
    return to_python_path_like(ended.front());
}

// ----------------- Wrapper to expose expand_paths -----------------
static py::list expand_paths_wrapper(py::object py_path_obj) {
    // Extract SubgraphMetadata (src) from the Python Path object
    py::object src_obj = py_path_obj.attr("src");
    SrcView src = make_src_view(src_obj);

    // Create a C++ Path object from Python attributes
    Path p;
    p.src = &src;
    p.ref_idx = py::cast<int>(py_path_obj.attr("ref_idx"));
    p.hyp_idx = py::cast<int>(py_path_obj.attr("hyp_idx"));
    p.last_ref_idx = py::cast<int>(py_path_obj.attr("last_ref_idx"));
    p.last_hyp_idx = py::cast<int>(py_path_obj.attr("last_hyp_idx"));
    p.closed_cost = py::cast<double>(py_path_obj.attr("closed_cost"));
    p.open_cost = py::cast<double>(py_path_obj.attr("open_cost"));
    p.at_unambiguous_match_node = py::cast<bool>(py_path_obj.attr("at_unambiguous_match_node"));
    p.end_indices = py::cast<std::vector<std::tuple<int,int,double>>>(py_path_obj.attr("end_indices"));
    p.sort_id = py::cast<std::uint64_t>(py_path_obj.attr("sort_id"));


    // Expand it
    std::vector<Path> children = expand_paths(p);

    // Convert results to Python objects
    py::list py_children;
    for (auto &child : children) {
        py_children.append(to_python_path_like(child));
    }
    return py_children;
}

// ----------------- PYBIND11 MODULE -----------------
PYBIND11_MODULE(_cpp_beam_search, m) {
    m.doc() = "C++ implementation of error_align_beam_search returning a Path-like object";
    m.def("error_align_beam_search",
          &error_align_beam_search_cpp,
          py::arg("src"),
          py::arg("beam_size") = 100,
          "Perform beam search on a fully-initialized SubgraphMetadata and return a Path-like object");
    m.def("expand_paths",
        &expand_paths_wrapper,
        py::arg("path"),
        "Expand a Path object into its child paths (C++ implementation).");
}
