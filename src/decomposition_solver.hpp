#pragma once

#include "array_region.hpp"
#include "opcodes.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef CT_MIN_TILE_1D
#ifdef USE_KOKKOS
#define CT_MIN_TILE_1D 1048576
#define CT_MIN_TILE_2D 1024
#define CT_MIN_TILE_3D 128
#else
#define CT_MIN_TILE_1D 262144
#define CT_MIN_TILE_2D 512
#define CT_MIN_TILE_3D 64
#endif
#endif

namespace decomposition_solver {

namespace detail {

inline int ct_min_tile(int ndims) {
    switch (ndims) {
    case 1:
        return CT_MIN_TILE_1D;
    case 2:
        return CT_MIN_TILE_2D;
    case 3:
        return CT_MIN_TILE_3D;
    default:
        return 1;
    }
}

inline int64_t tile_count(const std::array<int, 3>& shape, int ndims, int tile) {
    int64_t count = 1;
    for (int d = 0; d < ndims; ++d)
        count *= (std::max(1, shape[d]) + tile - 1) / tile;
    return count;
}

// Elements on the busiest PE when an array of this shape (offset 0) is cut into
// tiles placed by chare_home_pe(); estimated as tiles per PE times a full tile when
// !exact.
inline int64_t busiest_pe_elems(const std::array<int, 3>& shape, int ndims, int tile,
                                int num_pes, bool exact) {
    std::array<int, 3> n = {1, 1, 1};
    int64_t count = 1, full = 1;
    for (int d = 0; d < ndims; ++d) {
        n[d] = (std::max(1, shape[d]) + tile - 1) / tile;
        count *= n[d];
        full *= std::min(tile, std::max(1, shape[d]));
    }
    if (!exact)
        return (count + num_pes - 1) / num_pes * full;
    std::unordered_map<int, int64_t> load;
    int64_t busiest = 0;
    std::array<int, 3> idx = {0, 0, 0};
    for (int64_t c = 0; c < count; ++c) {
        int64_t elems = 1;
        for (int d = 0; d < ndims; ++d)
            elems *= std::min(tile, std::max(1, shape[d]) - idx[d] * tile);
        busiest = std::max(busiest, load[chare_home_pe(ndims, idx.data(), num_pes)] += elems);
        for (int d = ndims - 1; d >= 0 && ++idx[d] == n[d]; --d)
            idx[d] = 0;
    }
    return busiest;
}

// Near-equal tiles: ceil(n / k) >= min_tile along some dimension, rounded up to a small
// multiple, at most max_tiles of them; the one that minimizes the busiest PE's elements
// (near-ties: more tiles) replaces the smallest power-of-two multiple of min_tile within
// max_tiles when it unloads the busiest PE by 20%. Smaller gains do not pay for tiles
// that cut the array's sub-blocks unevenly (a (7*365,160,240) array filled 365 rows
// at a time: tile 320 instead of 256 balances the whole array 17% better, and its
// slab writes 10% worse).
inline int balanced_tile(const std::array<int, 3>& shape, int ndims, int64_t max_tiles,
                         int num_pes, int min_tile) {
    int smax = 1;
    for (int d = 0; d < ndims; ++d)
        smax = std::max(smax, shape[d]);
    if (smax <= min_tile)
        return min_tile;
    struct Key {
        std::array<int, 3> shape;
        int ndims, num_pes, min_tile;
        int64_t max_tiles;
        bool operator<(const Key& o) const {
            return std::tie(shape, ndims, num_pes, min_tile, max_tiles) <
                   std::tie(o.shape, o.ndims, o.num_pes, o.min_tile, o.max_tiles);
        }
    };
    static thread_local std::map<Key, int> cache;
    Key key{{shape[0], ndims > 1 ? shape[1] : 0, ndims > 2 ? shape[2] : 0}, ndims,
            std::max(1, num_pes), min_tile, max_tiles};
    if (auto it = cache.find(key); it != cache.end())
        return it->second;
    int pow2 = min_tile;
    while (tile_count(shape, ndims, pow2) > max_tiles && pow2 < smax)
        pow2 *= 2;
    if (key.num_pes == 1)
        return cache[key] = pow2;

    int align = std::max(1, std::min(ndims == 1 ? 1024 : ndims == 2 ? 32 : 16, min_tile));
    std::set<int> cands;
    for (int d = 0; d < ndims; ++d)
        for (int64_t k = 1; k <= std::min<int64_t>(std::max<int64_t>(1, max_tiles), 256); ++k) {
            int64_t t = ((std::max(1, shape[d]) + k - 1) / k + align - 1) / align * align;
            if (t < min_tile)
                break;
            cands.insert((int)std::min<int64_t>(t, std::numeric_limits<int>::max()));
        }
    cands.insert((smax + align - 1) / align * align);
    // one estimator for every candidate, and at most ~1M chare visits per shape
    bool exact = max_tiles <= 4096 && (int64_t)cands.size() * max_tiles <= (1 << 20);
    int best = -1;
    int64_t best_load = 0, best_count = 0;
    for (auto it = cands.rbegin(); it != cands.rend(); ++it) {
        int t = *it;
        int64_t count = tile_count(shape, ndims, t);
        if (count > max_tiles && best >= 0)
            continue;
        int64_t load = busiest_pe_elems(shape, ndims, t, key.num_pes, exact);
        if (best < 0 || load < best_load * 0.98 ||
            (load <= best_load * 1.02 && count > best_count)) {
            best = t;
            best_load = load;
            best_count = count;
        }
    }
    if (best_load >= 0.8 * busiest_pe_elems(shape, ndims, pow2, key.num_pes, exact))
        best = pow2;
    return cache[key] = best;
}

template <typename Fn>
inline void for_each_node_topo(DAG* dag, Fn&& fn) {
    if (dag == nullptr)
        return;

    std::queue<DAGNode*> topo_queue;
    std::unordered_map<DAGNode*, int> remaining_parents;
    remaining_parents.reserve(dag->nodes.size());

    // ascending node ids: every PE visits the nodes in the same order
    std::vector<std::pair<int, DAGNode*>> by_id(dag->nodes.begin(), dag->nodes.end());
    std::sort(by_id.begin(), by_id.end(),
              [](auto const& a, auto const& b) { return a.first < b.first; });
    for (auto& [id, node] : by_id) {
        remaining_parents[node] = node->num_parents;
        if (node->num_parents == 0)
            topo_queue.push(node);
    }

    int processed = 0;
    while (!topo_queue.empty()) {
        DAGNode* dag_node = topo_queue.front();
        topo_queue.pop();
        processed++;
        fn(dag_node);

        for (DAGNode* child : dag_node->children) {
            auto it = remaining_parents.find(child);
            if (it == remaining_parents.end())
                continue;
            if (--it->second == 0)
                topo_queue.push(child);
        }
    }

    assert(processed == static_cast<int>(dag->nodes.size()));
}

// ---- Fused-kernel layouts ----
// execute_node.cpp computes a fused elementwise kernel over the chare grid of one array
// (fused_output_decomp_name: the last set_region's target, else the first kept result)
// and writes every kept result, and every other set_region's result buffer, in it.

inline bool is_kept_root(ASTNode* root) {
    auto op = static_cast<Opcode>(root->opcode);
    return !root->is_temp && op != Opcode::NOOP && op != Opcode::CREATE &&
           op != Opcode::SET_REGION;
}

/// Nodes the fused elementwise executor runs (cross-rank set_regions run as copies).
inline bool is_fused_kernel(DAGNode* node) {
    bool any = false;
    for (ASTNode* root : node->ast->roots) {
        auto op = static_cast<Opcode>(root->opcode);
        if (op == Opcode::NOOP || op == Opcode::CREATE)
            continue;
        if (op == Opcode::SET_REGION) {
            if (root->operands.size() < 2 || root->region == nullptr)
                return false;
            ASTNode* rhs = root->operands[1];
            if (!rhs->is_scalar && !rhs->is_broadcast && rhs->ndims != root->ndims)
                return false;
        } else if (!is_elementwise(op) && op != Opcode::COPY) {
            return false;
        }
        any = true;
    }
    return any;
}

/// Start and element count per dimension of `region` (nullptr/global: the whole shape).
struct Box {
    int nd = 0;
    std::array<int, 3> start{}, extent{}, step{1, 1, 1};
    bool unit = true;
};

inline Box region_box(Region* region, int nd, std::array<int, 3> const& shape) {
    Box b;
    b.nd = nd;
    for (int d = 0; d < nd; ++d)
        b.extent[d] = shape[d];
    if (region == nullptr || region->is_global)
        return b;
    auto fill = [&](auto* r, int n) {
        for (int d = 0; d < n; ++d) {
            int st = r->step[d] > 0 ? r->step[d] : 1;
            b.start[d] = r->start[d];
            b.step[d] = st;
            b.extent[d] = std::max(0, (r->stop[d] - r->start[d] + st - 1) / st);
            b.unit &= st == 1;
        }
    };
    switch (nd) {
    case 1: fill(static_cast<ArrayRegion<1>*>(region), 1); break;
    case 2: fill(static_cast<ArrayRegion<2>*>(region), 2); break;
    case 3: fill(static_cast<ArrayRegion<3>*>(region), 3); break;
    default: break;
    }
    return b;
}

/// The array whose decomposition (offset + shift) lays out the kernel, and its extent.
struct KernelLayout {
    int anchor = -1;
    ASTNode* anchor_set = nullptr;
    Box box;
};

template <typename ArrayMetaMap>
bool kernel_layout(ArrayMetaMap& meta, DAGNode* node, KernelLayout& k) {
    for (auto it = node->ast->roots.rbegin(); it != node->ast->roots.rend(); ++it)
        if (static_cast<Opcode>((*it)->opcode) == Opcode::SET_REGION) {
            auto m = meta.find((*it)->operands[0]->result_name);
            if (m == meta.end())
                return false;
            k.anchor = m->first;
            k.anchor_set = *it;
            k.box = region_box((*it)->region, m->second.ndims, m->second.global_shape);
            return true;
        }
    for (ASTNode* root : node->ast->roots)
        if (is_kept_root(root)) {
            auto m = meta.find(root->result_name);
            if (m == meta.end())
                return false;
            k.anchor = m->first;
            k.box = region_box(nullptr, m->second.ndims, m->second.global_shape);
            return true;
        }
    return false;
}

/// Whether an array of layout `m`, shifted by `start`, is the kernel's layout.
template <typename Meta>
bool same_layout(Meta const& m, std::array<int, 3> const& start, Box const& extent,
                 Meta const& anchor, KernelLayout const& k) {
    if (!k.box.unit || !extent.unit || m.ndims != k.box.nd || m.tile != anchor.tile)
        return false;
    for (int d = 0; d < m.ndims; ++d)
        if (extent.extent[d] != k.box.extent[d] ||
            m.offset[d] + start[d] != anchor.offset[d] + k.box.start[d])
            return false;
    return true;
}

/// Ties each fused kernel's kept results and set_region targets to its anchor with a
/// union-find over offset differences, so later kernels never undo earlier ones. Arrays
/// other executors fix (earlier DAGs, zero-offset results, non-elementwise results) pin
/// their set; a tie that contradicts a pin is skipped.
template <typename ArrayMetaMap>
void tie_kernel_layouts(ArrayMetaMap& meta, DAG* dag, std::unordered_set<int> const& pinned) {
    using Off = std::array<int, 3>;
    std::unordered_set<int> movable;
    for_each_node_topo(dag, [&](DAGNode* node) {
        bool fused = is_fused_kernel(node);
        for (ASTNode* root : node->ast->roots) {
            auto op = static_cast<Opcode>(root->opcode);
            if ((fused && is_kept_root(root)) || is_creation(op))
                movable.insert(root->result_name);
        }
    });
    auto is_movable = [&](int name) {
        auto it = meta.find(name);
        return it != meta.end() && !it->second.decomp_final && movable.count(name) &&
               !pinned.count(name);
    };

    std::map<int, int> parent;
    std::map<int, Off> rel; // offset(x) = offset(parent) + rel
    std::map<int, std::vector<int>> members;
    std::map<int, bool> fixed;
    std::map<int, Off> rep_off;
    std::map<int, int> rep_tile;
    auto add = [&](int n) {
        if (parent.count(n))
            return;
        parent[n] = n;
        rel[n] = {0, 0, 0};
        members[n] = {n};
        fixed[n] = !is_movable(n);
        rep_off[n] = meta.at(n).offset;
        rep_tile[n] = meta.at(n).tile;
    };
    auto find = [&](int n) {
        Off s = {0, 0, 0};
        int r = n;
        while (parent[r] != r) {
            for (int d = 0; d < 3; ++d)
                s[d] += rel[r][d];
            r = parent[r];
        }
        if (n != r) {
            parent[n] = r;
            rel[n] = s;
        }
        return std::make_pair(r, s);
    };
    // offset(a) = offset(b) + shift
    auto unite = [&](int a, int b, Off shift, int nd) {
        add(a);
        add(b);
        auto [ra, sa] = find(a);
        auto [rb, sb] = find(b);
        Off diff = {0, 0, 0}; // offset(ra) = offset(rb) + diff
        for (int d = 0; d < nd; ++d)
            diff[d] = sb[d] + shift[d] - sa[d];
        if (ra == rb)
            return;
        auto fits = [&](int rep, Off off) {
            for (int m : members[rep]) {
                Off sm = find(m).second;
                for (int d = 0; d < nd; ++d)
                    if (off[d] + sm[d] < 0)
                        return false;
            }
            return true;
        };
        if (fixed[ra] && fixed[rb]) {
            if (rep_tile[ra] != rep_tile[rb])
                return;
            for (int d = 0; d < nd; ++d)
                if (rep_off[ra][d] != rep_off[rb][d] + diff[d])
                    return;
        } else if (fixed[ra]) {
            Off off = rep_off[ra];
            for (int d = 0; d < nd; ++d)
                off[d] -= diff[d];
            if (!fits(rb, off))
                return;
            rep_off[rb] = off;
            rep_tile[rb] = rep_tile[ra];
            fixed[rb] = true;
        } else if (fixed[rb]) {
            Off off = rep_off[rb];
            for (int d = 0; d < nd; ++d)
                off[d] += diff[d];
            if (!fits(ra, off))
                return;
        }
        parent[ra] = rb;
        rel[ra] = diff;
        members[rb].insert(members[rb].end(), members[ra].begin(), members[ra].end());
        members.erase(ra);
    };

    for_each_node_topo(dag, [&](DAGNode* node) {
        KernelLayout k;
        if (!is_fused_kernel(node) || !kernel_layout(meta, node, k) || !k.box.unit)
            return;
        int nd = k.box.nd;
        for (ASTNode* root : node->ast->roots) {
            int name = -1;
            Box b;
            if (is_kept_root(root)) {
                auto it = meta.find(root->result_name);
                if (it == meta.end() || root->result_name == k.anchor)
                    continue;
                name = it->first;
                b = region_box(nullptr, it->second.ndims, it->second.global_shape);
            } else if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION &&
                       root != k.anchor_set) {
                auto it = meta.find(root->operands[0]->result_name);
                if (it == meta.end() || it->first == k.anchor)
                    continue;
                name = it->first;
                b = region_box(root->region, it->second.ndims, it->second.global_shape);
            } else {
                continue;
            }
            bool fits = b.unit && b.nd == nd;
            for (int d = 0; fits && d < nd; ++d)
                fits = b.extent[d] == k.box.extent[d];
            if (!fits)
                continue;
            Off shift = {0, 0, 0};
            for (int d = 0; d < nd; ++d)
                shift[d] = k.box.start[d] - b.start[d];
            unite(name, k.anchor, shift, nd);
        }
    });

    for (auto& [rep, list] : members) {
        if (list.size() < 2)
            continue;
        int nd = meta.at(rep).ndims;
        Off off = rep_off[rep];
        int tile = rep_tile[rep];
        if (!fixed[rep] && tile > 0)
            for (int d = 0; d < nd; ++d) {
                int lo = 0;
                for (int m : list)
                    lo = std::min(lo, off[d] + find(m).second[d]);
                if (lo < 0)
                    off[d] += (-lo + tile - 1) / tile * tile;
            }
        for (int m : list) {
            if (!is_movable(m))
                continue;
            Off sm = find(m).second;
            auto& mm = meta.at(m);
            mm.tile = tile;
            for (int d = 0; d < nd; ++d)
                mm.offset[d] = off[d] + sm[d];
        }
    }
}

} // namespace detail

/// Chares the receive pass inserts for `root`'s result: the box [0, ext) of array indices,
/// which includes the decomposition offset. Returns its rank, or 0 for none.
template <typename ArrayMetaMap>
int chare_cover(ArrayMetaMap const& meta, ASTNode* root, std::array<int, 3>& ext, int& tile) {
    auto op = static_cast<Opcode>(root->opcode);
    auto it = meta.find(root->result_name);
    if (it == meta.end())
        return 0;
    auto const& m = it->second;
    int nd = m.ndims;
    std::array<int, 3> shape = m.global_shape;
    if (is_creation(op)) {
        if (root->region == nullptr)
            return 0;
        nd = root->ndims;
        shape = detail::region_box(root->region, nd, m.global_shape).extent;
    } else if (op == Opcode::MATMUL) {
        nd = 1;
    } else if (op == Opcode::MATMATMUL) {
        nd = 2;
    } else if (op == Opcode::DIAG) {
        shape[1] = shape[0];
    } else if (op == Opcode::TRANSPOSE) {
        if (nd < 2)
            return 0;
    } else if (is_elementwise(op) || op == Opcode::COPY || is_castscansort(op)) {
        if (shape[0] <= 0)
            return 0;
    } else if (!(op == Opcode::TILE || is_reduction_op(op) || op == Opcode::LINALG ||
                 is_shape_op(op) || is_signal_op(op) || op == Opcode::GATHER ||
                 op == Opcode::EXPAND || op == Opcode::COMPRESS)) {
        return 0;
    }
    if (nd < 1 || nd > 3)
        return 0;
    tile = m.tile;
    for (int d = 0; d < nd; ++d)
        ext[d] = m.offset[d] + shape[d];
    return nd;
}

template <typename ArrayMetaMap>
void compute_decompositions(ArrayMetaMap& array_meta, DAG* dag, int odf, int num_pes) {
    using ArrayMetadata = typename ArrayMetaMap::mapped_type;

    auto zero_offset = std::array<int, 3>{0, 0, 0};

    // Tiles are square, so a skinny matrix gets many tiny tiles; also cap the count by
    // size so each 2-D tile holds at least a quarter of a minimum square tile.
    auto max_tiles_for = [&](const ArrayMetadata& m) -> int64_t {
        int64_t elems = 1, min_elems = 1;
        for (int d = 0; d < m.ndims; d++) {
            elems *= std::max(1, m.global_shape[d]);
            min_elems *= detail::ct_min_tile(m.ndims);
        }
        int64_t by_size = m.ndims == 2 ? std::max<int64_t>(1, 4 * elems / min_elems)
                                       : std::numeric_limits<int>::max();
        return std::min<int64_t>(std::max(1, odf) * std::max(1, num_pes), by_size);
    };
    auto own_tile = [&](const ArrayMetadata& m) {
        return detail::balanced_tile(m.global_shape, m.ndims, max_tiles_for(m), num_pes,
                                     detail::ct_min_tile(m.ndims));
    };
    auto clamp_tile_for_max_count = [&](ArrayMetadata& m) {
        if (detail::tile_count(m.global_shape, m.ndims, m.tile) > max_tiles_for(m))
            m.tile = std::max(m.tile, own_tile(m));
    };

    for (auto& [name, meta] : array_meta) {
        if (meta.tile == 0)
            meta.tile = detail::ct_min_tile(meta.ndims);

        if (!meta.decomp_final) {
            meta.tile = own_tile(meta);
            meta.offset = zero_offset;
        }
    }

    if (dag != nullptr) {
        std::unordered_set<int> matmul_arrays;

        auto positive_mod = [](int value, int mod) {
            int r = value % mod;
            return r < 0 ? r + mod : r;
        };

        auto gcd_positive = [](int a, int b) {
            if (a < 0)
                a = -a;
            if (b < 0)
                b = -b;
            while (b != 0) {
                int t = a % b;
                a = b;
                b = t;
            }
            return a;
        };

        auto region_stride_factor = [&](Region* region_base, int ndims) {
            if (region_base == nullptr || region_base->is_global)
                return 1;

            int factor = 0;
            auto fold_step = [&](int step) {
                if (step < 0)
                    step = -step;
                if (step <= 1)
                    return;
                factor = (factor == 0) ? step : gcd_positive(factor, step);
            };

            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                fold_step(r->step[0]);
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                fold_step(r->step[0]);
                fold_step(r->step[1]);
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                fold_step(r->step[0]);
                fold_step(r->step[1]);
                fold_step(r->step[2]);
                break;
            }
            default:
                break;
            }

            return factor > 1 ? factor : 1;
        };

        auto region_is_dense = [](Region* region_base, int ndims) {
            if (region_base == nullptr || region_base->is_global)
                return true;

            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                return r->step[0] == 1;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                return r->step[0] == 1 && r->step[1] == 1;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                return r->step[0] == 1 && r->step[1] == 1 && r->step[2] == 1;
            }
            default:
                return false;
            }
        };

        // smallest tile each array was offered through a strided or cross-rank access
        std::unordered_map<int, int> tile_cap;
        std::set<std::pair<int, int>> unit_links;
        std::unordered_set<int> anchored;

        auto try_assign_tile = [&](int array_name, int candidate_tile) {
            auto array_it = array_meta.find(array_name);
            if (array_it == array_meta.end())
                return false;

            auto& meta = array_it->second;
            if (meta.decomp_final)
                return false;
            if (candidate_tile <= 0)
                return false;

            candidate_tile = std::max(candidate_tile, detail::ct_min_tile(meta.ndims));
            auto cap = tile_cap.emplace(array_name, candidate_tile).first;
            cap->second = std::min(cap->second, candidate_tile);

            if (meta.tile > 0 && candidate_tile >= meta.tile)
                return false;

            meta.tile = candidate_tile;
            return true;
        };

        auto propose_tile_from_access = [&](int array_name, int ndims, int source_name,
                                            Region* access_region) {
            if (source_name == array_name)
                return false;

            auto src_it = array_meta.find(source_name);
            if (src_it == array_meta.end() || src_it->second.ndims != ndims)
                return false;

            // unit-stride accesses share a tile (assign_class_tiles)
            int stride_factor = region_stride_factor(access_region, ndims);
            if (stride_factor <= 1) {
                unit_links.insert({std::min(array_name, source_name),
                                   std::max(array_name, source_name)});
                return false;
            }
            int candidate_tile = std::max(1, src_it->second.tile / stride_factor);
            return try_assign_tile(array_name, candidate_tile);
        };

        // Each temp is expanded once per walk: fused chains reuse temps, and
        // re-expanding per use is exponential in chain length.
        auto collect_tile_candidates =
            [&](auto&& self, int array_name, int ndims, ASTNode* node, Region* carried_region,
                const std::unordered_map<int, ASTNode*>& temp_defs,
                std::unordered_set<int>& visited_temps) -> bool {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return false;

            auto meta_it = array_meta.find(node->result_name);
            if (meta_it != array_meta.end()) {
                if (meta_it->second.ndims == ndims)
                    return propose_tile_from_access(array_name, ndims, node->result_name,
                                                    carried_region);
                return false;
            }

            if (node->operands.empty()) {
                auto def_it = temp_defs.find(node->result_name);
                if (def_it != temp_defs.end()) {
                    if (!visited_temps.insert(node->result_name).second)
                        return false;
                    ASTNode* def_root = def_it->second;
                    bool changed = false;
                    for (int i = 0; i < (int)def_root->operands.size(); ++i) {
                        changed = self(self, array_name, ndims, def_root->operands[i],
                                       def_root->get_operand_region(i), temp_defs,
                                       visited_temps) ||
                                  changed;
                    }
                    return changed;
                }
                return false;
            }

            bool changed = false;
            for (int child_idx = 0; child_idx < (int)node->operands.size(); ++child_idx) {
                changed = self(self, array_name, ndims, node->operands[child_idx],
                               node->get_operand_region(child_idx), temp_defs, visited_temps) ||
                          changed;
            }
            return changed;
        };

        auto is_ast_leaf = [](ASTNode* node) -> bool {
            for (ASTNode* child : node->operands)
                if (child != nullptr && !child->is_scalar && !child->is_broadcast)
                    return false;
            return true;
        };

        // Arrays joined by unit-stride elementwise accesses form one tile class, so their
        // tiles line up and only halos move. A class with an array from an earlier DAG
        // (or a TRANSPOSE result) takes the heaviest such array's tile; otherwise the
        // balanced tile of its largest extents, capped by strided accesses.
        auto assign_class_tiles = [&]() -> bool {
            std::unordered_map<int, int> parent;
            std::function<int(int)> find = [&](int x) -> int {
                auto it = parent.find(x);
                if (it == parent.end() || it->second == x)
                    return parent[x] = x;
                int r = find(it->second);
                return parent[x] = r;
            };
            auto unite = [&](int a, int b) {
                auto ia = array_meta.find(a), ib = array_meta.find(b);
                if (a == b || ia == array_meta.end() || ib == array_meta.end() ||
                    ia->second.ndims != ib->second.ndims)
                    return;
                int ra = find(a), rb = find(b);
                parent[rb] = ra;
            };
            std::unordered_set<int> anchors;
            for (auto& [name, meta] : array_meta)
                if (meta.decomp_final)
                    anchors.insert(name);
            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                for (ASTNode* root : dag_node->ast->roots) {
                    Opcode op = static_cast<Opcode>(root->opcode);
                    if (op == Opcode::TRANSPOSE && !root->operands.empty()) {
                        auto in = array_meta.find(root->operands[0]->result_name);
                        auto out = array_meta.find(root->result_name);
                        if (in != array_meta.end() && out != array_meta.end() &&
                            !out->second.decomp_final) {
                            out->second.tile = in->second.tile;
                            anchors.insert(root->result_name);
                        }
                    } else if (op == Opcode::REDUCE && root->operands.size() >= 2) {
                        unite(root->operands[0]->result_name, root->operands[1]->result_name);
                    }
                }
            });
            for (auto& [a, b] : unit_links)
                unite(a, b);

            // Members under half the largest one's size keep their own tile: a class
            // tile sized for a big array would leave a small one on few PEs.
            struct TileClass {
                std::vector<std::pair<int, int64_t>> members;
                int64_t max_elems = 0;
                int anchor = -1;
                int64_t anchor_elems = -1;
                int cap = std::numeric_limits<int>::max();
                ArrayMetadata envelope{};
            };
            auto elems_of = [](const ArrayMetadata& m) {
                int64_t elems = 1;
                for (int d = 0; d < m.ndims; ++d)
                    elems *= std::max(1, m.global_shape[d]);
                return elems;
            };
            std::map<int, TileClass> classes;
            for (auto& [name, meta] : array_meta) {
                if (!parent.count(name))
                    continue;
                auto& c = classes[find(name)];
                int64_t elems = elems_of(meta);
                c.members.push_back({name, elems});
                c.max_elems = std::max(c.max_elems, elems);
            }
            for (auto& [rep, c] : classes) {
                auto members = std::move(c.members);
                c.members.clear();
                for (auto [name, elems] : members) {
                    if (2 * elems < c.max_elems)
                        continue;
                    auto& meta = array_meta[name];
                    if (c.members.empty())
                        c.envelope = meta;
                    c.members.push_back({name, elems});
                    for (int d = 0; d < meta.ndims; ++d)
                        c.envelope.global_shape[d] =
                            std::max(c.envelope.global_shape[d], meta.global_shape[d]);
                    if (anchors.count(name) && (elems > c.anchor_elems ||
                                                (elems == c.anchor_elems && name < c.anchor))) {
                        c.anchor = name;
                        c.anchor_elems = elems;
                    }
                    if (auto cap = tile_cap.find(name); cap != tile_cap.end())
                        c.cap = std::min(c.cap, cap->second);
                }
            }

            bool changed = false;
            for (auto& [rep, c] : classes) {
                if (c.members.size() < 2)
                    continue;
                int tile = c.anchor >= 0 ? array_meta[c.anchor].tile
                                         : std::min(own_tile(c.envelope), c.cap);
                int64_t anchor_tiles = 0;
                if (c.anchor >= 0) {
                    auto& a = array_meta[c.anchor];
                    anchor_tiles = detail::tile_count(a.global_shape, a.ndims, a.tile);
                }
                for (auto [name, elems] : c.members) {
                    auto& meta = array_meta[name];
                    bool too_many = detail::tile_count(meta.global_shape, meta.ndims, tile) >
                                    std::max(max_tiles_for(meta), anchor_tiles);
                    if (c.anchor >= 0 && (anchors.count(name) || !too_many || meta.tile == tile))
                        anchored.insert(name);
                    if (anchors.count(name) || too_many || meta.tile == tile)
                        continue;
                    meta.tile = tile;
                    changed = true;
                }
            }
            return changed;
        };

        bool tiles_changed = true;
        int max_tile_passes = std::max(64, static_cast<int>(array_meta.size()));
        // strided proposals to a fixpoint, then tile classes; repeat while classes move
        for (int iter = 0, rounds = 0; iter < max_tile_passes; ++iter) {
            if (!tiles_changed && (rounds++ == 4 || !assign_class_tiles()))
                break;
            tiles_changed = false;

            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                std::unordered_map<int, ASTNode*> temp_defs;
                for (ASTNode* root : dag_node->ast->roots) {
                    if (root->is_temp)
                        temp_defs[root->result_name] = root;
                }

                for (ASTNode* ast_node : dag_node->ast->roots) {
                    Opcode op = static_cast<Opcode>(ast_node->opcode);

                    if (op == Opcode::SET_REGION) {
                        if (ast_node->operands.size() < 2)
                            continue;

                        int dst_name = ast_node->operands[0]->result_name;
                        auto dst_it = array_meta.find(dst_name);
                        if (dst_it == array_meta.end())
                            continue;

                        if (region_is_dense(ast_node->region, dst_it->second.ndims)) {
                            for (int op_idx = 1; op_idx < (int)ast_node->operands.size();
                                 ++op_idx) {
                                std::unordered_set<int> visited_temps;
                                tiles_changed =
                                    collect_tile_candidates(collect_tile_candidates, dst_name,
                                                            dst_it->second.ndims,
                                                            ast_node->operands[op_idx],
                                                            ast_node->get_operand_region(op_idx),
                                                            temp_defs, visited_temps) ||
                                    tiles_changed;
                            }
                        }

                        for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                            ASTNode* src_op = ast_node->operands[op_idx];
                            if (!is_ast_leaf(src_op))
                                continue;
                            auto src_it_cd = array_meta.find(src_op->result_name);
                            if (src_it_cd == array_meta.end())
                                continue;
                            if (src_it_cd->second.ndims == dst_it->second.ndims)
                                continue;
                            int higher_tile;
                            int lower_name_cd;
                            if (src_it_cd->second.ndims > dst_it->second.ndims) {
                                higher_tile = src_it_cd->second.tile;
                                lower_name_cd = dst_name;
                            } else {
                                higher_tile = dst_it->second.tile;
                                lower_name_cd = src_op->result_name;
                            }
                            tiles_changed =
                                try_assign_tile(lower_name_cd, higher_tile) || tiles_changed;
                        }
                        continue;
                    }

                    if (is_elementwise(op) || op == Opcode::COPY) {
                        auto out_it = array_meta.find(ast_node->result_name);
                        if (out_it == array_meta.end())
                            continue;

                        for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                            std::unordered_set<int> visited_temps;
                            tiles_changed =
                                collect_tile_candidates(collect_tile_candidates,
                                                        ast_node->result_name,
                                                        out_it->second.ndims,
                                                        ast_node->operands[op_idx],
                                                        ast_node->get_operand_region(op_idx),
                                                        temp_defs, visited_temps) ||
                                tiles_changed;
                        }
                    }
                }
            });
        }

        for (auto& [name, meta] : array_meta) {
            if (!meta.decomp_final && !anchored.count(name))
                clamp_tile_for_max_count(meta);
        }

        auto extract_region_start = [](Region* region_base, int ndims) {
            std::array<int, 3> region_start = {0, 0, 0};
            if (region_base == nullptr || region_base->is_global)
                return region_start;
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_start[0] = r->start[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_start[0] = r->start[0];
                region_start[1] = r->start[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_start[0] = r->start[0];
                region_start[1] = r->start[1];
                region_start[2] = r->start[2];
                break;
            }
            default:
                break;
            }
            return region_start;
        };

        auto extract_region_step = [](Region* region_base, int ndims) {
            std::array<int, 3> region_step = {1, 1, 1};
            if (region_base == nullptr || region_base->is_global)
                return region_step;
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_step[0] = r->step[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_step[0] = r->step[0];
                region_step[1] = r->step[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_step[0] = r->step[0];
                region_step[1] = r->step[1];
                region_step[2] = r->step[2];
                break;
            }
            default:
                break;
            }
            return region_step;
        };

        auto extract_region_stop = [](Region* region_base, int ndims,
                                      const std::array<int, 3>& global_shape) {
            std::array<int, 3> region_stop = {0, 0, 0};
            if (region_base == nullptr || region_base->is_global) {
                for (int d = 0; d < ndims; ++d)
                    region_stop[d] = global_shape[d];
                return region_stop;
            }
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_stop[0] = r->stop[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_stop[0] = r->stop[0];
                region_stop[1] = r->stop[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_stop[0] = r->stop[0];
                region_stop[1] = r->stop[1];
                region_stop[2] = r->stop[2];
                break;
            }
            default:
                break;
            }
            return region_stop;
        };

        auto circ_dist = [](int a, int b, int mod) -> int {
            int d = ((a - b) % mod + mod) % mod;
            return std::min(d, mod - d);
        };

        std::unordered_map<int, int> uf_parent, uf_rank;

        auto uf_make = [&](int x) {
            if (uf_parent.find(x) == uf_parent.end()) {
                uf_parent[x] = x;
                uf_rank[x] = 0;
            }
        };

        std::function<int(int)> uf_find = [&](int x) -> int {
            if (uf_parent[x] != x)
                uf_parent[x] = uf_find(uf_parent[x]);
            return uf_parent[x];
        };

        auto uf_unite = [&](int x, int y) {
            int rx = uf_find(x), ry = uf_find(y);
            if (rx == ry)
                return;
            if (uf_rank[rx] < uf_rank[ry])
                std::swap(rx, ry);
            uf_parent[ry] = rx;
            if (uf_rank[rx] == uf_rank[ry])
                uf_rank[rx]++;
        };

        // results below must not depend on the map's iteration order
        std::vector<int> sorted_names;
        for (auto& [name, meta] : array_meta)
            sorted_names.push_back(name);
        std::sort(sorted_names.begin(), sorted_names.end());

        for (int name : sorted_names)
            uf_make(name);

        std::unordered_set<int> source_arrays;

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);
                // TRANSPOSE results get a fixed offset (align_transpose_results)
                if (is_creation(op) || op == Opcode::TRANSPOSE)
                    source_arrays.insert(ast_node->result_name);
            }
        });

        auto collect_unionable_nonfixed_leaves = [&](auto&& self, ASTNode* node, int out_name,
                                                     int nd, int out_tile,
                                                     Region* carried_region,
                                                     std::vector<int>& leaves) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;

            if (is_ast_leaf(node)) {
                int src_name = node->result_name;
                if (src_name == out_name)
                    return;

                auto src_it = array_meta.find(src_name);
                if (src_it == array_meta.end() || src_it->second.ndims != nd ||
                    src_it->second.tile != out_tile)
                    return;

                bool is_fixed_source = src_it->second.decomp_final || source_arrays.count(src_name);
                if (is_fixed_source)
                    return;

                if (carried_region != nullptr && !carried_region->is_global) {
                    auto rstart = extract_region_start(carried_region, nd);
                    auto rstep = extract_region_step(carried_region, nd);
                    for (int d = 0; d < nd; ++d) {
                        if (rstart[d] != 0 || rstep[d] != 1)
                            return;
                    }
                }

                leaves.push_back(src_name);
                return;
            }

            for (int i = 0; i < (int)node->operands.size(); ++i) {
                ASTNode* child = node->operands[i];
                if (child == nullptr || child->is_scalar || child->is_broadcast)
                    continue;
                Region* child_region = node->get_operand_region(i);
                self(self, child, out_name, nd, out_tile,
                     child_region ? child_region : carried_region, leaves);
            }
        };

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);

                if (is_creation(op)) {
                    continue;
                }

                if (is_elementwise(op) || op == Opcode::COPY) {
                    int out_name = ast_node->result_name;
                    auto out_it = array_meta.find(out_name);
                    if (out_it == array_meta.end())
                        continue;
                    int nd = out_it->second.ndims;
                    int out_tile = out_it->second.tile;

                    std::vector<int> union_leaves;
                    for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* operand = ast_node->operands[op_idx];
                        if (operand == nullptr || operand->is_scalar || operand->is_broadcast)
                            continue;
                        Region* op_region = ast_node->get_operand_region(op_idx);
                        collect_unionable_nonfixed_leaves(collect_unionable_nonfixed_leaves,
                                                          operand, out_name, nd, out_tile,
                                                          op_region, union_leaves);
                    }
                    for (int src_name : union_leaves)
                        uf_unite(out_name, src_name);
                    continue;
                }

                // REDUCE takes the dot of the two operands' local tiles
                if (op == Opcode::REDUCE && ast_node->operands.size() >= 2) {
                    int a = ast_node->operands[0]->result_name;
                    int b = ast_node->operands[1]->result_name;
                    auto ia = array_meta.find(a), ib = array_meta.find(b);
                    if (a != b && ia != array_meta.end() && ib != array_meta.end() &&
                        ia->second.tile == ib->second.tile &&
                        !ia->second.decomp_final && !source_arrays.count(a) &&
                        !ib->second.decomp_final && !source_arrays.count(b))
                        uf_unite(a, b);
                    continue;
                }

                if (op == Opcode::MATMUL || op == Opcode::MATMATMUL) {
                    for (ASTNode* operand : ast_node->operands)
                        if (operand != nullptr && !operand->is_scalar)
                            matmul_arrays.insert(operand->result_name);
                    matmul_arrays.insert(ast_node->result_name);
                }

                // These results are routed by a zero-offset decomposition; BINCOUNT
                // also needs x and weights laid out identically.
                if (is_reduction_op(op) || op == Opcode::LINALG || is_shape_op(op) ||
                    is_signal_op(op) || op == Opcode::GATHER || op == Opcode::EXPAND ||
                    op == Opcode::COMPRESS || op == Opcode::TILE || op == Opcode::DIAG) {
                    matmul_arrays.insert(ast_node->result_name);
                    if (op == Opcode::BINCOUNT)
                        for (ASTNode* operand : ast_node->operands)
                            if (operand != nullptr && !operand->is_scalar)
                                matmul_arrays.insert(operand->result_name);
                }
            }
        });

        // TRANSPOSE result: source tile, permuted (source offset + view start),
        // so source tiles map 1:1 onto result tiles. Matmul operands stay at 0.
        auto align_transpose_results = [&]() {
            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                for (ASTNode* ast_node : dag_node->ast->roots) {
                    if (static_cast<Opcode>(ast_node->opcode) != Opcode::TRANSPOSE ||
                        ast_node->operands.empty())
                        continue;
                    auto in_it = array_meta.find(ast_node->operands[0]->result_name);
                    auto out_it = array_meta.find(ast_node->result_name);
                    if (in_it == array_meta.end() || out_it == array_meta.end() ||
                        out_it->second.decomp_final)
                        continue;
                    auto& in = in_it->second;
                    auto& out = out_it->second;
                    int nd = out.ndims;
                    if (in.ndims != nd || (int)ast_node->operands.size() < nd + 1)
                        continue;
                    Region* view = ast_node->get_operand_region(0);
                    auto vstart = extract_region_start(view, nd);
                    auto vstep = extract_region_step(view, nd);
                    out.tile = in.tile;
                    out.offset = zero_offset;
                    if (!matmul_arrays.count(ast_node->result_name)) {
                        for (int k = 0; k < nd; ++k) {
                            int d = (int)ast_node->operands[k + 1]->scalar;
                            if (vstep[d] == 1)
                                out.offset[k] = positive_mod(in.offset[d] + vstart[d], in.tile);
                        }
                    }
                }
            });
        };
        align_transpose_results();

        struct ClassInfo {
            int tile = 0;
            int ndims = 0;
            bool has_fixed_offset = false;
            std::array<int, 3> fixed_offset = {0, 0, 0};
        };
        std::unordered_map<int, ClassInfo> class_info;

        for (int name : sorted_names) {
            auto& meta = array_meta[name];
            int rep = uf_find(name);
            auto it = class_info.find(rep);
            bool is_fixed = meta.decomp_final || source_arrays.count(name);
            if (it == class_info.end()) {
                class_info[rep] = {meta.tile, meta.ndims, is_fixed, meta.offset};
            } else if (is_fixed) {
                it->second.has_fixed_offset = true;
                it->second.fixed_offset = meta.offset;
            }
        }

        struct ShiftEdge {
            int from_class;
            int to_class;
            std::array<int, 3> shift;
            std::array<int, 3> stride;
            int64_t weight;
            bool forward;
            std::array<int, 3> from_to_to = {0, 1, 2};
            std::array<int, 3> to_to_from = {0, 1, 2};
        };
        std::vector<ShiftEdge> shift_edges;

        auto add_shift_edge_if_needed = [&](int out_name, int nd, ASTNode* leaf, Region* region) {
            int src_name = leaf->result_name;
            if (src_name == out_name)
                return;
            auto src_it = array_meta.find(src_name);
            if (src_it == array_meta.end() || src_it->second.ndims != nd)
                return;

            int class_src = uf_find(src_name);
            int class_out = uf_find(out_name);
            if (class_src == class_out)
                return;

            std::array<int, 3> rstart = {0, 0, 0};
            std::array<int, 3> rstep = {1, 1, 1};
            if (region != nullptr && !region->is_global) {
                rstart = extract_region_start(region, nd);
                rstep = extract_region_step(region, nd);
            }

            int64_t weight = 1;
            for (int d = 0; d < nd; ++d)
                weight *= static_cast<int64_t>(src_it->second.global_shape[d]);

            shift_edges.push_back({class_src, class_out, rstart, rstep, weight, true});
        };

        auto visit_leaves_for_edges = [&](auto&& self, ASTNode* node, int out_name, int nd,
                                          Region* carried_region) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;
            if (is_ast_leaf(node)) {
                add_shift_edge_if_needed(out_name, nd, node, carried_region);
                return;
            }
            for (int i = 0; i < (int)node->operands.size(); ++i) {
                ASTNode* child = node->operands[i];
                if (child == nullptr || child->is_scalar || child->is_broadcast)
                    continue;
                Region* child_region = node->get_operand_region(i);
                if (is_ast_leaf(child)) {
                    add_shift_edge_if_needed(out_name, nd, child,
                                             child_region ? child_region : carried_region);
                } else {
                    self(self, child, out_name, nd, child_region);
                }
            }
        };

        auto visit_set_region_leaves = [&](auto&& self, ASTNode* node, int target_name, int nd,
                                           const std::array<int, 3>& region_start) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;
            if (is_ast_leaf(node)) {
                int src_name = node->result_name;
                if (src_name == target_name)
                    return;
                auto src_it = array_meta.find(src_name);
                if (src_it == array_meta.end() || src_it->second.ndims != nd)
                    return;

                int class_src = uf_find(src_name);
                int class_tgt = uf_find(target_name);
                if (class_src == class_tgt)
                    return;

                int64_t weight = 1;
                for (int d = 0; d < nd; ++d)
                    weight *= static_cast<int64_t>(src_it->second.global_shape[d]);

                shift_edges.push_back(
                    {class_tgt, class_src, region_start, {1, 1, 1}, weight, true});
                return;
            }
            for (ASTNode* child : node->operands)
                self(self, child, target_name, nd, region_start);
        };

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);

                if (op == Opcode::SET_REGION) {
                    if (ast_node->operands.size() < 2 || ast_node->region == nullptr)
                        continue;
                    int target_name = ast_node->operands[0]->result_name;
                    auto target_it = array_meta.find(target_name);
                    if (target_it == array_meta.end())
                        continue;
                    int nd = target_it->second.ndims;
                    auto region_start = extract_region_start(ast_node->region, nd);

                    for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx)
                        visit_set_region_leaves(visit_set_region_leaves,
                                                ast_node->operands[op_idx], target_name, nd,
                                                region_start);

                    for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* src_op = ast_node->operands[op_idx];
                        if (!is_ast_leaf(src_op))
                            continue;
                        int src_name_cd = src_op->result_name;
                        if (src_name_cd == target_name)
                            continue;
                        auto src_it_cd = array_meta.find(src_name_cd);
                        if (src_it_cd == array_meta.end())
                            continue;
                        int nd_src = src_it_cd->second.ndims;
                        if (nd_src == nd)
                            continue;

                        std::array<int, 3> from_to_to_cd = {-1, -1, -1};
                        std::array<int, 3> to_to_from_cd = {-1, -1, -1};

                        if (nd < nd_src) {
                            int td = 0;
                            for (int sd = 0; sd < nd_src && td < nd; ++sd) {
                                if (src_it_cd->second.global_shape[sd] > 1) {
                                    from_to_to_cd[td] = sd;
                                    to_to_from_cd[sd] = td;
                                    ++td;
                                }
                            }
                            if (td != nd)
                                continue;
                        } else {
                            auto region_stop = extract_region_stop(
                                ast_node->region, nd, target_it->second.global_shape);
                            int sd = 0;
                            for (int td = 0; td < nd && sd < nd_src; ++td) {
                                int size = region_stop[td] - region_start[td];
                                if (size > 1) {
                                    from_to_to_cd[td] = sd;
                                    to_to_from_cd[sd] = td;
                                    ++sd;
                                }
                            }
                            if (sd != nd_src)
                                continue;
                        }

                        int class_tgt_cd = uf_find(target_name);
                        int class_src_cd = uf_find(src_name_cd);
                        if (class_tgt_cd == class_src_cd)
                            continue;

                        int64_t weight_cd = 1;
                        int nd_lo = std::min(nd, nd_src);
                        auto& lo_meta = (nd <= nd_src) ? target_it->second : src_it_cd->second;
                        for (int d = 0; d < nd_lo; ++d)
                            weight_cd *= static_cast<int64_t>(lo_meta.global_shape[d]);

                        shift_edges.push_back({class_tgt_cd, class_src_cd, region_start,
                                               {1, 1, 1}, weight_cd, true, from_to_to_cd,
                                               to_to_from_cd});
                    }
                    continue;
                }

                if (op == Opcode::REDUCE && ast_node->operands.size() >= 2) {
                    int a = ast_node->operands[0]->result_name;
                    int b = ast_node->operands[1]->result_name;
                    auto ia = array_meta.find(a), ib = array_meta.find(b);
                    if (a == b || ia == array_meta.end() || ib == array_meta.end() ||
                        ia->second.ndims != 1 || ib->second.ndims != 1 ||
                        uf_find(a) == uf_find(b))
                        continue;
                    int64_t weight = 4 * (int64_t)std::max(1, ia->second.global_shape[0]);
                    shift_edges.push_back(
                        {uf_find(a), uf_find(b), {0, 0, 0}, {1, 1, 1}, weight, true});
                    continue;
                }

                if (is_elementwise(op) || op == Opcode::COPY) {
                    int out_name = ast_node->result_name;
                    auto out_it = array_meta.find(out_name);
                    if (out_it == array_meta.end())
                        continue;
                    int nd = out_it->second.ndims;

                    for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* operand = ast_node->operands[op_idx];
                        if (operand == nullptr || operand->is_scalar || operand->is_broadcast)
                            continue;
                        Region* op_region = ast_node->get_operand_region(op_idx);
                        if (is_ast_leaf(operand)) {
                            add_shift_edge_if_needed(out_name, nd, operand, op_region);
                        } else {
                            visit_leaves_for_edges(visit_leaves_for_edges, operand, out_name, nd,
                                                   op_region);
                        }
                    }
                }
            }
        });

        struct AdjEdge {
            int neighbor;
            std::array<int, 3> shift;
            std::array<int, 3> stride;
            std::array<int, 3> neighbor_dim = {0, 1, 2};
            int64_t weight;
            bool forward;
            int from_class;
            int to_class;
        };
        std::unordered_map<int, std::vector<AdjEdge>> adj;

        for (auto& e : shift_edges) {
            if (e.from_class == e.to_class)
                continue;
            adj[e.from_class].push_back(
                {e.to_class, e.shift, e.stride, e.from_to_to, e.weight, true, e.from_class,
                 e.to_class});
            adj[e.to_class].push_back(
                {e.from_class, e.shift, e.stride, e.to_to_from, e.weight, false, e.from_class,
                 e.to_class});
        }

        std::unordered_map<int, int> comp_id;
        std::unordered_map<int, int> tree_parent;
        std::unordered_map<int, std::vector<int>> tree_children;
        std::unordered_set<int> cyclic_components;
        int num_components = 0;

        std::set<int> all_classes;
        for (auto& [name, _] : array_meta)
            all_classes.insert(uf_find(name));

        std::map<int, int> comp_root;

        for (int cls : all_classes) {
            if (comp_id.count(cls))
                continue;
            int cid = num_components++;
            comp_root[cid] = cls;

            std::queue<int> bfs;
            bfs.push(cls);
            comp_id[cls] = cid;
            tree_parent[cls] = -1;

            while (!bfs.empty()) {
                int v = bfs.front();
                bfs.pop();
                std::unordered_set<int> seen_neighbors;
                for (auto& e : adj[v]) {
                    if (seen_neighbors.count(e.neighbor))
                        continue;
                    seen_neighbors.insert(e.neighbor);
                    if (!comp_id.count(e.neighbor)) {
                        comp_id[e.neighbor] = cid;
                        tree_parent[e.neighbor] = v;
                        tree_children[v].push_back(e.neighbor);
                        bfs.push(e.neighbor);
                    } else if (e.neighbor != tree_parent[v]) {
                        cyclic_components.insert(cid);
                    }
                }
                auto ci = class_info.find(v);
                if (ci != class_info.end() && ci->second.has_fixed_offset)
                    comp_root[cid] = v;
            }
        }

        tree_parent.clear();
        tree_children.clear();

        for (auto& [cid, root] : comp_root) {
            std::queue<int> bfs;
            bfs.push(root);
            tree_parent[root] = -1;

            std::unordered_set<int> visited;
            visited.insert(root);

            while (!bfs.empty()) {
                int v = bfs.front();
                bfs.pop();
                for (auto& e : adj[v]) {
                    if (!visited.count(e.neighbor) && comp_id[e.neighbor] == cid) {
                        visited.insert(e.neighbor);
                        tree_parent[e.neighbor] = v;
                        tree_children[v].push_back(e.neighbor);
                        bfs.push(e.neighbor);
                    }
                }
            }
        }

        constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 2;

        std::unordered_map<int, std::array<std::vector<int>, 3>> cands;

        auto add_cand = [](std::vector<int>& v, int o) {
            auto it = std::lower_bound(v.begin(), v.end(), o);
            if (it == v.end() || *it != o)
                v.insert(it, o);
        };

        auto compute_desired = [&](int o_from, int shift_d, int stride_d, int tile_to) -> int {
            int step = stride_d > 1 ? stride_d : 1;
            return positive_mod((o_from + shift_d) / step, tile_to);
        };

        auto reverse_desired = [&](int o_target, int shift_d, int stride_d, int tile_from,
                                   int tile_to) -> std::vector<int> {
            int step = stride_d > 1 ? stride_d : 1;
            std::vector<int> result;
            int base = o_target * step - shift_d;
            int period = tile_to * step;
            if (period <= 0)
                period = 1;
            int num_periods = (tile_from + period - 1) / period;
            for (int k = 0; k <= num_periods; ++k) {
                for (int delta = 0; delta < step; ++delta) {
                    int raw = base + delta + k * period;
                    int cand = ((raw % tile_from) + tile_from) % tile_from;
                    if (cand >= 0 && cand < tile_from &&
                        compute_desired(cand, shift_d, stride_d, tile_to) == o_target) {
                        result.push_back(cand);
                    }
                }
            }
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
            return result;
        };

        std::vector<int> post_order;
        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;

            std::stack<std::pair<int, bool>> stk;
            stk.push({root, false});
            while (!stk.empty()) {
                auto [v, processed] = stk.top();
                stk.pop();
                if (processed) {
                    post_order.push_back(v);
                    continue;
                }
                stk.push({v, true});
                for (int c : tree_children[v])
                    stk.push({c, false});
            }
        }

        for (int v : post_order) {
            auto ci = class_info.find(v);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;
            int tile_v = ci->second.tile;
            if (tile_v <= 0)
                tile_v = 1;

            auto& cands_v = cands[v];

            if (ci->second.has_fixed_offset) {
                for (int d = 0; d < nd; ++d)
                    add_cand(cands_v[d], positive_mod(ci->second.fixed_offset[d], tile_v));
            }

            for (int c : tree_children[v]) {
                auto cc = class_info.find(c);
                if (cc == class_info.end())
                    continue;
                int tile_c = cc->second.tile;
                if (tile_c <= 0)
                    tile_c = 1;

                for (auto& e : adj[v]) {
                    if (e.neighbor != c)
                        continue;
                    for (int d = 0; d < nd; ++d) {
                        int d_c = e.neighbor_dim[d];
                        if (d_c < 0)
                            continue;
                        int d_from = e.forward ? d : d_c;
                        for (int oc : cands[c][d_c]) {
                            if (e.forward) {
                                for (int rv : reverse_desired(oc, e.shift[d_from],
                                                              e.stride[d_from], tile_v, tile_c))
                                    add_cand(cands_v[d], rv);
                            } else {
                                add_cand(cands_v[d], compute_desired(oc, e.shift[d_from],
                                                                     e.stride[d_from], tile_v));
                            }
                        }
                    }
                }
            }

            for (int d = 0; d < nd; ++d) {
                if (cands_v[d].empty())
                    add_cand(cands_v[d], 0);
            }
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();
                auto ci = class_info.find(v);
                if (ci == class_info.end())
                    continue;
                int nd = ci->second.ndims;
                int tile_v = ci->second.tile;
                if (tile_v <= 0)
                    tile_v = 1;

                for (int c : tree_children[v]) {
                    auto cc = class_info.find(c);
                    if (cc == class_info.end())
                        continue;
                    int tile_c = cc->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    for (auto& e : adj[v]) {
                        if (e.neighbor != c)
                            continue;
                        for (int d = 0; d < nd; ++d) {
                            int d_c = e.neighbor_dim[d];
                            if (d_c < 0)
                                continue;
                            int d_from = e.forward ? d : d_c;
                            for (int ov : cands[v][d]) {
                                if (e.forward) {
                                    add_cand(cands[c][d_c],
                                             compute_desired(ov, e.shift[d_from],
                                                             e.stride[d_from], tile_c));
                                } else {
                                    for (int rc : reverse_desired(ov, e.shift[d_from],
                                                                  e.stride[d_from], tile_c, tile_v))
                                        add_cand(cands[c][d_c], rc);
                                }
                            }
                        }
                    }
                    work.push(c);
                }
            }
        }

        std::unordered_map<int, std::array<std::vector<int64_t>, 3>> dp;

        for (int v : post_order) {
            auto ci = class_info.find(v);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;
            int tile_v = ci->second.tile;
            if (tile_v <= 0)
                tile_v = 1;

            auto& dp_v = dp[v];
            auto& cands_v = cands[v];

            for (int d = 0; d < 3; ++d) {
                int nc = (int)cands_v[d].size();
                dp_v[d].assign(nc, 0);
                if (d < nd && ci->second.has_fixed_offset) {
                    int fixed = positive_mod(ci->second.fixed_offset[d], tile_v);
                    for (int i = 0; i < nc; ++i)
                        dp_v[d][i] = (cands_v[d][i] == fixed) ? 0 : INF;
                }
            }

            for (int c : tree_children[v]) {
                auto cc = class_info.find(c);
                if (cc == class_info.end())
                    continue;
                int tile_c = cc->second.tile;
                if (tile_c <= 0)
                    tile_c = 1;

                std::vector<const AdjEdge*> edges_vc;
                for (auto& e : adj[v])
                    if (e.neighbor == c)
                        edges_vc.push_back(&e);

                auto& cands_c = cands[c];
                auto& dp_c = dp[c];

                for (int d = 0; d < nd; ++d) {
                    int d_c = d;
                    for (auto* ep : edges_vc) {
                        if (ep->neighbor_dim[d] != d) {
                            d_c = ep->neighbor_dim[d];
                            break;
                        }
                    }
                    if (d_c < 0)
                        continue;

                    int nc_v = (int)cands_v[d].size();
                    int nc_cd = (int)cands_c[d_c].size();

                    std::vector<int64_t> h(nc_v, INF);
                    for (int iv = 0; iv < nc_v; ++iv) {
                        int ov = cands_v[d][iv];
                        for (int ic = 0; ic < nc_cd; ++ic) {
                            int oc = cands_c[d_c][ic];
                            int64_t cost = dp_c[d_c][ic];
                            if (cost >= INF)
                                continue;
                            for (auto* ep : edges_vc) {
                                int ep_dc = ep->neighbor_dim[d];
                                if (ep_dc < 0)
                                    continue;
                                int d_from = ep->forward ? d : ep_dc;
                                int o_from, tile_to, o_to;
                                if (ep->forward) {
                                    o_from = ov;
                                    tile_to = tile_c;
                                    o_to = oc;
                                } else {
                                    o_from = oc;
                                    tile_to = tile_v;
                                    o_to = ov;
                                }
                                int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                int desired = positive_mod(
                                    (o_from + ep->shift[d_from]) / step, tile_to);
                                cost += ep->weight * circ_dist(desired, o_to, tile_to);
                                if (cost >= INF)
                                    break;
                            }
                            if (cost < h[iv])
                                h[iv] = cost;
                        }
                    }
                    for (int iv = 0; iv < nc_v; ++iv) {
                        if (dp_v[d][iv] < INF && h[iv] < INF)
                            dp_v[d][iv] += h[iv];
                        else
                            dp_v[d][iv] = INF;
                    }
                }
            }
        }

        std::unordered_map<int, std::array<int, 3>> class_phase;

        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;
            auto ci = class_info.find(root);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;

            std::array<int, 3> best = {0, 0, 0};
            for (int d = 0; d < nd; ++d) {
                int64_t best_cost = INF;
                auto& cv = cands[root][d];
                auto& dv = dp[root][d];
                for (int i = 0; i < (int)cv.size(); ++i) {
                    if (dv[i] < best_cost) {
                        best_cost = dv[i];
                        best[d] = cv[i];
                    }
                }
            }
            class_phase[root] = best;
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();
                auto ci_v = class_info.find(v);
                if (ci_v == class_info.end())
                    continue;
                int nd = ci_v->second.ndims;
                int tile_v = ci_v->second.tile;
                if (tile_v <= 0)
                    tile_v = 1;

                for (int c : tree_children[v]) {
                    auto ci_c = class_info.find(c);
                    if (ci_c == class_info.end())
                        continue;
                    int tile_c = ci_c->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    std::vector<const AdjEdge*> edges_vc;
                    for (auto& e : adj[v])
                        if (e.neighbor == c)
                            edges_vc.push_back(&e);

                    auto& cands_c = cands[c];
                    auto& dp_c = dp[c];
                    int nd_c = ci_c->second.ndims;

                    std::array<int, 3> best_c = {0, 0, 0};
                    for (int d_c = 0; d_c < nd_c; ++d_c) {
                        int ov = -1;
                        int d_parent = -1;
                        for (int dp_d = 0; dp_d < nd; ++dp_d) {
                            for (auto* ep : edges_vc) {
                                if (ep->neighbor_dim[dp_d] == d_c) {
                                    d_parent = dp_d;
                                    ov = class_phase[v][dp_d];
                                    break;
                                }
                            }
                            if (d_parent >= 0)
                                break;
                        }

                        if (ov < 0) {
                            int64_t best_cost = INF;
                            for (int ic = 0; ic < (int)cands_c[d_c].size(); ++ic) {
                                if (dp_c[d_c][ic] < best_cost) {
                                    best_cost = dp_c[d_c][ic];
                                    best_c[d_c] = cands_c[d_c][ic];
                                }
                            }
                            continue;
                        }

                        int64_t best_cost = INF;
                        int nc_cd = (int)cands_c[d_c].size();
                        for (int ic = 0; ic < nc_cd; ++ic) {
                            int oc = cands_c[d_c][ic];
                            int64_t cost = dp_c[d_c][ic];
                            if (cost >= INF)
                                continue;
                            for (auto* ep : edges_vc) {
                                int ep_dc = ep->neighbor_dim[d_parent];
                                if (ep_dc < 0 || ep_dc != d_c)
                                    continue;
                                int d_from = ep->forward ? d_parent : d_c;
                                int o_from, tile_to, o_to;
                                if (ep->forward) {
                                    o_from = ov;
                                    tile_to = tile_c;
                                    o_to = oc;
                                } else {
                                    o_from = oc;
                                    tile_to = tile_v;
                                    o_to = ov;
                                }
                                int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                int desired = positive_mod(
                                    (o_from + ep->shift[d_from]) / step, tile_to);
                                cost += ep->weight * circ_dist(desired, o_to, tile_to);
                                if (cost >= INF)
                                    break;
                            }
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_c[d_c] = oc;
                            }
                        }
                    }
                    class_phase[c] = best_c;
                    work.push(c);
                }
            }
        }

        auto positive_mod64 = [](int64_t value, int mod) -> int {
            if (mod <= 0)
                return static_cast<int>(value);
            int64_t r = value % mod;
            if (r < 0)
                r += mod;
            return static_cast<int>(r);
        };

        auto abs_dist64 = [](int64_t a, int64_t b) -> int64_t {
            return (a >= b) ? (a - b) : (b - a);
        };

        auto nearest_congruent = [&](int64_t anchor, int phase, int tile) -> int {
            if (tile <= 0)
                return static_cast<int>(anchor);

            int target = positive_mod(phase, tile);
            int64_t lower = target;
            if (anchor >= target) {
                lower += ((anchor - target) / tile) * static_cast<int64_t>(tile);
            } else {
                lower -= ((target - anchor + tile - 1) / tile) * static_cast<int64_t>(tile);
            }
            int64_t upper = lower + tile;
            return (abs_dist64(upper, anchor) < abs_dist64(lower, anchor))
                       ? static_cast<int>(upper)
                       : static_cast<int>(lower);
        };

        auto reverse_absolute = [&](int64_t target_abs, int shift_d, int stride_d, int phase_from,
                                    int tile_from) -> int {
            int step = stride_d > 1 ? stride_d : 1;
            int64_t low = target_abs * step - shift_d;
            int64_t center2 = 2 * low + (step - 1);
            bool found = false;
            int64_t best = low;
            int64_t best_score = 0;
            int phase = positive_mod(phase_from, tile_from);

            for (int delta = 0; delta < step; ++delta) {
                int64_t cand = low + delta;
                if (positive_mod64(cand, tile_from) != phase)
                    continue;
                if ((cand + shift_d) / step != target_abs)
                    continue;
                int64_t score = abs_dist64(2 * cand, center2);
                if (!found || score < best_score || (score == best_score && cand < best)) {
                    found = true;
                    best = cand;
                    best_score = score;
                }
            }

            if (found)
                return static_cast<int>(best);

            return nearest_congruent(low, phase_from, tile_from);
        };

        struct AbsoluteProposal {
            int value = 0;
            int64_t weight = 0;
        };

        auto choose_absolute = [&](const std::vector<AbsoluteProposal>& proposals, int phase,
                                   int tile) -> int {
            std::vector<int> candidates;
            candidates.push_back(nearest_congruent(phase, phase, tile));
            for (auto const& p : proposals) {
                auto it = std::lower_bound(candidates.begin(), candidates.end(), p.value);
                if (it == candidates.end() || *it != p.value)
                    candidates.insert(it, p.value);
            }

            int best = candidates.front();
            int64_t best_cost = std::numeric_limits<int64_t>::max();
            for (int cand : candidates) {
                int64_t cost = 0;
                for (auto const& p : proposals)
                    cost += p.weight * abs_dist64(cand, p.value);
                if (cost < best_cost || (cost == best_cost && cand < best)) {
                    best_cost = cost;
                    best = cand;
                }
            }
            return best;
        };

        std::unordered_map<int, std::array<int, 3>> class_abs_offset;

        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;

            auto ci = class_info.find(root);
            auto phase_it = class_phase.find(root);
            if (ci == class_info.end() || phase_it == class_phase.end())
                continue;

            std::array<int, 3> root_abs = {0, 0, 0};
            int tile_root = ci->second.tile;
            if (tile_root <= 0)
                tile_root = 1;

            for (int d = 0; d < ci->second.ndims; ++d) {
                if (ci->second.has_fixed_offset)
                    root_abs[d] = ci->second.fixed_offset[d];
                else
                    root_abs[d] =
                        nearest_congruent(phase_it->second[d], phase_it->second[d], tile_root);
            }
            class_abs_offset[root] = root_abs;
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid) && class_abs_offset.count(root))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();

                auto ci_v = class_info.find(v);
                auto abs_v_it = class_abs_offset.find(v);
                if (ci_v == class_info.end() || abs_v_it == class_abs_offset.end())
                    continue;
                int nd_v = ci_v->second.ndims;

                for (int c : tree_children[v]) {
                    auto ci_c = class_info.find(c);
                    auto phase_c_it = class_phase.find(c);
                    if (ci_c == class_info.end() || phase_c_it == class_phase.end())
                        continue;

                    int tile_c = ci_c->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    std::vector<const AdjEdge*> edges_vc;
                    for (auto& e : adj[v])
                        if (e.neighbor == c)
                            edges_vc.push_back(&e);

                    std::array<int, 3> abs_c = {0, 0, 0};
                    for (int d_c = 0; d_c < ci_c->second.ndims; ++d_c) {
                        if (ci_c->second.has_fixed_offset) {
                            abs_c[d_c] = ci_c->second.fixed_offset[d_c];
                            continue;
                        }

                        std::vector<AbsoluteProposal> proposals;
                        for (auto* ep : edges_vc) {
                            for (int d_parent = 0; d_parent < nd_v; ++d_parent) {
                                if (ep->neighbor_dim[d_parent] != d_c)
                                    continue;

                                int value = 0;
                                if (ep->forward) {
                                    int d_from = d_parent;
                                    int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                    int64_t anchor =
                                        (static_cast<int64_t>(abs_v_it->second[d_parent]) +
                                         ep->shift[d_from]) /
                                        step;
                                    value =
                                        nearest_congruent(anchor, phase_c_it->second[d_c], tile_c);
                                } else {
                                    int d_from = d_c;
                                    value = reverse_absolute(abs_v_it->second[d_parent],
                                                             ep->shift[d_from],
                                                             ep->stride[d_from],
                                                             phase_c_it->second[d_c], tile_c);
                                }
                                proposals.push_back({value, ep->weight});
                            }
                        }

                        abs_c[d_c] =
                            choose_absolute(proposals, phase_c_it->second[d_c], tile_c);
                    }

                    class_abs_offset[c] = abs_c;
                    work.push(c);
                }
            }
        }

        for (auto& [name, meta] : array_meta) {
            if (meta.decomp_final)
                continue;
            if (matmul_arrays.count(name))
                continue;
            int rep = uf_find(name);
            auto comp_it = comp_id.find(rep);
            if (comp_it != comp_id.end() && cyclic_components.count(comp_it->second))
                continue;
            auto it = class_abs_offset.find(rep);
            if (it != class_abs_offset.end())
                meta.offset = it->second;
        }

        if (!cyclic_components.empty()) {
            struct DesiredOffset {
                std::array<int, 3> desired_phase;
                std::array<int, 3> desired_anchor;
                int64_t weight;
            };
            std::map<int, std::vector<DesiredOffset>> all_desired;

            auto collect_fallback_producer =
                [&](auto&& self, ASTNode* node, int out_name, int nd, int out_tile) -> void {
                if (node == nullptr || node->is_scalar || node->is_broadcast)
                    return;
                if (is_ast_leaf(node)) {
                    int src_name = node->result_name;
                    if (src_name == out_name)
                        return;
                    auto src_it = array_meta.find(src_name);
                    if (src_it == array_meta.end() || src_it->second.ndims != nd)
                        return;
                    if (out_tile <= 0)
                        return;
                    std::array<int, 3> desired_phase = {0, 0, 0};
                    std::array<int, 3> desired_anchor = {0, 0, 0};
                    for (int d = 0; d < nd; ++d) {
                        desired_anchor[d] = src_it->second.offset[d];
                        desired_phase[d] = positive_mod(src_it->second.offset[d], out_tile);
                    }
                    int64_t weight = 1;
                    for (int d = 0; d < nd; ++d)
                        weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                    all_desired[out_name].push_back({desired_phase, desired_anchor, weight});
                    return;
                }
                for (int i = 0; i < (int)node->operands.size(); ++i) {
                    ASTNode* child = node->operands[i];
                    if (child == nullptr || child->is_scalar || child->is_broadcast)
                        continue;
                    Region* child_region = node->get_operand_region(i);
                    if (is_ast_leaf(child) && child_region != nullptr && !child_region->is_global) {
                        int src_name = child->result_name;
                        if (src_name == out_name)
                            continue;
                        auto src_it = array_meta.find(src_name);
                        if (src_it == array_meta.end() || src_it->second.ndims != nd)
                            continue;
                        if (out_tile <= 0)
                            continue;
                        auto rstart = extract_region_start(child_region, nd);
                        auto rstep = extract_region_step(child_region, nd);
                        std::array<int, 3> desired_phase = {0, 0, 0};
                        std::array<int, 3> desired_anchor = {0, 0, 0};
                        for (int d = 0; d < nd; ++d) {
                            int step = (rstep[d] > 1) ? rstep[d] : 1;
                            int raw = src_it->second.offset[d] + rstart[d];
                            desired_anchor[d] = (step > 1) ? (raw / step) : raw;
                            desired_phase[d] = (step > 1)
                                                   ? positive_mod(raw / step, out_tile)
                                                   : positive_mod(raw, out_tile);
                        }
                        int64_t weight = 1;
                        for (int d = 0; d < nd; ++d)
                            weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                        all_desired[out_name].push_back(
                            {desired_phase, desired_anchor, weight});
                    } else {
                        self(self, child, out_name, nd, out_tile);
                    }
                }
            };

            auto collect_fallback_consumer =
                [&](auto&& self, ASTNode* node, int target_name, int nd,
                    const ArrayMetadata& target_meta, const std::array<int, 3>& region_start)
                -> void {
                if (node == nullptr || node->is_scalar || node->is_broadcast)
                    return;
                if (is_ast_leaf(node)) {
                    int src_name = node->result_name;
                    if (src_name == target_name)
                        return;
                    auto src_it = array_meta.find(src_name);
                    if (src_it == array_meta.end() || src_it->second.ndims != nd)
                        return;
                    int tile = src_it->second.tile;
                    if (tile <= 0)
                        return;
                    std::array<int, 3> desired_phase = {0, 0, 0};
                    std::array<int, 3> desired_anchor = {0, 0, 0};
                    for (int d = 0; d < nd; ++d) {
                        desired_anchor[d] = target_meta.offset[d] + region_start[d];
                        desired_phase[d] =
                            positive_mod(target_meta.offset[d] + region_start[d], tile);
                    }
                    int64_t weight = 1;
                    for (int d = 0; d < nd; ++d)
                        weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                    all_desired[src_name].push_back({desired_phase, desired_anchor, weight});
                    return;
                }
                for (ASTNode* child : node->operands)
                    self(self, child, target_name, nd, target_meta, region_start);
            };

            auto is_cyclic_array = [&](int name) {
                int rep = uf_find(name);
                auto cit = comp_id.find(rep);
                return cit != comp_id.end() && cyclic_components.count(cit->second);
            };

            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                for (ASTNode* ast_node : dag_node->ast->roots) {
                    Opcode op = static_cast<Opcode>(ast_node->opcode);

                    if (op == Opcode::SET_REGION) {
                        if (ast_node->operands.size() < 2 || !ast_node->region)
                            continue;
                        int target_name = ast_node->operands[0]->result_name;
                        auto target_it = array_meta.find(target_name);
                        if (target_it == array_meta.end())
                            continue;
                        int nd = target_it->second.ndims;
                        auto region_start = extract_region_start(ast_node->region, nd);
                        for (int i = 1; i < (int)ast_node->operands.size(); ++i)
                            collect_fallback_consumer(collect_fallback_consumer,
                                                      ast_node->operands[i], target_name, nd,
                                                      target_it->second, region_start);
                        continue;
                    }

                    if (is_elementwise(op) || op == Opcode::COPY) {
                        int out_name = ast_node->result_name;
                        if (!is_cyclic_array(out_name))
                            continue;
                        auto out_it = array_meta.find(out_name);
                        if (out_it == array_meta.end())
                            continue;
                        int nd = out_it->second.ndims;
                        int out_tile = out_it->second.tile;
                        for (int i = 0; i < (int)ast_node->operands.size(); ++i) {
                            ASTNode* operand = ast_node->operands[i];
                            if (!operand || operand->is_scalar || operand->is_broadcast)
                                continue;
                            collect_fallback_producer(collect_fallback_producer, operand, out_name,
                                                      nd, out_tile);
                        }
                    }
                }
            });

            for (auto& [name, constraints] : all_desired) {
                if (constraints.empty())
                    continue;
                if (!is_cyclic_array(name))
                    continue;
                auto meta_it = array_meta.find(name);
                if (meta_it == array_meta.end() || meta_it->second.decomp_final)
                    continue;
                if (matmul_arrays.count(name))
                    continue;

                int nd = meta_it->second.ndims;
                int tile = meta_it->second.tile;
                if (tile <= 0)
                    continue;

                std::array<int, 3> best_phase = {0, 0, 0};
                for (int d = 0; d < nd; ++d) {
                    std::vector<int> fb_cands;
                    fb_cands.push_back(0);
                    for (auto& c : constraints) {
                        int o = positive_mod(c.desired_phase[d], tile);
                        auto it2 = std::lower_bound(fb_cands.begin(), fb_cands.end(), o);
                        if (it2 == fb_cands.end() || *it2 != o)
                            fb_cands.insert(it2, o);
                    }

                    int64_t best_cost = std::numeric_limits<int64_t>::max();
                    int best_o = 0;
                    for (int o : fb_cands) {
                        int64_t cost = 0;
                        for (auto& c : constraints)
                            cost += c.weight * circ_dist(o, c.desired_phase[d], tile);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_o = o;
                        }
                    }
                    best_phase[d] = best_o;
                }

                std::array<int, 3> best_offset = {0, 0, 0};
                for (int d = 0; d < nd; ++d) {
                    std::vector<AbsoluteProposal> proposals;
                    for (auto& c : constraints) {
                        proposals.push_back(
                            {nearest_congruent(c.desired_anchor[d], best_phase[d], tile),
                             c.weight});
                    }
                    best_offset[d] = choose_absolute(proposals, best_phase[d], tile);
                }
                meta_it->second.offset = best_offset;
            }
        }

        // sources' offsets may have moved since
        align_transpose_results();

        for (int name : matmul_arrays) {
            auto meta_it = array_meta.find(name);
            if (meta_it != array_meta.end() && !meta_it->second.decomp_final)
                meta_it->second.offset = zero_offset;
        }

        // Keep each fused kernel's results in its chare layout where the fixed arrays
        // allow it; resolve_kernel_layouts() handles the rest.
        for (auto& [name, meta] : array_meta) {
            if (meta.decomp_final || meta.tile <= 0)
                continue;
            for (int d = 0; d < meta.ndims; ++d)
                if (meta.offset[d] < 0)
                    meta.offset[d] = positive_mod(meta.offset[d], meta.tile);
        }
        detail::tie_kernel_layouts(array_meta, dag, matmul_arrays);
        align_transpose_results();
    }

    // Chare indices are global / tile, so a negative offset (from a view that
    // starts left of its anchor) moves to the same tile phase in [0, tile).
    for (auto& [name, meta] : array_meta) {
        if (meta.decomp_final || meta.tile <= 0)
            continue;
        for (int d = 0; d < meta.ndims; ++d)
            if (meta.offset[d] < 0)
                meta.offset[d] = ((meta.offset[d] % meta.tile) + meta.tile) % meta.tile;
    }

    for (auto& [name, meta] : array_meta)
        meta.decomp_final = true;
}


namespace detail {

inline uint64_t mix64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= 0xbf58476d1ce4e5b9ULL;
    return h ^ (h >> 29);
}

inline uint64_t mix_box(uint64_t h, Region* region, int nd) {
    Box b = region_box(region, nd, {0, 0, 0});
    h = mix64(h, region == nullptr ? 1 : region->is_global ? 2 : 3);
    for (int d = 0; d < nd; ++d) {
        h = mix64(h, (uint32_t)b.start[d]);
        h = mix64(h, (uint32_t)b.extent[d]);
        h = mix64(h, (uint32_t)b.step[d]);
    }
    return h;
}

/// Identifier of a kernel built here (the frontend hashes the structure, regions included).
inline int64_t synth_identifier(uint64_t seed, AST* ast) {
    uint64_t h = mix64(0x6c61796f75747321ULL, seed);
    for (ASTNode* root : ast->roots) {
        h = mix64(h, (uint32_t)root->opcode);
        h = mix64(h, (uint32_t)root->dtype);
        h = mix64(h, (uint32_t)root->ndims);
        h = mix64(h, root->is_temp);
        h = mix_box(h, root->region, root->ndims);
        for (size_t i = 0; i < root->operands.size(); ++i) {
            ASTNode* op = root->operands[i];
            h = mix64(h, op->is_scalar ? 1 : op->is_broadcast ? 2 : 3);
            h = mix64(h, (uint32_t)op->dtype);
            h = mix64(h, (uint32_t)op->opcode);
            h = mix64(h, (uint32_t)op->ndims);
            if (!op->is_scalar)
                h = mix_box(h, root->get_operand_region((int)i), op->ndims);
        }
    }
    return (int64_t)h;
}

inline Region* whole_region(int nd, std::array<int, 3> const& shape) {
    switch (nd) {
    case 1: return new ArrayRegion<1>({0}, {shape[0]}, {1});
    case 2: return new ArrayRegion<2>({0, 0}, {shape[0], shape[1]}, {1, 1});
    default: return new ArrayRegion<3>({0, 0, 0}, {shape[0], shape[1], shape[2]}, {1, 1, 1});
    }
}

/// Whether a fused kernel without set_regions keeps results of different shapes. The
/// executor writes them all over its anchor's iteration space, and R = X cannot reshape.
template <typename ArrayMetaMap>
bool kept_shapes_differ(ArrayMetaMap const& meta, DAGNode* node) {
    KernelLayout k;
    if (!is_fused_kernel(node) || !kernel_layout(meta, node, k) || k.anchor_set != nullptr)
        return false;
    auto const& am = meta.at(k.anchor);
    for (ASTNode* root : node->ast->roots) {
        if (!is_kept_root(root))
            continue;
        auto it = meta.find(root->result_name);
        if (it == meta.end())
            continue;
        if (it->second.ndims != am.ndims)
            return true;
        for (int d = 0; d < am.ndims; ++d)
            if (it->second.global_shape[d] != am.global_shape[d])
                return true;
    }
    return false;
}

} // namespace detail

struct LayoutFixResult {
    int follow_up_nodes = 0;
    std::string error;
};

/// Makes every fused kernel's outputs match the layout the executor writes them in,
/// after all decompositions are final (tie_kernel_layouts leaves only conflicts with
/// fixed arrays). A kept result R in another layout becomes a result X in the kernel's
/// layout, then a follow-up node R = X redistributes it; if the kernel has set_regions
/// and anything disagrees, every set_region T[r] = v becomes X = v, then T[r] = X in a
/// follow-up node. Follow-ups run between the kernel and its children, so reads in the
/// kernel still see the values from before its writes; the R = X copies run before the
/// T[r] = X writes, since a kernel writes R before any set_region into it. X names and
/// follow-up node ids count down from next_name / next_node_id.
template <typename ArrayMetaMap>
LayoutFixResult resolve_kernel_layouts(ArrayMetaMap& meta, DAG* dag, int& next_name,
                                       int& next_node_id) {
    using Meta = typename ArrayMetaMap::mapped_type;
    LayoutFixResult res;
    std::vector<DAGNode*> order;
    detail::for_each_node_topo(dag, [&](DAGNode* n) { order.push_back(n); });

    for (DAGNode* node : order) {
        if (!detail::is_fused_kernel(node))
            continue;
        if (detail::kept_shapes_differ(meta, node)) {
            res.error = "fused kernel (node " + std::to_string(node->id) +
                        ") keeps results of different shapes";
            return res;
        }
        auto& roots = node->ast->roots;
        // (follow-up root, X it retires or -1): whole-result copies, then set_regions
        std::vector<std::pair<ASTNode*, int>> copies, sets;
        std::unordered_set<int> synthetic;
        bool rebuilt = false;
        auto new_meta = [&](int x, Meta base, int nd, std::array<int, 3> extent,
                            std::array<int, 3> offset) {
            base.ndims = nd;
            base.global_shape = {0, 0, 0};
            base.offset = {0, 0, 0};
            for (int d = 0; d < nd; ++d) {
                base.global_shape[d] = extent[d];
                base.offset[d] = offset[d];
            }
            base.decomp_final = true;
            meta[x] = base;
        };
        auto x_leaf = [&](int x, ASTNode* like) {
            auto* leaf = new ASTNode(x, static_cast<int>(Opcode::NOOP));
            leaf->ndims = like->ndims;
            leaf->dtype = like->dtype;
            return leaf;
        };

        detail::KernelLayout k;
        if (!detail::kernel_layout(meta, node, k))
            continue;
        if (k.anchor_set != nullptr) {
            bool has_data = false, conflict = false;
            Meta const& am = meta.at(k.anchor);
            for (ASTNode* root : roots) {
                auto op = static_cast<Opcode>(root->opcode);
                if (detail::is_kept_root(root)) {
                    has_data = true;
                    auto it = meta.find(root->result_name);
                    if (it == meta.end())
                        continue;
                    detail::Box b = detail::region_box(nullptr, it->second.ndims,
                                                       it->second.global_shape);
                    conflict |= !detail::same_layout(it->second, {0, 0, 0}, b, am, k);
                } else if (op == Opcode::SET_REGION) {
                    ASTNode* rhs = root->operands[1];
                    has_data |= !rhs->is_scalar && !rhs->is_broadcast;
                    if (root == k.anchor_set)
                        continue;
                    Meta const& tm = meta.at(root->operands[0]->result_name);
                    detail::Box b = detail::region_box(root->region, tm.ndims, tm.global_shape);
                    conflict |= !detail::same_layout(tm, b.start, b, am, k);
                }
            }
            if (!has_data || !conflict)
                continue;
            for (size_t i = 0; i < roots.size(); ++i) {
                ASTNode* s = roots[i];
                if (static_cast<Opcode>(s->opcode) != Opcode::SET_REGION)
                    continue;
                ASTNode* target = s->operands[0];
                ASTNode* rhs = s->operands[1];
                Meta const& tm = meta.at(target->result_name);
                detail::Box b = detail::region_box(s->region, tm.ndims, tm.global_shape);
                auto* fs = new ASTNode(s->result_name, s->opcode, false, false, 0.0, s->ndims,
                                       s->dtype, false);
                fs->region = s->region;
                fs->add_operand(target);
                fs->operand_regions.push_back(s->get_operand_region(0));
                int x = -1;
                if (rhs->is_scalar || rhs->is_broadcast) {
                    fs->add_operand(rhs);
                    fs->operand_regions.push_back(s->get_operand_region(1));
                    roots.erase(roots.begin() + i--);
                } else {
                    x = next_name--;
                    auto* xr = new ASTNode(x, static_cast<int>(Opcode::CONVERT), false, false,
                                           0.0, s->ndims, target->dtype, false);
                    xr->add_operand(rhs);
                    xr->operand_regions.push_back(s->get_operand_region(1));
                    roots[i] = xr;
                    std::array<int, 3> off = {0, 0, 0};
                    for (int d = 0; d < tm.ndims; ++d)
                        off[d] = b.unit ? tm.offset[d] + b.start[d] : 0;
                    new_meta(x, tm, tm.ndims, b.extent, off);
                    synthetic.insert(x);
                    fs->add_operand(x_leaf(x, xr));
                    fs->operand_regions.push_back(detail::whole_region(tm.ndims, b.extent));
                }
                sets.push_back({fs, x});
            }
            rebuilt = true;
        }

        bool has_set = false;
        std::vector<ASTNode*> kept;
        for (ASTNode* root : roots) {
            has_set |= static_cast<Opcode>(root->opcode) == Opcode::SET_REGION;
            if (detail::is_kept_root(root) && meta.count(root->result_name))
                kept.push_back(root);
        }
        if (!has_set && !kept.empty()) {
            int ref = kept[0]->result_name;
            for (ASTNode* r : kept)
                if (!synthetic.count(r->result_name)) {
                    ref = r->result_name;
                    break;
                }
            for (int x : synthetic) {
                Meta& xm = meta.at(x);
                Meta const& rm = meta.at(ref);
                if (xm.ndims == rm.ndims && xm.global_shape == rm.global_shape) {
                    xm.tile = rm.tile;
                    xm.offset = rm.offset;
                }
            }
            detail::KernelLayout ka;
            detail::kernel_layout(meta, node, ka);
            Meta const am = meta.at(ka.anchor);
            for (ASTNode* r : kept) {
                int name = r->result_name;
                if (name == ka.anchor)
                    continue;
                Meta const rm = meta.at(name);
                detail::Box b = detail::region_box(nullptr, rm.ndims, rm.global_shape);
                if (detail::same_layout(rm, {0, 0, 0}, b, am, ka))
                    continue;
                // R is written as X in the kernel's layout, then redistributed
                int x = next_name--;
                new_meta(x, rm, rm.ndims, rm.global_shape, am.offset);
                meta[x].tile = am.tile;
                r->result_name = x;
                for (ASTNode* other : roots)
                    for (ASTNode*& op : other->operands)
                        if (op->result_name == name && !op->is_scalar &&
                            static_cast<Opcode>(op->opcode) == Opcode::NOOP) {
                            auto* leaf = new ASTNode(x, static_cast<int>(Opcode::NOOP));
                            leaf->is_broadcast = op->is_broadcast;
                            leaf->ndims = op->ndims;
                            leaf->dtype = op->dtype;
                            op = leaf;
                        }
                auto* fr = new ASTNode(name, static_cast<int>(Opcode::CONVERT), false, false,
                                       0.0, r->ndims, r->dtype, false);
                fr->add_operand(x_leaf(x, r));
                fr->operand_regions.push_back(detail::whole_region(rm.ndims, rm.global_shape));
                copies.push_back({fr, x});
            }
        }
        std::vector<std::pair<ASTNode*, int>> follow = std::move(copies);
        follow.insert(follow.end(), sets.begin(), sets.end());
        if (follow.empty())
            continue;

        node->ast->nodes.clear();
        for (ASTNode* root : roots)
            node->ast->nodes[root->result_name] = root;
        if (rebuilt)
            node->identifier = detail::synth_identifier((uint64_t)node->identifier, node->ast);

        std::vector<DAGNode*> children = std::move(node->children);
        std::vector<int> retire = std::move(node->retire_names);
        node->children.clear();
        node->retire_names.clear();
        auto pos = std::find(dag->node_order.begin(), dag->node_order.end(), node->id);
        DAGNode* prev = node;
        for (auto [root, x] : follow) {
            auto* f = new DAGNode(next_node_id--, new AST({root}));
            f->fusible = true;
            f->identifier = detail::synth_identifier(0, f->ast);
            prev->children.push_back(f);
            f->num_parents = 1;
            if (x >= 0) {
                dag->retire_remaining[x] = 1;
                f->retire_names.push_back(x);
            }
            dag->nodes[f->id] = f;
            if (pos != dag->node_order.end())
                pos = dag->node_order.insert(pos + 1, f->id);
            else
                dag->node_order.push_back(f->id);
            prev = f;
        }
        prev->children = std::move(children);
        prev->retire_names.insert(prev->retire_names.end(), retire.begin(), retire.end());
        dag->num_nodes = (int)dag->nodes.size();
        res.follow_up_nodes += (int)follow.size();
    }
    return res;
}

/// FNV-1a hash of the solved DAG (node ids, roots, their layouts) and the synthetic name
/// and node counters. resolve_kernel_layouts runs on every PE, so PEs must agree on it.
template <typename ArrayMetaMap>
uint64_t layout_hash(ArrayMetaMap const& meta, DAG* dag, int next_name, int next_node_id) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](int64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (uint64_t)(v >> (8 * b)) & 0xff;
            h *= 1099511628211ull;
        }
    };
    mix(next_name);
    mix(next_node_id);
    detail::for_each_node_topo(dag, [&](DAGNode* node) {
        mix(node->id);
        for (ASTNode* root : node->ast->roots) {
            mix(root->result_name);
            mix(root->opcode);
            auto it = meta.find(root->result_name);
            if (it == meta.end())
                continue;
            mix(it->second.tile);
            for (int d = 0; d < 3; ++d) {
                mix(it->second.offset[d]);
                mix(it->second.global_shape[d]);
            }
        }
    });
    return h;
}

} // namespace decomposition_solver
