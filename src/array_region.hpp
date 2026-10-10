#pragma once

#include "charm++.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <charmtyles/core/dag.hpp> // extract<T>
#include <charmtyles/core/region.hpp>
#include <cstdio>
#include <string>
#include <vector>

#ifndef DBG_PRINT
#ifdef NDEBUG
#define DBG_PRINT(...)
#else
#define DBG_PRINT(...) CkPrintf(__VA_ARGS__)
#endif
#endif

template <int N>
class ArrayRegion : public Region {
  public:
    std::array<int, N> start, stop, step;

    ArrayRegion(std::array<int, N> start_, std::array<int, N> stop_, std::array<int, N> step_)
        : start(start_), stop(stop_), step(step_) {}

    ArrayRegion() : start{}, stop{}, step{} {}

    ArrayRegion(bool is_global_) : start{}, stop{}, step{} { is_global = is_global_; }

    static constexpr int ndims() { return N; }

    static int positive_mod(int value, int mod) {
        int result = value % mod;
        return result < 0 ? result + mod : result;
    }

    static int gcd_int(int a, int b) {
        a = std::abs(a);
        b = std::abs(b);
        while (b != 0) {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    static long long extended_gcd(long long a, long long b, long long& x, long long& y) {
        if (b == 0) {
            x = 1;
            y = 0;
            return a;
        }
        long long x1 = 0, y1 = 0;
        long long g = extended_gcd(b, a % b, x1, y1);
        x = y1;
        y = x1 - (a / b) * y1;
        return g;
    }

    static int modular_inverse(int value, int mod) {
        long long x = 0, y = 0;
        long long g = extended_gcd(value, mod, x, y);
        (void)g;
        assert(g == 1 && "modular inverse requires coprime inputs");
        long long inv = x % mod;
        if (inv < 0)
            inv += mod;
        return static_cast<int>(inv);
    }

    static bool first_aligned_point(int start_a, int step_a, int start_b, int step_b, int lo,
                                    int hi, int& aligned, int& aligned_step) {
        int g = gcd_int(step_a, step_b);
        if (positive_mod(start_a - start_b, g) != 0)
            return false;

        long long reduced_a = step_a / g;
        long long reduced_b = step_b / g;
        long long diff = (start_b - start_a) / g;

        long long t0 = 0;
        if (reduced_b != 1) {
            long long a_mod = reduced_a % reduced_b;
            if (a_mod < 0)
                a_mod += reduced_b;
            t0 = (diff % reduced_b + reduced_b) % reduced_b;
            t0 = (t0 * modular_inverse(static_cast<int>(a_mod), static_cast<int>(reduced_b))) %
                 reduced_b;
        }

        long long first = static_cast<long long>(start_a) + static_cast<long long>(step_a) * t0;
        long long period = (static_cast<long long>(step_a) / g) * step_b;
        if (first < lo) {
            long long k = (static_cast<long long>(lo) - first + period - 1) / period;
            first += k * period;
        }
        if (first >= hi)
            return false;

        aligned = static_cast<int>(first);
        aligned_step = static_cast<int>(period);
        return true;
    }

    /// Number of elements along dimension d
    int size(int d) const { return (stop[d] - start[d] + step[d] - 1) / step[d]; }

    /// Total number of elements across all dimensions
    int size() const {
        int total = 1;
        for (int d = 0; d < N; ++d)
            total *= size(d);
        return total;
    }

    /// Deserialize from byte buffer (matches Python ArrayRegion.serialize)
    static ArrayRegion* deserialize(char*& msg) {
        int is_global = extract<int>(msg);
        if (is_global)
            return new ArrayRegion(true);
        int nd = extract<int>(msg);
        assert(nd == N && "Deserialized ndims does not match template parameter N");
        std::array<int, N> s{}, e{}, st{};
        for (int d = 0; d < N; ++d) {
            s[d] = extract<int>(msg);
            e[d] = extract<int>(msg);
            st[d] = extract<int>(msg);
        }
        return new ArrayRegion(s, e, st);
    }

    bool overlaps(ArrayRegion<N> const& other) const {
        if (is_global || other.is_global)
            return true;
        if (start == other.start && stop == other.stop && step == other.step)
            return true;

        for (int d = 0; d < N; ++d) {
            int lo = std::max(start[d], other.start[d]);
            int hi = std::min(stop[d], other.stop[d]);
            if (lo >= hi)
                return false;

            int aligned = 0;
            int aligned_step = 0;
            if (!first_aligned_point(start[d], step[d], other.start[d], other.step[d], lo, hi,
                                     aligned, aligned_step))
                return false;
        }

        return true;
    }

    bool covers(ArrayRegion<N> const& other) const {
        if (is_global)
            return true;
        if (other.is_global)
            return false;
        if (start == other.start && stop == other.stop && step == other.step)
            return true;

        for (int d = 0; d < N; ++d) {
            if (other.start[d] < start[d] || other.stop[d] > stop[d])
                return false;

            if (step[d] == 1)
                continue;

            if (positive_mod(other.start[d] - start[d], step[d]) != 0)
                return false;

            if (other.size(d) <= 1)
                continue;

            if (other.step[d] % step[d] != 0)
                return false;
        }

        return true;
    }

    bool intersect(ArrayRegion<N> const& other, ArrayRegion<N>& result) const {
        if (is_global && other.is_global) {
            result = other;
            return true;
        }
        if (is_global) {
            result = other;
            return true;
        }
        if (other.is_global) {
            result = *this;
            return true;
        }
        if (start == other.start && stop == other.stop && step == other.step) {
            result = *this;
            return true;
        }

        for (int d = 0; d < N; ++d) {
            int lo = std::max(start[d], other.start[d]);
            int hi = std::min(stop[d], other.stop[d]);
            if (lo >= hi)
                return false;

            int aligned = 0;
            int aligned_step = 0;
            if (!first_aligned_point(start[d], step[d], other.start[d], other.step[d], lo, hi,
                                     aligned, aligned_step))
                return false;

            result.start[d] = aligned;
            result.stop[d] = hi;
            result.step[d] = aligned_step;
        }

        return true;
    }

    bool overlaps(Region const& other) const override {
        auto other_region = dynamic_cast<ArrayRegion<N> const*>(&other);
        return other_region != nullptr && overlaps(*other_region);
    }

    bool covers(Region const& other) const override {
        auto other_region = dynamic_cast<ArrayRegion<N> const*>(&other);
        return other_region != nullptr && covers(*other_region);
    }

    bool intersect(Region const& other, Region& result) const override {
        auto other_region = dynamic_cast<ArrayRegion<N> const*>(&other);
        auto result_region = dynamic_cast<ArrayRegion<N>*>(&result);
        return other_region != nullptr && result_region != nullptr &&
               intersect(*other_region, *result_region);
    }

    void map_region(Region& other) override {}
};

template <int N>
ArrayRegion<N>* make_array_region_handle(const int* start, const int* stop, const int* step,
                                         bool is_global) {
    if (is_global)
        return new ArrayRegion<N>(true);

    std::array<int, N> s{};
    std::array<int, N> e{};
    std::array<int, N> st{};
    for (int d = 0; d < N; ++d) {
        s[d] = start[d];
        e[d] = stop[d];
        st[d] = step[d];
    }
    return new ArrayRegion<N>(s, e, st);
}

template <int N>
void delete_array_region_handle(void* ptr) {
    delete reinterpret_cast<ArrayRegion<N>*>(ptr);
}

template <int N>
bool overlaps_array_region_handle(void* lhs, void* rhs) {
    return reinterpret_cast<ArrayRegion<N>*>(lhs)->overlaps(*reinterpret_cast<ArrayRegion<N>*>(rhs));
}

template <int N>
bool covers_array_region_handle(void* lhs, void* rhs) {
    return reinterpret_cast<ArrayRegion<N>*>(lhs)->covers(*reinterpret_cast<ArrayRegion<N>*>(rhs));
}

template <int N>
bool intersect_array_region_handle(void* lhs, void* rhs, int* out_start, int* out_stop,
                                   int* out_step) {
    ArrayRegion<N> result;
    bool ok =
        reinterpret_cast<ArrayRegion<N>*>(lhs)->intersect(*reinterpret_cast<ArrayRegion<N>*>(rhs), result);
    if (!ok)
        return false;

    for (int d = 0; d < N; ++d) {
        out_start[d] = result.start[d];
        out_stop[d] = result.stop[d];
        out_step[d] = result.step[d];
    }
    return true;
}

template <int N>
class ArrayDecomp {
  public:
    std::array<int, N> offset;       // per-dim offset into global tile grid
    std::array<int, N> global_shape; // array shape in each dimension
    int tile;                        // tile size (same for all dims of this N)

    ArrayDecomp() : offset{}, global_shape{}, tile(0) {}

    /// Default decomposition (offset=0, standard tile).
    static ArrayDecomp default_decomp(std::array<int, N> shape, int tile_size) {
        ArrayDecomp d;
        d.offset = {};
        d.global_shape = shape;
        d.tile = tile_size;
        return d;
    }

    /// Offset decomposition.
    static ArrayDecomp offset_decomp(std::array<int, N> shape, std::array<int, N> off,
                                     int tile_size) {
        ArrayDecomp d;
        d.offset = off;
        d.global_shape = shape;
        d.tile = tile_size;
        return d;
    }

    /// Map local coordinate to global coordinate along dimension d.
    int to_global(int d, int local_coord) const { return local_coord + offset[d]; }

    /// Map global coordinate to local coordinate along dimension d.
    int to_local(int d, int global_coord) const { return global_coord - offset[d]; }

    /// Map a local-space region to global-space region.
    ArrayRegion<N> to_global(ArrayRegion<N> const& r) const {
        ArrayRegion<N> result;
        for (int d = 0; d < N; ++d) {
            result.start[d] = r.start[d] + offset[d];
            result.stop[d] = r.stop[d] + offset[d];
            result.step[d] = r.step[d];
        }
        return result;
    }

    /// Map a global-space region to local-space region.
    ArrayRegion<N> to_local(ArrayRegion<N> const& r) const {
        ArrayRegion<N> result;
        for (int d = 0; d < N; ++d) {
            result.start[d] = r.start[d] - offset[d];
            result.stop[d] = r.stop[d] - offset[d];
            result.step[d] = r.step[d];
        }
        return result;
    }

    /// Number of chares along dimension d for this array.
    int num_chares(int d) const {
        int gstart = offset[d];
        int gstop = offset[d] + global_shape[d];
        return (gstop + tile - 1) / tile - gstart / tile;
    }

    /// Global-space region owned by chare ci (clipped to this array's extent).
    ArrayRegion<N> chare_region_global(std::array<int, N> const& ci) const {
        ArrayRegion<N> r;
        for (int d = 0; d < N; ++d) {
            r.start[d] = std::max(ci[d] * tile, offset[d]);
            r.stop[d] = std::min((ci[d] + 1) * tile, offset[d] + global_shape[d]);
            r.step[d] = 1;
        }
        return r;
    }

    /// Local-space region owned by chare ci (clipped to this array's extent).
    ArrayRegion<N> chare_region_local(std::array<int, N> const& ci) const {
        return to_local(chare_region_global(ci));
    }

    /// Global start coordinate for chare ci along dimension d.
    int chare_start_global(int d, int ci_d) const {
        return std::max(ci_d * tile, offset[d]);
    }

    /// Index range [lo, hi) along axis d of the view (vstart, vstep, vsize; vstart in
    /// decomposition-global coords) that chare c holds; false if it is empty.
    bool view_block(int d, int c, int vstart, int vstep, int vsize, int& lo, int& hi) const {
        int cs = chare_start_global(d, c);
        int ce = std::min((c + 1) * tile, offset[d] + global_shape[d]);
        int a = cs - vstart;
        int b = ce - vstart;
        lo = a <= 0 ? 0 : (a + vstep - 1) / vstep;
        hi = b <= 0 ? 0 : std::min(vsize, (b + vstep - 1) / vstep);
        return lo < hi;
    }
};

/// Format an N-dimensional ArrayRegion as "[s0:e0, s1:e1, ...]"
template <int N>
std::string fmt_region(ArrayRegion<N> const& r) {
    std::string s = "[";
    for (int d = 0; d < N; ++d) {
        if (d)
            s += ", ";
        s += std::to_string(r.start[d]) + ":" + std::to_string(r.stop[d]);
        if (r.step[d] != 1)
            s += ":" + std::to_string(r.step[d]);
    }
    s += "]";
    return s;
}

/// N-D dynamic memref descriptor matching MLIR's LLVM ABI for memref<?x...xT>.
/// Layout: allocated, aligned, offset, sizes[N], strides[N].
template <int N, typename T = float>
struct MemRef {
    T* allocated;
    T* aligned;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];

    static constexpr int fields_per_memref() { return 3 + 2 * N; }
};

/// A fragment of data: a pointer to the buffer and the region it covers.
/// src_strides: actual strides of the source buffer (0 = packed, compute
/// from region dimensions).  Needed when the fragment points into a larger
/// 2D array whose row stride differs from the fragment's column count.
template <int N, typename T = float>
struct FragmentData {
    ArrayRegion<N> region;
    T* data;
    int64_t src_strides[N] = {};
};

/// An aligned fragment: the MemRef descriptor ready for kernel dispatch,
/// plus the region in global coordinates (needed for output offset computation).
template <int N, typename T = float>
struct AlignedMemRef {
    MemRef<N, T> memref;
    ArrayRegion<N> region;
};

template <int N>
class ChareIndex {
  public:
    int idx[N];

    bool operator==(ChareIndex const& other) const {
        for (int i = 0; i < N; ++i)
            if (idx[i] != other.idx[i])
                return false;
        return true;
    }
};

template <int N>
struct ChareIndexHash {
    std::size_t operator()(ChareIndex<N> const& ci) const {
        std::size_t h = 0;
        for (int i = 0; i < N; ++i)
            h ^= std::hash<int>()(ci.idx[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/// Intersection of two regions in the same coordinate space.
/// Returns {result, true} if the intersection is non-empty, {_, false} otherwise.
template <int N>
std::pair<ArrayRegion<N>, bool> intersect(ArrayRegion<N> const& r1, ArrayRegion<N> const& r2) {
    ArrayRegion<N> result;
    return {result, r1.intersect(r2, result)};
}

/// Subtract r2 from r1: returns the fragments of r1 not covered by r2.
/// Produces up to 2*N axis-aligned slabs by peeling one dimension at a time.
template <int N>
std::vector<ArrayRegion<N>> subtract(ArrayRegion<N> const& r1, ArrayRegion<N> const& r2) {
    auto [overlap, has_overlap] = intersect(r1, r2);
    if (!has_overlap)
        return {r1};

    std::vector<ArrayRegion<N>> fragments;

    // Current remainder starts as r1; we narrow it per dimension as we peel
    ArrayRegion<N> remainder = r1;

    for (int d = 0; d < N; ++d) {
        // For strided regions, overlap.stop may be an exclusive bound that is
        // not itself on remainder's lattice (for example [3:65:2] intersect
        // [0:64) => overlap [3:64:2]). When peeling the right slab, align the
        // covered stop back onto remainder's lattice so we do not synthesize
        // bogus fragments like [64:65:2] that contain no real source element.
        int covered_stop = overlap.stop[d];
        if (remainder.step[d] > 1)
            covered_stop = overlap.start[d] + overlap.size(d) * remainder.step[d];

        // Left slab: remainder up to the overlap start in dimension d
        if (remainder.start[d] < overlap.start[d]) {
            ArrayRegion<N> slab = remainder;
            slab.stop[d] = overlap.start[d];
            fragments.push_back(slab);
        }

        // Right slab: remainder from overlap stop in dimension d
        if (covered_stop < remainder.stop[d]) {
            ArrayRegion<N> slab = remainder;
            slab.start[d] = covered_stop;
            fragments.push_back(slab);
        }

        // Narrow remainder to the overlap range in this dimension
        remainder.start[d] = overlap.start[d];
        remainder.stop[d] = covered_stop;
    }

    return fragments;
}

/// Given sub-region r1 (a subset of parent1), find the corresponding
/// sub-region in parent2's coordinate space.  The i-th element of parent1
/// corresponds to the i-th element of parent2.
template <int N>
ArrayRegion<N> map(ArrayRegion<N> const& r1, ArrayRegion<N> const& parent1,
                   ArrayRegion<N> const& parent2) {
    ArrayRegion<N> result;

    for (int d = 0; d < N; ++d) {
        // Position of r1 within parent1 in logical coordinates.
        // Use the sub-region's element count to compute the mapped stop so
        // stepped regions preserve their full extent even when the exclusive
        // stop is not aligned to parent1.step (for example [64:127:2]).
        int lo = (r1.start[d] - parent1.start[d]) / parent1.step[d];
        int ls = r1.step[d] / parent1.step[d];
        int count = r1.size(d);

        // Translate to parent2's coordinate space
        result.start[d] = parent2.start[d] + lo * parent2.step[d];
        result.step[d] = ls * parent2.step[d];
        result.stop[d] = result.start[d] + count * result.step[d];
    }

    return result;
}

/// Result of local_inputs: the input sub-regions this chare needs, plus
/// how many remote messages to expect.
template <int N>
struct LocalInputs {
    std::vector<ArrayRegion<N>> my_inputs;
    int expected_msgs;
};

/// For a given output region and list of input regions, compute:
///   - the sub-region of each input needed to produce this chare's output
///   - how many messages to expect from remote chares for data not owned locally
///
/// All regions must be in global space.
/// output_decomp:  decomposition of the output array
/// input_decomps:  decomposition of each input array (one per input)
template <int N>
LocalInputs<N> local_inputs(ArrayRegion<N> const& r_out, std::vector<ArrayRegion<N>> const& inputs,
                            std::array<int, N> const& nd_idx,
                            ArrayDecomp<N> const& output_decomp,
                            std::vector<ArrayDecomp<N>> const& input_decomps) {
    LocalInputs<N> result;
    result.expected_msgs = 0;

    ArrayRegion<N> r_chare_out = output_decomp.chare_region_global(nd_idx);

    DBG_PRINT("  local_inputs: r_out=%s, r_chare_out=%s, %d inputs\n", fmt_region(r_out).c_str(),
              fmt_region(r_chare_out).c_str(), (int)inputs.size());

    // Portion of the output that belongs to this chare
    auto [r_myout, has_output] = intersect(r_out, r_chare_out);
    if (!has_output) {
        DBG_PRINT("  local_inputs: no output overlap -> returning empty\n");
        return result;
    }
    DBG_PRINT("  local_inputs: r_myout=%s\n", fmt_region(r_myout).c_str());

    for (int idx = 0; idx < (int)inputs.size(); ++idx) {
        auto const& r_inp = inputs[idx];
        ArrayRegion<N> r_chare_inp = input_decomps[idx].chare_region_global(nd_idx);
        DBG_PRINT("  local_inputs: input[%d] r_inp=%s r_chare_inp=%s\n", idx,
                  fmt_region(r_inp).c_str(), fmt_region(r_chare_inp).c_str());

        // Map my output slice to the corresponding input slice
        ArrayRegion<N> r_myinp = map(r_myout, r_out, r_inp);
        DBG_PRINT("  local_inputs: r_myinp (mapped)=%s\n", fmt_region(r_myinp).c_str());

        auto [local_input, has_local] = intersect(r_myinp, r_chare_inp);
        if (has_local)
            DBG_PRINT("  local_inputs: local_input=%s\n", fmt_region(local_input).c_str());
        else
            DBG_PRINT("  local_inputs: no local overlap\n");
        // Always push the full mapped input region so the fragment path
        // knows the complete input range (local + remote portions).
        result.my_inputs.push_back(r_myinp);

        // The part of my input that is NOT local — need remote messages
        auto remote_fragments = subtract(r_myinp, r_chare_inp);
        DBG_PRINT("  local_inputs: %d remote fragments\n", (int)remote_fragments.size());
        for (auto const& frag : remote_fragments) {
            DBG_PRINT("  local_inputs:   remote frag=%s\n", fmt_region(frag).c_str());
            auto chare_map = decompose(frag, input_decomps[idx]);
            DBG_PRINT("  local_inputs:   decomposed into %d chare(s)\n", (int)chare_map.size());
            result.expected_msgs += static_cast<int>(chare_map.size());
        }
    }

    DBG_PRINT("  local_inputs: total expected_msgs=%d, %d my_inputs\n", result.expected_msgs,
              (int)result.my_inputs.size());
    return result;
}

/// A message to send: the input sub-region destined for a remote chare.
template <int N>
struct RemoteSend {
    ChareIndex<N> target;
    ArrayRegion<N> region;
    int input_index; // which input this send corresponds to
};

/// For each input, determine what local data this chare must send to remote
/// chares that need it for their portion of the output.
///
/// All regions must be in global space.
/// For each input:
///   1. intersect(r_inp, r_chare_inp) — the part of this input that I own
///   2. map to output space — what output does my local input contribute to
///   3. subtract r_chare_out — the output pieces that belong to remote chares
///   4. decompose with output_decomp — which remote chares own each piece
///   5. map back to input space — the actual input sub-region to send
template <int N>
std::vector<RemoteSend<N>> send_remote_inputs(ArrayRegion<N> const& r_out,
                                              std::vector<ArrayRegion<N>> const& inputs,
                                              std::array<int, N> const& nd_idx,
                                              ArrayDecomp<N> const& output_decomp,
                                              std::vector<ArrayDecomp<N>> const& input_decomps) {
    std::vector<RemoteSend<N>> sends;

    ArrayRegion<N> r_chare_out = output_decomp.chare_region_global(nd_idx);

    for (int inp_idx = 0; inp_idx < (int)inputs.size(); ++inp_idx) {
        auto const& r_inp = inputs[inp_idx];
        ArrayRegion<N> r_chare_inp = input_decomps[inp_idx].chare_region_global(nd_idx);

        // Part of this input that I own
        auto [r_myinp, has_input] = intersect(r_inp, r_chare_inp);
        if (!has_input)
            continue;

        // What output does my local input contribute to
        ArrayRegion<N> r_myout = map(r_myinp, r_inp, r_out);

        // Output pieces that belong to remote chares (not me)
        auto remote_out_frags = subtract(r_myout, r_chare_out);
        for (auto const& frag : remote_out_frags) {
            // Which remote chares own each piece
            auto chare_map = decompose(frag, output_decomp);
            for (auto const& [index, r_outsend] : chare_map) {
                // Map the output piece back to input coordinates
                ArrayRegion<N> r_inpsend = map(r_outsend, r_out, r_inp);
                sends.push_back({index, r_inpsend, inp_idx});
            }
        }
    }

    return sends;
}

/// Common refinement of K inputs.  Each input is a list of disjoint
/// region fragments (with data pointers) whose combined shape is the same
/// across all inputs.  parents[k] is the parent region for input k (used to
/// convert to logical indices).  Returns K outputs where every output has
/// the same M fragments, each with matching shape across all K outputs.
/// Each returned MemRef has the correct data pointer and strides for a
/// zero-copy view into the original fragment's buffer.
///
/// Works by collecting all fragment boundaries (in logical index space) per
/// dimension across all inputs, then splitting every fragment at that grid.
template <int N, typename T = float>
std::vector<std::vector<AlignedMemRef<N, T>>>
align_fragments(std::vector<std::vector<FragmentData<N, T>>> const& inputs,
                std::vector<ArrayRegion<N>> const& parents) {
    if (inputs.empty())
        return {};
    int K = static_cast<int>(inputs.size());

    // 1. Collect all unique logical boundaries per dimension
    std::array<std::vector<int>, N> grid;

    for (int k = 0; k < K; ++k) {
        for (auto const& fd : inputs[k]) {
            for (int d = 0; d < N; ++d) {
                int gs = parents[k].start[d];
                int gst = parents[k].step[d];
                int lo = (fd.region.start[d] - gs) / gst;
                assert(fd.region.step[d] % gst == 0 &&
                       "align_fragments: fragment step must align with parent step");
                int logical_step = fd.region.step[d] / gst;
                int hi = lo + fd.region.size(d) * logical_step;
                grid[d].push_back(lo);
                grid[d].push_back(hi);
            }
        }
    }

    for (int d = 0; d < N; ++d) {
        std::sort(grid[d].begin(), grid[d].end());
        grid[d].erase(std::unique(grid[d].begin(), grid[d].end()), grid[d].end());
    }

    // 2. For each input, split every fragment at the grid boundaries
    std::vector<std::vector<AlignedMemRef<N, T>>> result(K);

    for (int k = 0; k < K; ++k) {
        for (auto const& fd : inputs[k]) {
            auto const& frag = fd.region;
            int gs[N], gst[N], frag_lo[N], frag_hi[N];
            for (int d = 0; d < N; ++d) {
                gs[d] = parents[k].start[d];
                gst[d] = parents[k].step[d];
                frag_lo[d] = (frag.start[d] - gs[d]) / gst[d];
                assert(frag.step[d] % gst[d] == 0 &&
                       "align_fragments: fragment step must align with parent step");
                int logical_step = frag.step[d] / gst[d];
                frag_hi[d] = frag_lo[d] + frag.size(d) * logical_step;
            }

            // Use explicit source strides if provided, otherwise compute
            // row-major packed strides from the fragment dimensions.
            int64_t orig_strides[N];
            bool has_explicit = false;
            for (int d = 0; d < N; ++d)
                if (fd.src_strides[d] != 0) {
                    has_explicit = true;
                    break;
                }
            if (has_explicit) {
                for (int d = 0; d < N; ++d)
                    orig_strides[d] = fd.src_strides[d];
            } else {
                orig_strides[N - 1] = 1;
                for (int d = N - 2; d >= 0; --d)
                    orig_strides[d] = orig_strides[d + 1] * frag.size(d + 1);
            }

            // Find grid index range per dimension
            std::array<int, N> i_lo, i_hi;
            for (int d = 0; d < N; ++d) {
                i_lo[d] = static_cast<int>(
                    std::lower_bound(grid[d].begin(), grid[d].end(), frag_lo[d]) - grid[d].begin());
                i_hi[d] = static_cast<int>(
                    std::lower_bound(grid[d].begin(), grid[d].end(), frag_hi[d]) - grid[d].begin());
            }

            // Iterate over all grid cells within this fragment (odometer)
            std::array<int, N> idx;
            for (int d = 0; d < N; ++d)
                idx[d] = i_lo[d];

            while (true) {
                ArrayRegion<N> cell;
                for (int d = 0; d < N; ++d) {
                    cell.start[d] = gs[d] + grid[d][idx[d]] * gst[d];
                    cell.stop[d] = gs[d] + grid[d][idx[d] + 1] * gst[d];
                    cell.step[d] = frag.step[d];
                }

                // Compute data pointer offset into the original fragment
                int64_t data_offset = 0;
                for (int d = 0; d < N; ++d) {
                    int delta = cell.start[d] - frag.start[d];
                    int logical_delta = delta / frag.step[d];
                    data_offset += (has_explicit ? delta : logical_delta) * orig_strides[d];
                }
                T* ptr = fd.data + data_offset;

                MemRef<N, T> mr;
                mr.allocated = ptr;
                mr.aligned = ptr;
                mr.offset = 0;
                for (int d = 0; d < N; ++d) {
                    mr.sizes[d] = cell.size(d);
                    mr.strides[d] = has_explicit ? orig_strides[d] * cell.step[d]
                                                 : orig_strides[d];
                }
                result[k].push_back({mr, cell});

                // Advance odometer
                int d = N - 1;
                while (d >= 0) {
                    idx[d]++;
                    if (idx[d] < i_hi[d])
                        break;
                    idx[d] = i_lo[d];
                    --d;
                }
                if (d < 0)
                    break;
            }
        }
    }

    // Sort each input's sub-fragments by logical cell index to ensure
    // consistent ordering across all inputs.  Different inputs may have
    // different original fragments (local vs remote) that cover different
    // regions, so the odometer traversal can produce sub-cells in different
    // orders.  Sorting by logical position makes the pairing deterministic.
    for (int k = 0; k < K; ++k) {
        std::sort(result[k].begin(), result[k].end(),
                  [&](AlignedMemRef<N, T> const& a, AlignedMemRef<N, T> const& b) {
                      for (int d = 0; d < N; ++d) {
                          int la = (a.region.start[d] - parents[k].start[d]) / parents[k].step[d];
                          int lb = (b.region.start[d] - parents[k].start[d]) / parents[k].step[d];
                          if (la != lb)
                              return la < lb;
                      }
                      return false;
                  });
    }

    return result;
}

/// Decompose a global-space region into per-chare sub-regions.
/// Uses the ArrayDecomp's tile size for chare boundaries.
template <int N>
std::unordered_map<ChareIndex<N>, ArrayRegion<N>, ChareIndexHash<N>>
decompose(ArrayRegion<N> const& region, ArrayDecomp<N> const& decomp) {
    std::unordered_map<ChareIndex<N>, ArrayRegion<N>, ChareIndexHash<N>> decomposition;

    // Compute chare index ranges per dimension
    std::array<int, N> chare_start, chare_stop;
    for (int d = 0; d < N; ++d) {
        chare_start[d] = region.start[d] / decomp.tile;
        chare_stop[d] = (region.stop[d] + decomp.tile - 1) / decomp.tile;
    }

    // Iterate over all chare indices in the N-dimensional range
    ChareIndex<N> ci{};
    for (int d = 0; d < N; ++d)
        ci.idx[d] = chare_start[d];

    while (true) {
        // Intersect with the actual chare region so stepped fragments stay on
        // the source lattice and offset decompositions are respected. Raw
        // max/min clipping can synthesize bogus fragments at tile boundaries
        // (for example [65:129:2] clipped to a [128:129) sliver).
        std::array<int, N> ci_arr{};
        for (int d = 0; d < N; ++d)
            ci_arr[d] = ci.idx[d];
        ArrayRegion<N> chare_reg = decomp.chare_region_global(ci_arr);
        auto [subreg, valid] = intersect(region, chare_reg);
        if (valid)
            decomposition[ci] = subreg;

        // Advance the N-dimensional chare index (odometer-style)
        int d = N - 1;
        while (d >= 0) {
            ci.idx[d]++;
            if (ci.idx[d] < chare_stop[d])
                break;
            ci.idx[d] = chare_start[d];
            --d;
        }
        if (d < 0)
            break;
    }

    return decomposition;
}
