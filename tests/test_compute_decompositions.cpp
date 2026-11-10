#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define DBG_PRINT(...)
#include "decomposition_solver.hpp"

namespace {

extern "C" void CmiAbort(const char*, ...) {
    std::abort();
}

struct TestArrayMetadata {
    int ndims = 0;
    std::array<int, 3> global_shape = {0, 0, 0};
    std::array<int, 3> offset = {0, 0, 0};
    int tile = 0;
    bool decomp_final = false;
};

using MetaMap = std::unordered_map<int, TestArrayMetadata>;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect_eq(const std::string& label, int got, int want) {
    if (got == want)
        return;
    std::ostringstream oss;
    oss << label << ": expected " << want << ", got " << got;
    fail(oss.str());
}

void expect_true(const std::string& label, bool value) {
    if (!value)
        fail(label);
}

void expect_offset(const std::string& label, const std::array<int, 3>& got,
                   const std::array<int, 3>& want) {
    if (got == want)
        return;
    std::ostringstream oss;
    oss << label << ": expected (" << want[0] << ", " << want[1] << ", " << want[2]
        << "), got (" << got[0] << ", " << got[1] << ", " << got[2] << ")";
    fail(oss.str());
}

ASTNode* make_leaf(int name, int ndims = 1) {
    return new ASTNode(name, static_cast<int>(Opcode::NOOP),
                       /*is_temp=*/false, /*is_scalar=*/false,
                       /*scalar=*/0.0, ndims);
}

ASTNode* make_copy(int result_name, ASTNode* src, Region* src_region = nullptr,
                   bool is_temp = false, int ndims = 1) {
    auto* node = new ASTNode(result_name, static_cast<int>(Opcode::COPY),
                             is_temp, false, 0.0, ndims);
    node->add_operand(src);
    node->operand_regions.push_back(src_region);
    return node;
}

ASTNode* make_add(int result_name, ASTNode* lhs, Region* lhs_region,
                  ASTNode* rhs, Region* rhs_region, bool is_temp = false, int ndims = 1) {
    auto* node = new ASTNode(result_name, static_cast<int>(Opcode::ADD),
                             is_temp, false, 0.0, ndims);
    node->add_operand(lhs);
    node->operand_regions.push_back(lhs_region);
    node->add_operand(rhs);
    node->operand_regions.push_back(rhs_region);
    return node;
}

ASTNode* make_set_region(int result_name, ASTNode* dst, Region* dst_region, ASTNode* src,
                         int ndims = 1) {
    auto* node = new ASTNode(result_name, static_cast<int>(Opcode::SET_REGION),
                             /*is_temp=*/false, /*is_scalar=*/false,
                             /*scalar=*/0.0, ndims);
    node->add_operand(dst);
    node->add_operand(src);
    node->region = dst_region;
    return node;
}

template <int N>
ArrayRegion<N>* make_region(std::array<int, N> start, std::array<int, N> stop,
                            std::array<int, N> step) {
    return new ArrayRegion<N>(start, stop, step);
}

DAG* make_single_node_dag(std::vector<ASTNode*> roots) {
    auto* ast = new AST(std::move(roots));
    auto* node = new DAGNode(0, ast);
    auto* dag = new DAG();
    dag->roots.push_back(node);
    dag->nodes[0] = node;
    dag->num_nodes = 1;
    return dag;
}

TestArrayMetadata make_meta(int ndims, int size0, int offset0, int tile, bool final) {
    TestArrayMetadata meta;
    meta.ndims = ndims;
    meta.global_shape = {size0, 0, 0};
    meta.offset = {offset0, 0, 0};
    meta.tile = tile;
    meta.decomp_final = final;
    return meta;
}

TestArrayMetadata make_meta(int ndims, std::array<int, 3> shape, std::array<int, 3> offset,
                            int tile, bool final) {
    TestArrayMetadata meta;
    meta.ndims = ndims;
    meta.global_shape = shape;
    meta.offset = offset;
    meta.tile = tile;
    meta.decomp_final = final;
    return meta;
}

void test_shifted_copy_uses_absolute_offset() {
    std::cout << "=== test_shifted_copy_uses_absolute_offset ===" << std::endl;

    constexpr int A = 1;
    constexpr int B = 2;

    MetaMap meta;
    meta[A] = make_meta(/*ndims=*/1, /*size0=*/512, /*offset0=*/0, /*tile=*/64, /*final=*/true);
    meta[B] = make_meta(/*ndims=*/1, /*size0=*/200, /*offset0=*/0, /*tile=*/0, /*final=*/false);

    ASTNode* copy = make_copy(B, make_leaf(A),
                              make_region<1>({150}, {350}, {1}),
                              /*is_temp=*/false, /*ndims=*/1);
    DAG* dag = make_single_node_dag({copy});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("B.tile", meta.at(B).tile, 64);
    expect_eq("B.offset[0]", meta.at(B).offset[0], 150);
    expect_eq("B.phase", meta.at(B).offset[0] % meta.at(B).tile, 22);
    expect_true("B.decomp_final", meta.at(B).decomp_final);

    delete dag;
}

void test_shifted_copy_from_nonzero_source_offset() {
    std::cout << "=== test_shifted_copy_from_nonzero_source_offset ===" << std::endl;

    constexpr int A = 1;
    constexpr int B = 2;

    MetaMap meta;
    meta[A] = make_meta(/*ndims=*/1, /*size0=*/512, /*offset0=*/132, /*tile=*/64, /*final=*/true);
    meta[B] = make_meta(/*ndims=*/1, /*size0=*/100, /*offset0=*/0, /*tile=*/0, /*final=*/false);

    ASTNode* copy = make_copy(B, make_leaf(A),
                              make_region<1>({70}, {170}, {1}),
                              /*is_temp=*/false, /*ndims=*/1);
    DAG* dag = make_single_node_dag({copy});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("B.tile", meta.at(B).tile, 64);
    expect_eq("B.offset[0]", meta.at(B).offset[0], 202);
    expect_eq("B.phase", meta.at(B).offset[0] % meta.at(B).tile, 10);

    delete dag;
}

void test_shifted_expression_temp_uses_absolute_representative() {
    std::cout << "=== test_shifted_expression_temp_uses_absolute_representative ==="
              << std::endl;

    constexpr int A = 1;
    constexpr int TMP = 10;
    constexpr int SET = 11;

    MetaMap meta;
    meta[A] = make_meta(/*ndims=*/1, /*size0=*/512, /*offset0=*/0, /*tile=*/64, /*final=*/true);
    meta[TMP] = make_meta(/*ndims=*/1, /*size0=*/200, /*offset0=*/0, /*tile=*/0, /*final=*/false);

    ASTNode* tmp = make_add(TMP,
                            make_leaf(A), make_region<1>({100}, {300}, {1}),
                            make_leaf(A), make_region<1>({150}, {350}, {1}),
                            /*is_temp=*/true, /*ndims=*/1);
    ASTNode* set_region =
        make_set_region(SET, make_leaf(A), make_region<1>({200}, {400}, {1}), make_leaf(TMP));
    DAG* dag = make_single_node_dag({tmp, set_region});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("TMP.tile", meta.at(TMP).tile, 64);
    expect_eq("TMP.offset[0]", meta.at(TMP).offset[0], 150);
    expect_eq("TMP.phase", meta.at(TMP).offset[0] % meta.at(TMP).tile, 22);
    expect_eq("A.offset[0]", meta.at(A).offset[0], 0);

    delete dag;
}

void test_jacobi1d_temp_offset() {
    std::cout << "=== test_jacobi1d_temp_offset ===" << std::endl;

    constexpr int U = 1;
    constexpr int F = 2;
    constexpr int TMP_STENCIL = 10;
    constexpr int TMP_WEIGHTED = 11;
    constexpr int SET = 12;

    MetaMap meta;
    meta[U] = make_meta(/*ndims=*/1, /*size0=*/129, /*offset0=*/0, /*tile=*/64, /*final=*/true);
    meta[F] = make_meta(/*ndims=*/1, /*size0=*/129, /*offset0=*/0, /*tile=*/64, /*final=*/true);
    meta[TMP_STENCIL] =
        make_meta(/*ndims=*/1, /*size0=*/127, /*offset0=*/0, /*tile=*/64, /*final=*/false);
    meta[TMP_WEIGHTED] =
        make_meta(/*ndims=*/1, /*size0=*/127, /*offset0=*/0, /*tile=*/64, /*final=*/false);

    ASTNode* tmp_stencil =
        new ASTNode(TMP_STENCIL, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/1);
    tmp_stencil->add_operand(make_leaf(U));
    tmp_stencil->operand_regions.push_back(make_region<1>({0}, {127}, {1}));
    tmp_stencil->add_operand(make_leaf(U));
    tmp_stencil->operand_regions.push_back(make_region<1>({2}, {129}, {1}));
    tmp_stencil->add_operand(make_leaf(F));
    tmp_stencil->operand_regions.push_back(make_region<1>({1}, {128}, {1}));

    ASTNode* tmp_weighted =
        new ASTNode(TMP_WEIGHTED, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/1);
    tmp_weighted->add_operand(make_leaf(U));
    tmp_weighted->operand_regions.push_back(make_region<1>({1}, {128}, {1}));
    tmp_weighted->add_operand(make_leaf(TMP_STENCIL));
    tmp_weighted->operand_regions.push_back(nullptr);

    ASTNode* set_region =
        make_set_region(SET, make_leaf(U), make_region<1>({1}, {128}, {1}),
                        make_leaf(TMP_WEIGHTED));
    DAG* dag = make_single_node_dag({tmp_stencil, tmp_weighted, set_region});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("TMP_STENCIL.tile", meta.at(TMP_STENCIL).tile, 64);
    expect_eq("TMP_WEIGHTED.tile", meta.at(TMP_WEIGHTED).tile, 64);
    expect_eq("TMP_STENCIL.offset[0]", meta.at(TMP_STENCIL).offset[0], 1);
    expect_eq("TMP_WEIGHTED.offset[0]", meta.at(TMP_WEIGHTED).offset[0], 1);

    delete dag;
}

void test_jacobi2d_temp_offset() {
    std::cout << "=== test_jacobi2d_temp_offset ===" << std::endl;

    constexpr int U = 1;
    constexpr int F = 2;
    constexpr int TMP_STENCIL = 10;
    constexpr int TMP_WEIGHTED = 11;
    constexpr int SET = 12;

    MetaMap meta;
    meta[U] = make_meta(/*ndims=*/2, /*shape=*/{129, 129, 0}, /*offset=*/{0, 0, 0},
                        /*tile=*/64, /*final=*/true);
    meta[F] = make_meta(/*ndims=*/2, /*shape=*/{129, 129, 0}, /*offset=*/{0, 0, 0},
                        /*tile=*/64, /*final=*/true);
    meta[TMP_STENCIL] = make_meta(/*ndims=*/2, /*shape=*/{127, 127, 0}, /*offset=*/{0, 0, 0},
                                  /*tile=*/64, /*final=*/false);
    meta[TMP_WEIGHTED] = make_meta(/*ndims=*/2, /*shape=*/{127, 127, 0}, /*offset=*/{0, 0, 0},
                                   /*tile=*/64, /*final=*/false);

    ASTNode* tmp_stencil =
        new ASTNode(TMP_STENCIL, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/2);
    tmp_stencil->add_operand(make_leaf(U, 2));
    tmp_stencil->operand_regions.push_back(make_region<2>({0, 1}, {127, 128}, {1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 2));
    tmp_stencil->operand_regions.push_back(make_region<2>({2, 1}, {129, 128}, {1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 2));
    tmp_stencil->operand_regions.push_back(make_region<2>({1, 0}, {128, 127}, {1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 2));
    tmp_stencil->operand_regions.push_back(make_region<2>({1, 2}, {128, 129}, {1, 1}));
    tmp_stencil->add_operand(make_leaf(F, 2));
    tmp_stencil->operand_regions.push_back(make_region<2>({1, 1}, {128, 128}, {1, 1}));

    ASTNode* tmp_weighted =
        new ASTNode(TMP_WEIGHTED, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/2);
    tmp_weighted->add_operand(make_leaf(U, 2));
    tmp_weighted->operand_regions.push_back(make_region<2>({1, 1}, {128, 128}, {1, 1}));
    tmp_weighted->add_operand(make_leaf(TMP_STENCIL, 2));
    tmp_weighted->operand_regions.push_back(nullptr);

    ASTNode* set_region =
        make_set_region(SET, make_leaf(U, 2), make_region<2>({1, 1}, {128, 128}, {1, 1}),
                        make_leaf(TMP_WEIGHTED, 2), /*ndims=*/2);
    DAG* dag = make_single_node_dag({tmp_stencil, tmp_weighted, set_region});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("TMP_STENCIL.tile", meta.at(TMP_STENCIL).tile, 64);
    expect_eq("TMP_WEIGHTED.tile", meta.at(TMP_WEIGHTED).tile, 64);
    expect_offset("TMP_STENCIL.offset", meta.at(TMP_STENCIL).offset, {1, 1, 0});
    expect_offset("TMP_WEIGHTED.offset", meta.at(TMP_WEIGHTED).offset, {1, 1, 0});

    delete dag;
}

void test_jacobi3d_temp_offset() {
    std::cout << "=== test_jacobi3d_temp_offset ===" << std::endl;

    constexpr int U = 1;
    constexpr int TMP_STENCIL = 10;
    constexpr int TMP_WEIGHTED = 11;
    constexpr int SET = 12;

    MetaMap meta;
    meta[U] = make_meta(/*ndims=*/3, /*shape=*/{65, 65, 65}, /*offset=*/{0, 0, 0},
                        /*tile=*/64, /*final=*/true);
    meta[TMP_STENCIL] = make_meta(/*ndims=*/3, /*shape=*/{63, 63, 63}, /*offset=*/{0, 0, 0},
                                  /*tile=*/64, /*final=*/false);
    meta[TMP_WEIGHTED] = make_meta(/*ndims=*/3, /*shape=*/{63, 63, 63}, /*offset=*/{0, 0, 0},
                                   /*tile=*/64, /*final=*/false);

    ASTNode* tmp_stencil =
        new ASTNode(TMP_STENCIL, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/3);
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({0, 1, 1}, {63, 64, 64}, {1, 1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({2, 1, 1}, {65, 64, 64}, {1, 1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({1, 0, 1}, {64, 63, 64}, {1, 1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({1, 2, 1}, {64, 65, 64}, {1, 1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({1, 1, 0}, {64, 64, 63}, {1, 1, 1}));
    tmp_stencil->add_operand(make_leaf(U, 3));
    tmp_stencil->operand_regions.push_back(make_region<3>({1, 1, 2}, {64, 64, 65}, {1, 1, 1}));

    ASTNode* tmp_weighted =
        new ASTNode(TMP_WEIGHTED, static_cast<int>(Opcode::ADD),
                    /*is_temp=*/true, /*is_scalar=*/false,
                    /*scalar=*/0.0, /*ndims=*/3);
    tmp_weighted->add_operand(make_leaf(U, 3));
    tmp_weighted->operand_regions.push_back(make_region<3>({1, 1, 1}, {64, 64, 64}, {1, 1, 1}));
    tmp_weighted->add_operand(make_leaf(TMP_STENCIL, 3));
    tmp_weighted->operand_regions.push_back(nullptr);

    ASTNode* set_region =
        make_set_region(SET, make_leaf(U, 3), make_region<3>({1, 1, 1}, {64, 64, 64}, {1, 1, 1}),
                        make_leaf(TMP_WEIGHTED, 3), /*ndims=*/3);
    DAG* dag = make_single_node_dag({tmp_stencil, tmp_weighted, set_region});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("TMP_STENCIL.tile", meta.at(TMP_STENCIL).tile, 64);
    expect_eq("TMP_WEIGHTED.tile", meta.at(TMP_WEIGHTED).tile, 64);
    expect_offset("TMP_STENCIL.offset", meta.at(TMP_STENCIL).offset, {1, 1, 1});
    expect_offset("TMP_WEIGHTED.offset", meta.at(TMP_WEIGHTED).offset, {1, 1, 1});

    delete dag;
}

void test_strided_copy_reduces_tile_and_sets_offset() {
    std::cout << "=== test_strided_copy_reduces_tile_and_sets_offset ===" << std::endl;

    constexpr int A = 1;
    constexpr int B = 2;

    MetaMap meta;
    meta[A] = make_meta(/*ndims=*/1, /*size0=*/512, /*offset0=*/0, /*tile=*/128, /*final=*/true);
    meta[B] = make_meta(/*ndims=*/1, /*size0=*/100, /*offset0=*/0, /*tile=*/128, /*final=*/false);

    ASTNode* copy = make_copy(B, make_leaf(A),
                              make_region<1>({2}, {202}, {2}),
                              /*is_temp=*/false, /*ndims=*/1);
    DAG* dag = make_single_node_dag({copy});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("B.tile", meta.at(B).tile, 64);
    expect_eq("B.offset[0]", meta.at(B).offset[0], 1);
    expect_eq("B.phase", meta.at(B).offset[0] % meta.at(B).tile, 1);

    delete dag;
}

ASTNode* make_transpose(int result_name, ASTNode* src, Region* src_region,
                        std::vector<int> perm) {
    int nd = (int)perm.size();
    auto* node = new ASTNode(result_name, static_cast<int>(Opcode::TRANSPOSE),
                             /*is_temp=*/false, /*is_scalar=*/false, /*scalar=*/0.0, nd);
    node->add_operand(src);
    node->operand_regions.push_back(src_region);
    static int scalar_id = -1000;
    for (int p : perm) {
        node->add_operand(new ASTNode(scalar_id--, static_cast<int>(Opcode::NOOP),
                                      false, /*is_scalar=*/true, (double)p));
        node->operand_regions.push_back(nullptr);
    }
    return node;
}

void test_transpose_result_aligned_to_source_tiles() {
    std::cout << "=== test_transpose_result_aligned_to_source_tiles ===" << std::endl;

    constexpr int A = 1;
    constexpr int T = 2;
    constexpr int V = 3;
    constexpr int C = 4;

    // A: (1100, 700), live with offset (3, 5); T = A.T; V = A[100:1050, 30:690].T;
    // C = V + 1 (elementwise consumer of a transpose result).
    MetaMap meta;
    meta[A] = make_meta(2, {1100, 700, 0}, {3, 5, 0}, 512, /*final=*/true);
    meta[T] = make_meta(2, {700, 1100, 0}, {0, 0, 0}, 0, false);
    meta[V] = make_meta(2, {660, 950, 0}, {0, 0, 0}, 0, false);
    meta[C] = make_meta(2, {660, 950, 0}, {0, 0, 0}, 0, false);

    ASTNode* t = make_transpose(T, make_leaf(A, 2), nullptr, {1, 0});
    ASTNode* v = make_transpose(V, make_leaf(A, 2),
                                make_region<2>({100, 30}, {1050, 690}, {1, 1}), {1, 0});
    ASTNode* c = make_add(C, make_leaf(V, 2), nullptr, make_leaf(V, 2), nullptr,
                          /*is_temp=*/false, /*ndims=*/2);
    DAG* dag = make_single_node_dag({t, v, c});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/2);

    expect_eq("T.tile", meta.at(T).tile, 512);
    expect_offset("T.offset", meta.at(T).offset, {5, 3, 0});
    expect_eq("V.tile", meta.at(V).tile, 512);
    // view start (100, 30) + A offset (3, 5) -> source global (103, 35), permuted
    expect_offset("V.offset", meta.at(V).offset, {35, 103, 0});
    expect_offset("C.offset follows V", meta.at(C).offset, {35, 103, 0});

    delete dag;
}


void test_balanced_tiles() {
    std::cout << "=== test_balanced_tiles ===" << std::endl;
    using decomposition_solver::detail::balanced_tile;
    // 19% less on the busiest PE: not worth leaving the power of two
    expect_eq("800^3 on 4 PEs", balanced_tile({800, 800, 800}, 3, 16, 4, 128), 512);
    expect_eq("2555x160x240 on 4 PEs", balanced_tile({2555, 160, 240}, 3, 16, 4, 128), 256);
    expect_eq("1100^3 on 4 PEs", balanced_tile({1100, 1100, 1100}, 3, 16, 4, 128), 560);
    expect_eq("800^3 on 1 PE", balanced_tile({800, 800, 800}, 3, 4, 1, 128), 1024);
    expect_eq("4096^2 on 4 PEs", balanced_tile({4096, 4096, 0}, 2, 16, 4, 1024), 1024);
    expect_eq("4096^2 on 1 PE", balanced_tile({4096, 4096, 0}, 2, 4, 1, 1024), 2048);
    expect_eq("10000^2 on 4 PEs", balanced_tile({10000, 10000, 0}, 2, 16, 4, 1024), 2528);
    expect_eq("1500^2 on 4 PEs", balanced_tile({1500, 1500, 0}, 2, 16, 4, 1024), 1024);
    expect_eq("365x1x240 on 4 PEs", balanced_tile({365, 1, 240}, 3, 16, 4, 128), 128);
    expect_eq("small array", balanced_tile({300, 200, 0}, 2, 16, 4, 1024), 1024);
    // 192 would leave a 48-wide slab on one PE; the power of two is as good here
    expect_eq("160x365x240 on 4 PEs", balanced_tile({160, 365, 240}, 3, 16, 4, 128), 128);
    expect_eq("14016000 on 4 PEs", balanced_tile({14016000, 0, 0}, 1, 16, 4, 1 << 20),
              1 << 20);
    expect_eq("3000000 on 4 PEs", balanced_tile({3000000, 0, 0}, 1, 16, 4, 1 << 20),
              1 << 20);
}

void test_chare_home_pe_balance() {
    std::cout << "=== test_chare_home_pe_balance ===" << std::endl;
    std::vector<std::array<int, 3>> boxes = {{2, 2, 0}, {4, 4, 0}, {4, 1, 0}, {1, 4, 0},
                                             {2, 4, 0}, {2, 2, 2}, {4, 4, 4}, {2, 1, 2},
                                             {2, 2, 1}, {1, 2, 2}};
    // other PE counts: no PE left idle by a thin grid
    for (int pes : {3, 6, 12}) {
        for (std::array<int, 3> box : {std::array<int, 3>{pes, 1, 0}, {1, pes, 0},
                                       {pes, 1, 1}, {1, pes, 1}, {1, 1, pes}}) {
            int nd = box[2] == 0 ? 2 : 3;
            std::vector<int> load(pes, 0);
            for (int i = 0; i < box[0]; ++i)
                for (int j = 0; j < box[1]; ++j)
                    for (int k = 0; k < (nd == 3 ? box[2] : 1); ++k) {
                        int idx[3] = {i, j, k};
                        load[chare_home_pe(nd, idx, pes)]++;
                    }
            std::ostringstream label;
            label << "thin box " << box[0] << "x" << box[1] << "x" << box[2] << " on " << pes;
            expect_eq(label.str(), *std::max_element(load.begin(), load.end()), 1);
        }
    }
    for (int pes : {4, 8}) {
        for (auto box : boxes) {
            int nd = box[2] == 0 ? 2 : 3;
            int n = box[0] * box[1] * (nd == 3 ? box[2] : 1);
            if (n < pes)
                continue;
            std::vector<int> load(pes, 0);
            for (int i = 0; i < box[0]; ++i)
                for (int j = 0; j < box[1]; ++j)
                    for (int k = 0; k < (nd == 3 ? box[2] : 1); ++k) {
                        int idx[3] = {i, j, k};
                        load[chare_home_pe(nd, idx, pes)]++;
                    }
            std::ostringstream label;
            label << "box " << box[0] << "x" << box[1] << "x" << box[2] << " on " << pes;
            expect_eq(label.str(), *std::max_element(load.begin(), load.end()),
                      (n + pes - 1) / pes);
        }
    }
}

void test_misaligned_after_flush_takes_source_tile() {
    std::cout << "=== test_misaligned_after_flush_takes_source_tile ===" << std::endl;

    constexpr int G = 1;
    constexpr int W = 2;

    // grid from an earlier DAG (8194^2, tile 2080); W = grid[1:-1, 1:-1] + grid[0:-2, 1:-1]
    MetaMap meta;
    meta[G] = make_meta(2, {8194, 8194, 0}, {0, 0, 0}, 2080, /*final=*/true);
    meta[W] = make_meta(2, {8192, 8192, 0}, {0, 0, 0}, 0, false);
    ASTNode* w = make_add(W, make_leaf(G, 2), make_region<2>({1, 1}, {8193, 8193}, {1, 1}),
                          make_leaf(G, 2), make_region<2>({0, 1}, {8192, 8193}, {1, 1}),
                          /*is_temp=*/false, /*ndims=*/2);
    DAG* dag = make_single_node_dag({w});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/4);

    expect_eq("W.tile", meta.at(W).tile, 2080);
    expect_eq("W.offset[1]", meta.at(W).offset[1], 1);
    expect_true("W.offset[0] in {0, 1}", meta.at(W).offset[0] == 0 || meta.at(W).offset[0] == 1);

    delete dag;
}

void test_reduce_operands_share_layout() {
    std::cout << "=== test_reduce_operands_share_layout ===" << std::endl;

    constexpr int A = 1;
    constexpr int C = 2;
    constexpr int X = 3;
    constexpr int R = 4;

    MetaMap meta;
    meta[A] = make_meta(1, 500, /*offset0=*/5, /*tile=*/64, /*final=*/true);
    meta[C] = make_meta(1, 500, 0, 64, true);
    meta[X] = make_meta(1, 500, 0, 0, false);
    meta[R] = make_meta(1, 1, 0, 0, false);
    ASTNode* x = make_copy(X, make_leaf(C), nullptr);
    auto* r = new ASTNode(R, static_cast<int>(Opcode::REDUCE), false, false, 0.0, 1);
    r->add_operand(make_leaf(X));
    r->operand_regions.push_back(nullptr);
    r->add_operand(make_leaf(A));
    r->operand_regions.push_back(nullptr);
    DAG* dag = make_single_node_dag({x, r});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("X.tile", meta.at(X).tile, 64);
    expect_eq("X.offset[0]", meta.at(X).offset[0], 5);

    delete dag;
}

void test_small_class_member_keeps_own_tile() {
    std::cout << "=== test_small_class_member_keeps_own_tile ===" << std::endl;

    constexpr int BIG = 1;
    constexpr int SMALL = 2;
    constexpr int SET = 3;

    // SMALL is written into a slice of BIG: one tile class, but a tile sized for BIG
    // would leave SMALL on one or two PEs.
    MetaMap meta;
    meta[BIG] = make_meta(3, {1120, 365, 240}, {0, 0, 0}, 0, false);
    meta[SMALL] = make_meta(3, {160, 365, 240}, {0, 0, 0}, 0, false);
    ASTNode* set = make_set_region(SET, make_leaf(BIG, 3),
                                   make_region<3>({160, 0, 0}, {320, 365, 240}, {1, 1, 1}),
                                   make_leaf(SMALL, 3), /*ndims=*/3);
    DAG* dag = make_single_node_dag({set});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/4);

    expect_eq("SMALL.tile", meta.at(SMALL).tile,
              decomposition_solver::detail::balanced_tile({160, 365, 240}, 3, 16, 4, 64));
    expect_true("BIG has another tile", meta.at(BIG).tile != meta.at(SMALL).tile);

    delete dag;
}

void test_fused_kept_results_share_layout() {
    std::cout << "=== test_fused_kept_results_share_layout ===" << std::endl;

    constexpr int X = 1;
    constexpr int S = 2;
    constexpr int T = 3;

    // one kernel keeps S = X[0:500] + X[1:501] and T = X[2:502] + X[2:502]
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, /*final=*/true);
    meta[S] = make_meta(1, 500, 0, 0, false);
    meta[T] = make_meta(1, 500, 0, 0, false);
    ASTNode* s = make_add(S, make_leaf(X), make_region<1>({0}, {500}, {1}), make_leaf(X),
                          make_region<1>({1}, {501}, {1}));
    ASTNode* t = make_add(T, make_leaf(X), make_region<1>({2}, {502}, {1}), make_leaf(X),
                          make_region<1>({2}, {502}, {1}));
    DAG* dag = make_single_node_dag({s, t});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("T.tile", meta.at(T).tile, meta.at(S).tile);
    expect_offset("T.offset", meta.at(T).offset, meta.at(S).offset);

    delete dag;
}

void test_set_region_kept_result_takes_target_layout() {
    std::cout << "=== test_set_region_kept_result_takes_target_layout ===" << std::endl;

    constexpr int X = 1;
    constexpr int U = 2;
    constexpr int T = 3;
    constexpr int SET = 4;

    // one kernel keeps T = X[2:502] + X[2:502] and writes U[0:500] = T
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, /*final=*/true);
    meta[U] = make_meta(1, 500, 0, 64, /*final=*/true);
    meta[T] = make_meta(1, 500, 0, 0, false);
    ASTNode* t = make_add(T, make_leaf(X), make_region<1>({2}, {502}, {1}), make_leaf(X),
                          make_region<1>({2}, {502}, {1}));
    ASTNode* set = make_set_region(SET, make_leaf(U), make_region<1>({0}, {500}, {1}), t);
    DAG* dag = make_single_node_dag({t, set});

    decomposition_solver::compute_decompositions(meta, dag, /*odf=*/4, /*num_pes=*/1);

    expect_eq("T.tile", meta.at(T).tile, 64);
    expect_offset("T.offset", meta.at(T).offset, {0, 0, 0});

    delete dag;
}

// ---- fused-kernel layouts (tie_kernel_layouts + resolve_kernel_layouts) ----

DAG* make_chain_dag(std::vector<std::vector<ASTNode*>> nodes) {
    auto* dag = new DAG();
    DAGNode* prev = nullptr;
    for (int i = 0; i < (int)nodes.size(); ++i) {
        auto* node = new DAGNode(i, new AST(nodes[i]));
        node->fusible = true;
        node->identifier = 1000 + i;
        dag->nodes[i] = node;
        dag->node_order.push_back(i);
        if (prev) {
            prev->children.push_back(node);
            node->num_parents++;
        } else {
            dag->roots.push_back(node);
        }
        prev = node;
    }
    dag->num_nodes = (int)nodes.size();
    return dag;
}

template <int N>
ASTNode* add_views(int result, int src, std::array<int, N> a, std::array<int, N> b,
                   std::array<int, N> extent, bool is_temp = false) {
    std::array<int, N> a_stop, b_stop, ones;
    for (int d = 0; d < N; ++d) {
        a_stop[d] = a[d] + extent[d];
        b_stop[d] = b[d] + extent[d];
        ones[d] = 1;
    }
    return make_add(result, make_leaf(src, N), make_region<N>(a, a_stop, ones), make_leaf(src, N),
                    make_region<N>(b, b_stop, ones), is_temp, N);
}

ASTNode* matmul_of(int result, int operand) {
    auto* node = new ASTNode(result, static_cast<int>(Opcode::MATMUL), false, false, 0.0, 1);
    node->add_operand(make_leaf(operand));
    node->operand_regions.push_back(nullptr);
    node->add_operand(make_leaf(operand));
    node->operand_regions.push_back(nullptr);
    return node;
}

// Every fused kernel's kept results and set_region targets are in its anchor's layout.
void expect_kernels_consistent(const std::string& label, MetaMap& meta, DAG* dag) {
    namespace dt = decomposition_solver::detail;
    for (auto& [id, node] : dag->nodes) {
        dt::KernelLayout k;
        if (!dt::is_fused_kernel(node) || !dt::kernel_layout(meta, node, k))
            continue;
        auto const& am = meta.at(k.anchor);
        for (ASTNode* root : node->ast->roots) {
            if (dt::is_kept_root(root)) {
                auto const& m = meta.at(root->result_name);
                dt::Box b = dt::region_box(nullptr, m.ndims, m.global_shape);
                expect_true(label + ": node " + std::to_string(id) + " result " +
                                std::to_string(root->result_name),
                            dt::same_layout(m, {0, 0, 0}, b, am, k));
            } else if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION &&
                       root != k.anchor_set) {
                auto const& m = meta.at(root->operands[0]->result_name);
                dt::Box b = dt::region_box(root->region, m.ndims, m.global_shape);
                expect_true(label + ": node " + std::to_string(id) + " set_region target",
                            dt::same_layout(m, b.start, b, am, k));
            }
        }
    }
}

int resolve(MetaMap& meta, DAG* dag) {
    int next_name = 1 << 20, next_node = 1 << 20;
    return decomposition_solver::resolve_kernel_layouts(meta, dag, next_name, next_node)
        .follow_up_nodes;
}

void test_kernel_layout_cross_node_chain() {
    std::cout << "=== test_kernel_layout_cross_node_chain ===" << std::endl;
    constexpr int X = 1, A = 2, B = 3, C = 4, SET = 5;
    // node 0 keeps A = X[0:500] + X[1:501] and B = X[2:502] + X[2:502];
    // node 1 keeps C = X[3:503] + X[3:503] and writes A[0:500] = C
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    for (int n : {A, B, C})
        meta[n] = make_meta(1, 500, 0, 0, false);
    DAG* dag = make_chain_dag(
        {{add_views<1>(A, X, {0}, {1}, {500}), add_views<1>(B, X, {2}, {2}, {500})},
         {add_views<1>(C, X, {3}, {3}, {500}),
          make_set_region(SET, make_leaf(A), make_region<1>({0}, {500}, {1}), make_leaf(C))}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_offset("B.offset", meta.at(B).offset, meta.at(A).offset);
    expect_offset("C.offset", meta.at(C).offset, meta.at(A).offset);
    expect_eq("follow-ups", resolve(meta, dag), 0);
    expect_kernels_consistent("cross node", meta, dag);
    delete dag;
}

void test_kernel_layout_copy_roots() {
    std::cout << "=== test_kernel_layout_copy_roots ===" << std::endl;
    constexpr int X = 1, S = 2, T = 3;
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[S] = make_meta(1, 500, 0, 0, false);
    meta[T] = make_meta(1, 500, 0, 0, false);
    DAG* dag = make_chain_dag({{make_copy(S, make_leaf(X), make_region<1>({0}, {500}, {1})),
                                make_copy(T, make_leaf(X), make_region<1>({7}, {507}, {1}))}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("follow-ups", resolve(meta, dag), 0);
    expect_kernels_consistent("copy roots", meta, dag);
    delete dag;
}

void test_kernel_layout_final_target_and_matmul_result() {
    std::cout << "=== test_kernel_layout_final_target_and_matmul_result ===" << std::endl;
    constexpr int X = 1, T = 2, U = 3, SET = 4, M = 5;
    // T = X[0:500] + X[0:500] is a matmul operand (offset 0); U (offset 3) gets U[...] = T
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[U] = make_meta(1, 500, 3, 64, true);
    meta[T] = make_meta(1, 500, 0, 0, false);
    meta[M] = make_meta(1, 500, 0, 0, false);
    DAG* dag = make_chain_dag(
        {{add_views<1>(T, X, {0}, {0}, {500}),
          make_set_region(SET, make_leaf(U), make_region<1>({0}, {500}, {1}), make_leaf(T))},
         {matmul_of(M, T)}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("T.offset[0]", meta.at(T).offset[0], 0);
    expect_eq("follow-ups", resolve(meta, dag), 1);
    expect_kernels_consistent("final + matmul", meta, dag);
    // node 0 now keeps T and X = T, and a follow-up writes U[0:500] = X before the matmul
    DAGNode* n0 = dag->nodes.at(0);
    expect_eq("node 0 roots", (int)n0->ast->roots.size(), 2);
    expect_eq("X opcode", n0->ast->roots[1]->opcode, static_cast<int>(Opcode::CONVERT));
    expect_eq("node 0 children", (int)n0->children.size(), 1);
    DAGNode* f = n0->children[0];
    expect_eq("follow-up opcode", f->ast->roots[0]->opcode, static_cast<int>(Opcode::SET_REGION));
    expect_eq("follow-up target", f->ast->roots[0]->operands[0]->result_name, U);
    expect_eq("follow-up source", f->ast->roots[0]->operands[1]->result_name,
              n0->ast->roots[1]->result_name);
    expect_eq("follow-up child", f->children[0]->id, 1);
    expect_eq("dag nodes", dag->num_nodes, 3);
    delete dag;
}

void test_kernel_layout_two_final_targets() {
    std::cout << "=== test_kernel_layout_two_final_targets ===" << std::endl;
    constexpr int X = 1, T = 2, U = 3, W = 4, SU = 5, SW = 6;
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[U] = make_meta(1, 500, 0, 64, true);
    meta[W] = make_meta(1, 500, 5, 64, true);
    meta[T] = make_meta(1, 500, 0, 0, false);
    DAG* dag = make_chain_dag(
        {{add_views<1>(T, X, {1}, {1}, {500}, /*is_temp=*/true),
          make_set_region(SU, make_leaf(U), make_region<1>({0}, {500}, {1}), make_leaf(T)),
          make_set_region(SW, make_leaf(W), make_region<1>({0}, {500}, {1}), make_leaf(T))}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("follow-ups", resolve(meta, dag), 2);
    expect_kernels_consistent("two final targets", meta, dag);
    DAGNode* f1 = dag->nodes.at(0)->children[0];
    DAGNode* f2 = f1->children[0];
    expect_eq("first follow-up target", f1->ast->roots[0]->operands[0]->result_name, U);
    expect_eq("second follow-up target", f2->ast->roots[0]->operands[0]->result_name, W);
    expect_true("follow-ups end the DAG", f2->children.empty());
    delete dag;
}

void test_kernel_layout_shifted_set_regions() {
    std::cout << "=== test_kernel_layout_shifted_set_regions ===" << std::endl;
    constexpr int X = 1, T = 2, U = 3, W = 4, SU = 5, SW = 6;
    // U is created in this DAG, so it moves to W's layout shifted by W's region start
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[U] = make_meta(1, 500, 0, 0, false);
    meta[W] = make_meta(1, 510, 0, 64, true);
    meta[T] = make_meta(1, 500, 0, 0, false);
    auto* create = new ASTNode(U, static_cast<int>(Opcode::CREATE), false, false, 0.0, 1);
    create->region = make_region<1>({0}, {500}, {1});
    DAG* dag = make_chain_dag(
        {{create},
         {add_views<1>(T, X, {0}, {0}, {500}),
          make_set_region(SU, make_leaf(U), make_region<1>({0}, {500}, {1}), make_leaf(T)),
          make_set_region(SW, make_leaf(W), make_region<1>({3}, {503}, {1}), make_leaf(T))}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("U.offset[0]", meta.at(U).offset[0], 3);
    expect_eq("follow-ups", resolve(meta, dag), 0);
    expect_kernels_consistent("shifted set_regions", meta, dag);
    delete dag;
}

void test_kernel_layout_rename_kept_result() {
    std::cout << "=== test_kernel_layout_rename_kept_result ===" << std::endl;
    constexpr int X = 1, S = 2, T = 3, V = 4, M = 5;
    // S is fixed at offset 2 (the anchor), T is a matmul operand; V reads T in the kernel
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[S] = make_meta(1, 500, 2, 64, true);
    meta[T] = make_meta(1, 500, 0, 0, false);
    meta[V] = make_meta(1, 500, 0, 0, false);
    meta[M] = make_meta(1, 500, 0, 0, false);
    ASTNode* v = make_add(V, make_leaf(T), nullptr, make_leaf(T), nullptr);
    DAG* dag = make_chain_dag({{add_views<1>(S, X, {2}, {2}, {500}),
                                add_views<1>(T, X, {0}, {0}, {500}), v},
                               {matmul_of(M, T)}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("follow-ups", resolve(meta, dag), 1);
    expect_kernels_consistent("rename", meta, dag);
    DAGNode* n0 = dag->nodes.at(0);
    int x = n0->ast->roots[1]->result_name;
    expect_true("T renamed", x != T);
    expect_eq("V reads X", v->operands[0]->result_name, x);
    expect_eq("V reads X", v->operands[1]->result_name, x);
    expect_eq("T keeps offset 0", meta.at(T).offset[0], 0);
    ASTNode* f = n0->children[0]->ast->roots[0];
    expect_eq("follow-up writes T", f->result_name, T);
    expect_eq("follow-up reads X", f->operands[0]->result_name, x);
    expect_true("X retires after the follow-up",
                n0->children[0]->retire_names.size() == 1 && dag->retire_remaining.at(x) == 1);
    delete dag;
}

void test_kernel_layout_mixed_shapes() {
    std::cout << "=== test_kernel_layout_mixed_shapes ===" << std::endl;
    constexpr int X = 1, S = 2, T = 3;
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[S] = make_meta(1, 500, 0, 0, false);
    meta[T] = make_meta(1, 499, 0, 0, false);
    DAG* dag = make_chain_dag({{add_views<1>(S, X, {0}, {1}, {500}),
                                add_views<1>(T, X, {1}, {1}, {499})}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    meta.at(T).offset[0] = 3; // as if pinned elsewhere
    expect_true("mixed shapes detected",
                decomposition_solver::detail::kept_shapes_differ(meta, dag->nodes.at(0)));
    int next_name = 1 << 20, next_node = 1 << 20;
    auto fix = decomposition_solver::resolve_kernel_layouts(meta, dag, next_name, next_node);
    expect_true("mixed shapes reported", !fix.error.empty());
    meta.at(T).global_shape[0] = 500;
    expect_true("same shapes not reported",
                !decomposition_solver::detail::kept_shapes_differ(meta, dag->nodes.at(0)));
    uint64_t h = decomposition_solver::layout_hash(meta, dag, next_name, next_node);
    expect_true("layout hash repeats",
                h == decomposition_solver::layout_hash(meta, dag, next_name, next_node));
    expect_true("layout hash sees the counters",
                h != decomposition_solver::layout_hash(meta, dag, next_name - 1, next_node));
    meta.at(T).offset[0] = 4;
    expect_true("layout hash sees offsets",
                h != decomposition_solver::layout_hash(meta, dag, next_name, next_node));
    delete dag;
}

void test_kernel_layout_rewritten_result_then_set_region() {
    std::cout << "=== test_kernel_layout_rewritten_result_then_set_region ===" << std::endl;
    constexpr int X = 1, Q = 2, R = 3, R2 = 4, U = 5, C = 6, D = 7, SR = 8, SR2 = 9, SU = 10,
                  M = 11;
    // Q (matmul operand, offset 0) anchors the rewritten kernel; R and R2 are tied to the
    // earlier U (offset 5), then R[...] = C, R2[...] = D, U[...] = C. The copies R = X'
    // must run before the set_regions into R and R2, or they undo them.
    MetaMap meta;
    meta[X] = make_meta(1, 512, 0, 64, true);
    meta[U] = make_meta(1, 500, 5, 64, true);
    meta[C] = make_meta(1, 500, 0, 64, true);
    meta[D] = make_meta(1, 500, 0, 64, true);
    for (int n : {Q, R, R2, M})
        meta[n] = make_meta(1, 500, 0, 0, false);
    auto* whole = make_region<1>({0}, {500}, {1});
    DAG* dag = make_chain_dag({{add_views<1>(Q, X, {0}, {0}, {500}),
                                add_views<1>(R, X, {1}, {2}, {500}),
                                add_views<1>(R2, X, {3}, {3}, {500}),
                                make_set_region(SR, make_leaf(R), whole, make_leaf(C)),
                                make_set_region(SR2, make_leaf(R2), whole, make_leaf(D)),
                                make_set_region(SU, make_leaf(U), whole, make_leaf(C))},
                               {matmul_of(M, Q)}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 1);
    expect_eq("Q.offset[0]", meta.at(Q).offset[0], 0);
    expect_eq("R.offset[0]", meta.at(R).offset[0], 5);
    expect_eq("follow-ups", resolve(meta, dag), 5);
    expect_kernels_consistent("rewritten result + set_region", meta, dag);
    std::vector<ASTNode*> seq;
    for (DAGNode* n = dag->nodes.at(0)->children[0]; n->id != 1; n = n->children[0])
        seq.push_back(n->ast->roots[0]);
    expect_eq("chain length", (int)seq.size(), 5);
    auto copy_of = [&](ASTNode* f) {
        return f->opcode == static_cast<int>(Opcode::CONVERT) ? f->result_name : -1;
    };
    auto set_of = [&](ASTNode* f) {
        return f->opcode == static_cast<int>(Opcode::SET_REGION) ? f->operands[0]->result_name
                                                                 : -1;
    };
    expect_eq("1st: R = X'", copy_of(seq[0]), R);
    expect_eq("2nd: R2 = X'", copy_of(seq[1]), R2);
    expect_eq("3rd: R[...] = X", set_of(seq[2]), R);
    expect_eq("4th: R2[...] = X", set_of(seq[3]), R2);
    expect_eq("5th: U[...] = X", set_of(seq[4]), U);
    delete dag;
}

void test_kernel_layout_size1_results() {
    std::cout << "=== test_kernel_layout_size1_results ===" << std::endl;
    constexpr int S = 1, A = 2, B = 3;
    // size-1 results of a size-1 array (a reduction result has tile 0 until solved)
    MetaMap meta;
    meta[S] = make_meta(1, 1, 0, 0, false);
    meta[A] = make_meta(1, 1, 0, 0, false);
    meta[B] = make_meta(1, 1, 0, 0, false);
    DAG* dag = make_chain_dag({{add_views<1>(A, S, {0}, {0}, {1}),
                                make_add(B, make_leaf(S), nullptr, make_leaf(A), nullptr)}});
    decomposition_solver::compute_decompositions(meta, dag, 4, 4);
    for (int n : {S, A, B})
        expect_true("size-1 tile > 0", meta.at(n).tile > 0);
    expect_eq("size-1 follow-ups", resolve(meta, dag), 0);
    delete dag;
}

void test_kernel_layout_2d_3d() {
    std::cout << "=== test_kernel_layout_2d_3d ===" << std::endl;
    {
        constexpr int X = 1, A = 2, B = 3, C = 4, SET = 5;
        MetaMap meta;
        meta[X] = make_meta(2, {300, 300, 0}, {0, 0, 0}, 64, true);
        for (int n : {A, B, C})
            meta[n] = make_meta(2, {290, 290, 0}, {0, 0, 0}, 0, false);
        DAG* dag = make_chain_dag(
            {{add_views<2>(A, X, {0, 0}, {1, 2}, {290, 290}),
              add_views<2>(B, X, {3, 5}, {3, 5}, {290, 290})},
             {add_views<2>(C, X, {7, 1}, {7, 1}, {290, 290}),
              make_set_region(SET, make_leaf(A, 2), make_region<2>({0, 0}, {290, 290}, {1, 1}),
                              make_leaf(C, 2), 2)}});
        decomposition_solver::compute_decompositions(meta, dag, 4, 4);
        expect_offset("2-D B.offset", meta.at(B).offset, meta.at(A).offset);
        expect_offset("2-D C.offset", meta.at(C).offset, meta.at(A).offset);
        expect_eq("2-D follow-ups", resolve(meta, dag), 0);
        expect_kernels_consistent("2-D chain", meta, dag);
        delete dag;
    }
    {
        // two earlier 2-D arrays at different offsets written by one kernel
        constexpr int X = 1, T = 2, U = 3, W = 4, SU = 5, SW = 6;
        MetaMap meta;
        meta[X] = make_meta(2, {300, 300, 0}, {0, 0, 0}, 64, true);
        meta[U] = make_meta(2, {290, 290, 0}, {0, 0, 0}, 64, true);
        meta[W] = make_meta(2, {290, 290, 0}, {4, 9, 0}, 64, true);
        meta[T] = make_meta(2, {290, 290, 0}, {0, 0, 0}, 0, false);
        auto* whole = make_region<2>({0, 0}, {290, 290}, {1, 1});
        DAG* dag = make_chain_dag(
            {{add_views<2>(T, X, {1, 1}, {2, 2}, {290, 290}),
              make_set_region(SU, make_leaf(U, 2), whole, make_leaf(T, 2), 2),
              make_set_region(SW, make_leaf(W, 2), whole, make_leaf(T, 2), 2)}});
        decomposition_solver::compute_decompositions(meta, dag, 4, 4);
        expect_eq("2-D conflict follow-ups", resolve(meta, dag), 2);
        expect_kernels_consistent("2-D conflict", meta, dag);
        delete dag;
    }
    {
        constexpr int X = 1, A = 2, B = 3;
        MetaMap meta;
        meta[X] = make_meta(3, {80, 80, 80}, {0, 0, 0}, 16, true);
        meta[A] = make_meta(3, {70, 70, 70}, {0, 0, 0}, 0, false);
        meta[B] = make_meta(3, {70, 70, 70}, {0, 0, 0}, 0, false);
        DAG* dag = make_chain_dag({{add_views<3>(A, X, {0, 0, 0}, {1, 1, 1}, {70, 70, 70}),
                                    add_views<3>(B, X, {3, 5, 7}, {3, 5, 7}, {70, 70, 70})}});
        decomposition_solver::compute_decompositions(meta, dag, 4, 4);
        expect_offset("3-D B.offset", meta.at(B).offset, meta.at(A).offset);
        expect_eq("3-D B.tile", meta.at(B).tile, meta.at(A).tile);
        expect_eq("3-D follow-ups", resolve(meta, dag), 0);
        expect_kernels_consistent("3-D", meta, dag);
        delete dag;
    }
}
} // namespace

int main() {
    try {
        test_shifted_copy_uses_absolute_offset();
        test_shifted_copy_from_nonzero_source_offset();
        test_shifted_expression_temp_uses_absolute_representative();
        test_jacobi1d_temp_offset();
        test_jacobi2d_temp_offset();
        test_jacobi3d_temp_offset();
        test_strided_copy_reduces_tile_and_sets_offset();
        test_transpose_result_aligned_to_source_tiles();
        test_balanced_tiles();
        test_chare_home_pe_balance();
        test_misaligned_after_flush_takes_source_tile();
        test_reduce_operands_share_layout();
        test_small_class_member_keeps_own_tile();
        test_fused_kept_results_share_layout();
        test_set_region_kept_result_takes_target_layout();
        test_kernel_layout_cross_node_chain();
        test_kernel_layout_copy_roots();
        test_kernel_layout_final_target_and_matmul_result();
        test_kernel_layout_two_final_targets();
        test_kernel_layout_shifted_set_regions();
        test_kernel_layout_rename_kept_result();
        test_kernel_layout_mixed_shapes();
        test_kernel_layout_rewritten_result_then_set_region();
        test_kernel_layout_size1_results();
        test_kernel_layout_2d_3d();
    } catch (const std::exception& ex) {
        std::cerr << "test_compute_decompositions failed: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "test_compute_decompositions passed" << std::endl;
    return 0;
}
