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
    auto* dag = new DAG({node});
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
    } catch (const std::exception& ex) {
        std::cerr << "test_compute_decompositions failed: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "test_compute_decompositions passed" << std::endl;
    return 0;
}
