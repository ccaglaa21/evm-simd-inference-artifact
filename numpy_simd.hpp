#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <simdpp/simd.h>

namespace evmone::numpy {

    // --- EVM MATRIX HEADER (32 Byte) ---
    struct MatrixHeader {
        uint64_t rows;
        uint64_t cols;
        uint64_t stride_row;
        uint64_t stride_col;
    };

    // ==========================================
    // SECTION 1: BASE OPERATIONS (EXISTING)
    // ==========================================

    // --- 1. VIRTUAL TRANSPOSE (Zero-Copy) ---
    inline void op_xtranspose(uint8_t* memory_ptr) {
        MatrixHeader h;
        std::memcpy(&h, memory_ptr, sizeof(MatrixHeader));
        std::swap(h.rows, h.cols);
        std::swap(h.stride_row, h.stride_col);
        std::memcpy(memory_ptr, &h, sizeof(MatrixHeader));
    }

    // --- 2. STRIDE-AWARE SIMD MULTIPLY ---
#if 0   // OUT OF SCOPE - not used in this paper (no opcode, never called)
    inline void op_xmul_stride(uint8_t* ptr_A, uint8_t* ptr_B, uint8_t* ptr_Res) {
        MatrixHeader hA, hB;
        std::memcpy(&hA, ptr_A, sizeof(MatrixHeader));
        std::memcpy(&hB, ptr_B, sizeof(MatrixHeader));

        uint16_t* dataA = reinterpret_cast<uint16_t*>(ptr_A + sizeof(MatrixHeader));
        uint16_t* dataB = reinterpret_cast<uint16_t*>(ptr_B + sizeof(MatrixHeader));
        uint16_t* dataRes = reinterpret_cast<uint16_t*>(ptr_Res + sizeof(MatrixHeader));
        
        MatrixHeader hRes = {hA.rows, hA.cols, hA.cols, 1}; 
        std::memcpy(ptr_Res, &hRes, sizeof(MatrixHeader));

        size_t total = hA.rows * hA.cols;

        if (hA.stride_col == 1 && hB.stride_col == 1) {
            using SimdVec = simdpp::uint16<16>;
            for (size_t i = 0; i < total; i += 16) {
                auto a = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&dataA[i]));
                auto b = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&dataB[i]));
                auto res = simdpp::mul_lo(a, b);
                simdpp::store_u(reinterpret_cast<uint8_t*>(&dataRes[i]), res);
            }
        } else {
            for (size_t i = 0; i < hA.rows; ++i) {
                for (size_t j = 0; j < hA.cols; ++j) {
                    uint16_t val_a = dataA[i * hA.stride_row + j * hA.stride_col];
                    uint16_t val_b = dataB[i * hB.stride_row + j * hB.stride_col];
                    dataRes[i * hA.cols + j] = val_a * val_b;
                }
            }
        }
    }
#endif  // (out of scope: earlier thesis function)

    // ==========================================
    // SECTION 2: ADDITIONAL OPERATIONS (NEW)
    // ==========================================

    // --- 3. CONDITIONAL SELECT (np.where) ---
    // Selects according to the mask coming from the stack.
    // Result = (Mask == True) ? TrueVal : FalseVal
#if 0   // OUT OF SCOPE - not used in this paper (no opcode, never called)
    inline void op_xwhere(uint8_t* ptr_Mask, uint8_t* ptr_True, uint8_t* ptr_False, uint8_t* ptr_Res) {
        MatrixHeader h;
        std::memcpy(&h, ptr_True, sizeof(MatrixHeader));
        std::memcpy(ptr_Res, &h, sizeof(MatrixHeader));

        uint16_t* dMask = reinterpret_cast<uint16_t*>(ptr_Mask + sizeof(MatrixHeader));
        uint16_t* dTrue = reinterpret_cast<uint16_t*>(ptr_True + sizeof(MatrixHeader));
        uint16_t* dFalse = reinterpret_cast<uint16_t*>(ptr_False + sizeof(MatrixHeader));
        uint16_t* dRes = reinterpret_cast<uint16_t*>(ptr_Res + sizeof(MatrixHeader));

        size_t total = h.rows * h.cols;
        using SimdVec = simdpp::uint16<16>;
        
        for (size_t i = 0; i < total; i += 16) {
            auto mask = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&dMask[i]));
            auto vTrue = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&dTrue[i]));
            auto vFalse = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&dFalse[i]));
            
            // Mask bitwise logic: (Mask & True) | (~Mask & False)
            auto res = simdpp::bit_or(simdpp::bit_and(mask, vTrue), simdpp::bit_and(simdpp::bit_not(mask), vFalse));
            simdpp::store_u(reinterpret_cast<uint8_t*>(&dRes[i]), res);
        }
    }
#endif  // (out of scope: earlier thesis function)

    // --- 4. GATHER (Fancy Indexing) ---
    // Works like a database lookup: collects the requested indices into a new array.
    inline void op_xgather(uint8_t* ptr_Data, uint8_t* ptr_Indices, uint8_t* ptr_Res) {
        MatrixHeader hData, hInd;
        std::memcpy(&hData, ptr_Data, sizeof(MatrixHeader));
        std::memcpy(&hInd, ptr_Indices, sizeof(MatrixHeader));
        std::memmove(ptr_Res, &hInd, sizeof(MatrixHeader)); // The result has the size of the index list (memmove: the result region may overlap the operands)

        // EVM memory offsets carry no alignment, so the 16-bit elements are read and written with memcpy
        // (a uint16_t* into the memory would be a misaligned access, undefined in C++).
        uint8_t* dData = ptr_Data + sizeof(MatrixHeader);
        uint8_t* dInd = ptr_Indices + sizeof(MatrixHeader);
        uint8_t* dRes = ptr_Res + sizeof(MatrixHeader);

        size_t total_indices = hInd.rows * hInd.cols;
        size_t max_data_idx = hData.rows * hData.cols;

        for(size_t i = 0; i < total_indices; ++i) {
            uint16_t target_idx;
            std::memcpy(&target_idx, dInd + 2 * i, 2);
            uint16_t value = 0;
            // Bounds check (to avoid a segfault)
            if (target_idx < max_data_idx) {
                std::memcpy(&value, dData + 2 * static_cast<size_t>(target_idx), 2);
            }
            std::memcpy(dRes + 2 * i, &value, 2);
        }
    }
// --- 5. BROADCAST (virtual replication) ---
    // Makes a single number (scalar) behave like a large matrix.
    // Setting the strides to 0 makes the pointer "stay in place".
    // So instead of writing one million "5"s, we write a single "5" that reads as if there were a million.
    inline void op_xbroadcast(uint8_t* ptr_Scalar, uint64_t target_rows, uint64_t target_cols) {
        MatrixHeader h;
        // Build the header for the existing scalar value
        h.rows = target_rows;
        h.cols = target_cols;
        h.stride_row = 0; // Step when moving to the next row (read the same number)
        h.stride_col = 0; // Step when moving right (read the same number)
        
        // Write the header
        std::memcpy(ptr_Scalar, &h, sizeof(MatrixHeader));
    }

    // --- 6. REDUCTION (SUM and MAX) ---
    // Mode: 0 = SUM, 1 = MAX
    // Reduces the whole matrix to a single number.
#if 0   // OUT OF SCOPE - not used in this paper (no opcode, never called)
    inline uint16_t op_xreduce(uint8_t* ptr_Data, int mode) {
        MatrixHeader h;
        std::memcpy(&h, ptr_Data, sizeof(MatrixHeader));
        uint16_t* data = reinterpret_cast<uint16_t*>(ptr_Data + sizeof(MatrixHeader));
        size_t total = h.rows * h.cols;

        if (mode == 0) { // SUM
            // Sum 16 at a time with SIMD
            using SimdVec = simdpp::uint16<16>;
            SimdVec vAcc = simdpp::make_zero(); // Accumulator
            
            size_t i = 0;
            for (; i <= total - 16; i += 16) {
                auto v = simdpp::load_u<SimdVec>(reinterpret_cast<const uint8_t*>(&data[i]));
                vAcc = simdpp::add(vAcc, v);
            }
            
            // Sum the SIMD vector internally (horizontal sum)
            uint16_t sum = static_cast<uint16_t>(simdpp::reduce_add(vAcc));

            // Add the remainder (elements not divisible by 16) by hand
            for (; i < total; ++i) sum += data[i];
            return sum;
        } 
        else { // MAX
            uint16_t max_val = 0;
            for (size_t i = 0; i < total; ++i) {
                if (data[i] > max_val) max_val = data[i];
            }
            return max_val;
        }
    }
#endif  // (out of scope: earlier thesis function)

  // --- 7. RESHAPE ---
    // Updates only the dimensions and strides in the header, without copying the data.
#if 0   // OUT OF SCOPE - not used in this paper (no opcode, never called)
    inline void op_xreshape(uint8_t* ptr_Data, uint64_t new_rows, uint64_t new_cols) {
        MatrixHeader h;
        std::memcpy(&h, ptr_Data, sizeof(MatrixHeader));

        // Size check (does the element count match?)
        if ((h.rows * h.cols) != (new_rows * new_cols)) {
            // Error case: dimensions do not match. Do nothing.
            return; 
        }

        // Set the new dimensions
        h.rows = new_rows;
        h.cols = new_cols;
        
        // Compute the new strides (assuming C order / row-major)
        // 1 step to move sideways, 'cols' steps to move to the next row.
        h.stride_col = 1; 
        h.stride_row = new_cols;

        std::memcpy(ptr_Data, &h, sizeof(MatrixHeader));
    }
#endif  // (out of scope: earlier thesis function)

    // --- 8. MATMUL (matrix multiplication) ---
    // Computes C = A * B. (A: MxK, B: KxN -> C: MxN)
    // O(N^3) complexity, but runs in native C++ rather than in Solidity bytecode.
#if 0   // OUT OF SCOPE - not used in this paper (no opcode, never called)
    inline void op_xmatmul(uint8_t* ptr_A, uint8_t* ptr_B, uint8_t* ptr_C) {
        MatrixHeader hA, hB, hC;
        std::memcpy(&hA, ptr_A, sizeof(MatrixHeader));
        std::memcpy(&hB, ptr_B, sizeof(MatrixHeader));

        // Size check (columns of A must equal rows of B)
        if (hA.cols != hB.rows) return; 

        // Pointers (skip past the header)
const int16_t* dataA = reinterpret_cast<const int16_t*>(ptr_A + sizeof(MatrixHeader));
        const int16_t* dataB = reinterpret_cast<const int16_t*>(ptr_B + sizeof(MatrixHeader));
        int32_t* dataC = reinterpret_cast<int32_t*>(ptr_C + sizeof(MatrixHeader));
        // Set up the header of the result matrix
        hC.rows = hA.rows;
        hC.cols = hB.cols;
        hC.stride_row = hB.cols; // Standard C order
        hC.stride_col = 1;
        std::memcpy(ptr_C, &hC, sizeof(MatrixHeader));

        // --- MATRIX MULTIPLICATION ALGORITHM ---
        // Multiply while respecting the strides (works even for a transposed view)
        for (uint64_t i = 0; i < hA.rows; ++i) {
            for (uint64_t j = 0; j < hB.cols; ++j) {
                 int32_t sum = 0;
                for (uint64_t k = 0; k < hA.cols; ++k) {
                    int32_t valA = dataA[(i * hA.stride_row) + (k * hA.stride_col)];
                    int32_t valB = dataB[(k * hB.stride_row) + (j * hB.stride_col)];
                    sum += valA * valB;
                }
                // C[i][j]
                dataC[(i * hC.stride_row) + (j * hC.stride_col)] = sum;
            }
        }
    }
#endif  // (out of scope: earlier thesis function)




} // End of namespace
