// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// A^T v baseline, K=64. Same data: M[i]=(i%50)+1, v[c]=(c%7)+1,
/// out[r] = sum_c M[c*K+r]*v[c]. Gas is measured with gasleft() ONLY
/// around the inner-product loop (data setup is outside).
contract ATVBaseline {
    uint256 constant K = 64;

    /// Variant A (MAIN COMPARISON): packed int16 -- 16 lanes per word,
    /// same layout and same unpacking work as our Path C.
    function computePacked() external view
        returns (uint256 gasUsed, uint256 out0, uint256 out1, uint256 out2)
    {
        uint256 matWords = (K * K) / 16;
        uint256[] memory Mp = new uint256[](matWords);
        uint256[] memory Vp = new uint256[](K / 16 + 1);
        unchecked {
            for (uint256 i = 0; i < K * K; ++i) {
                uint256 val = (i % 50) + 1;
                Mp[i / 16] |= val << ((i % 16) * 16);
            }
            for (uint256 c = 0; c < K; ++c) {
                uint256 val = (c % 7) + 1;
                Vp[c / 16] |= val << ((c % 16) * 16);
            }
        }
        uint256[] memory outv = new uint256[](K);
        uint256 g0 = gasleft();
        unchecked {
            for (uint256 r = 0; r < K; ++r) {
                uint256 acc = 0;
                for (uint256 c = 0; c < K; ++c) {
                    uint256 mi = c * K + r;
                    uint256 me = (Mp[mi / 16] >> ((mi % 16) * 16)) & 0xFFFF;
                    uint256 ve = (Vp[c / 16] >> ((c % 16) * 16)) & 0xFFFF;
                    acc += me * ve;
                }
                outv[r] = acc;
            }
        }
        uint256 g1 = gasleft();
        gasUsed = g0 - g1;
        out0 = outv[0]; out1 = outv[1]; out2 = outv[2];
    }

    /// Variant B (CONTEXT): idiomatic unpacked int256[] -- cheap compute,
    /// but 16x storage in a real deployment (noted in the paper).
    function computeUnpacked() external view
        returns (uint256 gasUsed, uint256 out0, uint256 out1, uint256 out2)
    {
        uint256[] memory Mu = new uint256[](K * K);
        uint256[] memory Vu = new uint256[](K);
        unchecked {
            for (uint256 i = 0; i < K * K; ++i) Mu[i] = (i % 50) + 1;
            for (uint256 c = 0; c < K; ++c) Vu[c] = (c % 7) + 1;
        }
        uint256[] memory outv = new uint256[](K);
        uint256 g0 = gasleft();
        unchecked {
            for (uint256 r = 0; r < K; ++r) {
                uint256 acc = 0;
                for (uint256 c = 0; c < K; ++c) acc += Mu[c * K + r] * Vu[c];
                outv[r] = acc;
            }
        }
        uint256 g1 = gasleft();
        gasUsed = g0 - g1;
        out0 = outv[0]; out1 = outv[1]; out2 = outv[2];
    }
}
