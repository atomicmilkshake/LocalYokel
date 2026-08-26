#pragma once
// Device math that BIT-MATCHES python/freetoken/kvcache/quant_codec.py
// (WHT + L2 + Lloyd-Max). Shared by quant_store.cu and quant_dequant.cu.

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace quant_codec {

template <std::uint32_t kBits> inline constexpr int ncentroids() {
  if constexpr (kBits == 2)
    return 4;
  else if constexpr (kBits == 3)
    return 5;
  else
    return 16;  // true 4-bit (matches the Python 16-level oracle)
}

template <std::uint32_t kBits>
__device__ __forceinline__ void fill_centroids(float *c, std::size_t hd) {
  const float s = rsqrtf(static_cast<float>(hd));
  if constexpr (kBits == 2) {
    const float b[4] = {-1.45f, 0.f, 0.f, 1.45f};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      c[i] = b[i] * s;
  } else if constexpr (kBits == 3) {
    const float b[5] = {-1.8271f, -0.6904f, 0.f, 0.6904f, 1.8271f};
#pragma unroll
    for (int i = 0; i < 5; ++i)
      c[i] = b[i] * s;
  } else {
    const float b[16] = {-2.7393f, -2.0719f, -1.6213f, -1.2595f,
                         -0.9453f, -0.6591f, -0.3893f, -0.129f,
                          0.129f,  0.3892f,  0.6586f,  0.9442f,
                          1.2585f,  1.622f,  2.0747f,  2.7359f};
#pragma unroll
    for (int i = 0; i < 16; ++i)
      c[i] = b[i] * s;
  }
}

// Orthonormal fast Walsh-Hadamard along last dim (pow2). out = (1/sqrt n)*WHT.
template <std::size_t kDim>
__device__ __forceinline__ void fwht(float *x) {
  std::size_t h = 1;
  while (h < kDim) {
    for (std::size_t i = 0; i < kDim; i += (h << 1)) {
      for (std::size_t j = 0; j < h; ++j) {
        const float a = x[i + j];
        const float b = x[i + j + h];
        x[i + j] = a + b;
        x[i + j + h] = a - b;
      }
    }
    h <<= 1;
  }
  const float inv = rsqrtf(static_cast<float>(kDim));
  for (std::size_t i = 0; i < kDim; ++i)
    x[i] *= inv;
}

// Per-coordinate nearest Lloyd-Max index (matches nearest_code).
template <std::uint32_t kBits>
__device__ __forceinline__ std::uint8_t nearest_one(float q, const float *c) {
  constexpr int n = ncentroids<kBits>();
  int best = 0;
  float bd = (q - c[0]) * (q - c[0]);
#pragma unroll
  for (int k = 1; k < n; ++k) {
    const float e = q - c[k];
    const float d = e * e;
    if (d < bd) {
      bd = d;
      best = k;
    }
  }
  return static_cast<std::uint8_t>(best);
}

// ---- true bit-packed 4-bit storage: 2 codes per byte (nibble) ----------------
// This is the "perf commit": tq4 stores 0.5 B/elem (not 1 B int8), so resident KV
// drops by 2x vs int8 and ~4x vs fp16. bit-matches python quant_codec.pack_code_nibble
// (even index -> low nibble, odd index -> high nibble). Use uint8 so >> is logical.

template <std::uint32_t kBits, std::size_t kHeadDim>
inline constexpr std::size_t codes_width() {
  if constexpr (kBits == 4) {
    static_assert(kHeadDim % 2 == 0, "tq4 nibble pack requires even head_dim");
    return kHeadDim / 2;
  } else {
    return kHeadDim;
  }
}

__device__ __forceinline__ void pack_code_nibble(
    std::int8_t *dst, std::size_t i, std::uint8_t code) {
  auto *u = reinterpret_cast<std::uint8_t *>(dst);
  std::uint8_t &b = u[i / 2];
  if ((i & 1) == 0)
    b = static_cast<std::uint8_t>(code & 0x0F);
  else
    b = static_cast<std::uint8_t>((b & 0x0F) | ((code & 0x0F) << 4));
}

__device__ __forceinline__ std::uint8_t unpack_code_nibble(
    const std::int8_t *src, std::size_t i) {
  const auto *u = reinterpret_cast<const std::uint8_t *>(src);
  const std::uint8_t b = u[i / 2];
  return ((i & 1) == 0) ? static_cast<std::uint8_t>(b & 0x0F)
                        : static_cast<std::uint8_t>((b >> 4) & 0x0F);
}

} // namespace quant_codec
