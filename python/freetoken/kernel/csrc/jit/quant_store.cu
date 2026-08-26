#include "quant_codec.cuh"

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

// --------------------------------------------------------------------------
// quant_store -- WHT + L2 + Lloyd-Max encode, one (token, head) row per warp.
// Mirrors kvcache/quant_codec.py encode_block and store.cu scatter semantics:
//   dst_codes[pos * H + head] = encode(src[token, head, :])
//   dst_norms[pos * H + head] = L2(src[token, head, :])
// --------------------------------------------------------------------------

namespace {

struct QuantStoreParams {
  void *__restrict__ codes;
  void *__restrict__ norms;
  const void *__restrict__ indices;
  const void *__restrict__ src;
  const void *__restrict__ rot;
  std::size_t kv_heads;
  std::size_t head_stride; // codes_width (hd/2 nibble-packed when kBits==4)
  std::size_t slot_stride; // kv_heads * codes_width
  std::size_t length; // T * H warps
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kHeadDim, std::uint32_t kBits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    quant_store_kv(const __grid_constant__ QuantStoreParams params) {
  using namespace device;
  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);
  static_assert(kHeadDim >= 1 && (kHeadDim & (kHeadDim - 1)) == 0);
  constexpr auto kSrcBytes = kHeadDim * sizeof(__half);

  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane = threadIdx.x % kWarpThreads;
  PDL::wait<kUsePDL>();

  __shared__ __half smem_h[kWarpPerBlock][kHeadDim];
  __shared__ float smem_f[kWarpPerBlock][kHeadDim];

  if (warp_id < params.length) {
    const auto H = params.kv_heads;
    const auto token = warp_id / H;
    const auto head = warp_id % H;
    const auto pos = static_cast<const T *>(params.indices)[token];
    const auto src =
        pointer::offset(params.src, warp_id * kSrcBytes);
    const auto dst_codes = pointer::offset(
        params.codes, static_cast<std::size_t>(pos) * params.slot_stride +
                          head * params.head_stride);
    const auto dst_norm =
        pointer::offset(params.norms, (pos * H + head) * sizeof(__half));

    auto *row_h = smem_h[warp_in_block];
    if constexpr (kSrcBytes % 128 == 0) {
      warp::copy<kSrcBytes>(static_cast<void *>(row_h), src);
    } else {
      const auto *s = static_cast<const __half *>(src);
      for (std::size_t i = lane; i < kHeadDim; i += kWarpThreads)
        row_h[i] = s[i];
    }
    __syncwarp();

    if (lane == 0) {
      float *x = smem_f[warp_in_block];
      float n2 = 0.f;
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        x[i] = __half2float(row_h[i]);
        n2 += x[i] * x[i];
      }
      float n = sqrtf(n2);
      if (n < 1e-6f)
        n = 1e-6f;
      const auto *rot = static_cast<const __half *>(params.rot);
      for (std::size_t i = 0; i < kHeadDim; ++i)
        x[i] = (x[i] / n) * __half2float(rot[i]);
      quant_codec::fwht<kHeadDim>(x);
      float cents[16];
      quant_codec::fill_centroids<kBits>(cents, kHeadDim);
      auto *out_c = static_cast<std::int8_t *>(dst_codes);
      if constexpr (kBits == 4) {
        // true 4-bit: nibble-pack 2 codes per byte, dst width = kHeadDim/2
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          const std::uint8_t code =
              quant_codec::nearest_one<kBits>(x[i], cents);
          quant_codec::pack_code_nibble(out_c, i, code);
        }
      } else {
        for (std::size_t i = 0; i < kHeadDim; ++i)
          out_c[i] = static_cast<std::int8_t>(
              quant_codec::nearest_one<kBits>(x[i], cents));
      }
      *static_cast<__half *>(dst_norm) = __float2half(n);
    }
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t head_dim, std::uint32_t bits,
          std::size_t num_threads = 128, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct QuantStoreKernel {
  static void run(const tvm::ffi::TensorView codes,
                  const tvm::ffi::TensorView norms,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView src,
                  const tvm::ffi::TensorView rot) {
    using namespace host;
    auto L = SymbolicSize{"L"}; // tokens
    auto H = SymbolicSize{"H"};
    auto D = SymbolicSize{"D"};
    auto R = SymbolicSize{"R"}; // P*ps rows
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    constexpr auto kCodesWidth =
        static_cast<int64_t>(quant_codec::codes_width<bits, head_dim>());

    TensorMatcher({L, H, D}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(src);
    TensorMatcher({R, H, kCodesWidth}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<std::int8_t>()
        .verify(codes);
    TensorMatcher({R, H}) //
        .with_device<kDLCUDA>(device_)
        .verify(norms);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);
    TensorMatcher({D}) //
        .with_device<kDLCUDA>(device_)
        .verify(rot);

    RuntimeCheck(D.unwrap() == static_cast<int64_t>(head_dim));
    RuntimeCheck(dtype_bytes(dtype_.unwrap()) == 2);
    RuntimeCheck(bits == 2 || bits == 3 || bits == 4);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto tokens = static_cast<std::size_t>(L.unwrap());
    const auto kv_heads = static_cast<std::size_t>(H.unwrap());
    const auto length = tokens * kv_heads;
    const auto codes_w = static_cast<std::size_t>(kCodesWidth);

    const auto params = QuantStoreParams{
        .codes = codes.data_ptr(),
        .norms = norms.data_ptr(),
        .indices = indices.data_ptr(),
        .src = src.data_ptr(),
        .rot = rot.data_ptr(),
        .kv_heads = kv_heads,
        .head_stride = codes_w,
        .slot_stride = kv_heads * codes_w,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel =
        use_int32
            ? quant_store_kv<num_threads, max_concurrency, use_pdl, head_dim,
                             bits, int32_t>
            : quant_store_kv<num_threads, max_concurrency, use_pdl, head_dim,
                             bits, int64_t>;
    if (length == 0)
      return;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
