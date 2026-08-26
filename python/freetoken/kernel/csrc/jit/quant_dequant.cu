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
// quant_dequant -- gather referenced pages, inverse of encode_block, scatter
// into one-layer fp16 scratch of pool geometry (P, ps, H, hd).
// One warp = one (page, slot, head) row. page_id < 0 is skipped.
// --------------------------------------------------------------------------

namespace {

struct QuantDequantParams {
  void *__restrict__ scratch;
  const void *__restrict__ codes;
  const void *__restrict__ norms;
  const void *__restrict__ rot;
  const void *__restrict__ page_ids;
  std::size_t page_stride; // ps * H * codes_width  (int8 bytes per page)
  std::size_t slot_stride; // H * codes_width
  std::size_t head_stride; // codes_width (hd/2 when kBits==4)
  std::size_t norm_page_stride; // ps * H
  std::size_t page_size;
  std::size_t kv_heads;
  std::size_t length; // M * ps * H warps
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kHeadDim, std::uint32_t kBits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    quant_dequant_pages(const __grid_constant__ QuantDequantParams params) {
  using namespace device;
  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);
  static_assert(kHeadDim >= 1 && (kHeadDim & (kHeadDim - 1)) == 0);
  constexpr auto kDstBytes = kHeadDim * sizeof(__half);

  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane = threadIdx.x % kWarpThreads;
  PDL::wait<kUsePDL>();

  __shared__ float smem_f[kWarpPerBlock][kHeadDim];
  __shared__ __half smem_h[kWarpPerBlock][kHeadDim];

  if (warp_id < params.length) {
    const auto inner = params.page_size * params.kv_heads;
    const auto page_idx = warp_id / inner;
    const auto rem = warp_id % inner;
    const auto slot = rem / params.kv_heads;
    const auto head = rem % params.kv_heads;
    const auto page = static_cast<const T *>(params.page_ids)[page_idx];

    if (page >= 0) {
      const auto codes_row =
          static_cast<std::size_t>(page) * params.page_stride +
          slot * params.slot_stride + head * params.head_stride;
      const auto scratch_row =
          (static_cast<std::size_t>(page) * params.page_size * params.kv_heads +
           slot * params.kv_heads + head) *
          kHeadDim;
      const auto nidx = static_cast<std::size_t>(page) * params.norm_page_stride +
                        slot * params.kv_heads + head;
      const auto src_codes = pointer::offset(params.codes, codes_row);
      const auto src_norm =
          pointer::offset(params.norms, nidx * sizeof(__half));
      const auto dst =
          pointer::offset(params.scratch, scratch_row * sizeof(__half));

      if (lane == 0) {
        float *x = smem_f[warp_in_block];
        float cents[16];
        quant_codec::fill_centroids<kBits>(cents, kHeadDim);
        constexpr int ncent = quant_codec::ncentroids<kBits>();
        const auto *c8 = static_cast<const std::int8_t *>(src_codes);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          int idx;
          if constexpr (kBits == 4)
            idx = static_cast<int>(quant_codec::unpack_code_nibble(c8, i));
          else
            idx = static_cast<int>(c8[i]);
          if (idx < 0)
            idx = 0;
          if (idx >= ncent)
            idx = ncent - 1;
          x[i] = cents[idx];
        }
        quant_codec::fwht<kHeadDim>(x); // iWHT
        const auto *rot = static_cast<const __half *>(params.rot);
        const float n = __half2float(*static_cast<const __half *>(src_norm));
        auto *out_h = smem_h[warp_in_block];
        for (std::size_t i = 0; i < kHeadDim; ++i)
          out_h[i] = __float2half(x[i] * __half2float(rot[i]) * n);
      }
      __syncwarp();
      auto *row_h = smem_h[warp_in_block];
      if constexpr (kDstBytes % 128 == 0) {
        warp::copy<kDstBytes>(dst, static_cast<const void *>(row_h));
      } else {
        auto *d = static_cast<__half *>(dst);
        for (std::size_t i = lane; i < kHeadDim; i += kWarpThreads)
          d[i] = row_h[i];
      }
    }
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t head_dim, std::uint32_t bits,
          std::size_t num_threads = 128, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct QuantDequantKernel {
  static void run(const tvm::ffi::TensorView scratch,
                  const tvm::ffi::TensorView codes,
                  const tvm::ffi::TensorView norms,
                  const tvm::ffi::TensorView rot,
                  const tvm::ffi::TensorView page_ids) {
    using namespace host;
    auto P = SymbolicSize{"P"};
    auto PS = SymbolicSize{"PS"};
    auto H = SymbolicSize{"H"};
    auto D = SymbolicSize{"D"};
    auto M = SymbolicSize{"M"};
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    constexpr auto kCodesWidth =
        static_cast<int64_t>(quant_codec::codes_width<bits, head_dim>());

    TensorMatcher({P, PS, H, kCodesWidth}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<std::int8_t>()
        .verify(codes);
    TensorMatcher({P, PS, H, D}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(scratch);
    TensorMatcher({P, PS, H}) //
        .with_device<kDLCUDA>(device_)
        .verify(norms);
    TensorMatcher({D}) //
        .with_device<kDLCUDA>(device_)
        .verify(rot);
    TensorMatcher({M}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(page_ids);

    RuntimeCheck(D.unwrap() == static_cast<int64_t>(head_dim));
    RuntimeCheck(dtype_bytes(dtype_.unwrap()) == 2);
    RuntimeCheck(bits == 2 || bits == 3 || bits == 4);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto page_size = static_cast<std::size_t>(PS.unwrap());
    const auto kv_heads = static_cast<std::size_t>(H.unwrap());
    const auto n_pages = static_cast<std::size_t>(M.unwrap());
    const auto length = n_pages * page_size * kv_heads;
    const auto codes_w = static_cast<std::size_t>(kCodesWidth);

    const auto params = QuantDequantParams{
        .scratch = scratch.data_ptr(),
        .codes = codes.data_ptr(),
        .norms = norms.data_ptr(),
        .rot = rot.data_ptr(),
        .page_ids = page_ids.data_ptr(),
        .page_stride = page_size * kv_heads * codes_w,
        .slot_stride = kv_heads * codes_w,
        .head_stride = codes_w,
        .norm_page_stride = page_size * kv_heads,
        .page_size = page_size,
        .kv_heads = kv_heads,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel =
        use_int32
            ? quant_dequant_pages<num_threads, max_concurrency, use_pdl,
                                  head_dim, bits, int32_t>
            : quant_dequant_pages<num_threads, max_concurrency, use_pdl,
                                  head_dim, bits, int64_t>;
    if (length == 0)
      return;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
