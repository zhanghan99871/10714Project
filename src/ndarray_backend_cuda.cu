#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4

/**
 * Supported element types.
 *
 * float32 is stored and computed natively. float16 (__half, the same format as numpy.float16)
 * is a *storage* type: kernels load __half elements, convert them to float, compute in float and
 * convert the result back with round-to-nearest-even (__float2half). Matmul and reductions keep
 * their accumulators in float. Only the __half <-> float conversion intrinsics are used, so this
 * works on every architecture (no native half arithmetic, which needs sm_53+, is required).
 */
enum class DType { F32, F16 };

inline size_t ElemSize(DType dt) { return dt == DType::F32 ? sizeof(float) : sizeof(__half); }

inline const char* DTypeName(DType dt) { return dt == DType::F32 ? "float32" : "float16"; }

inline DType ParseDType(const std::string& name) {
  if (name == "float32") return DType::F32;
  if (name == "float16") return DType::F16;
  throw std::invalid_argument("unsupported dtype: " + name);
}

/// Element access used by every kernel: compute in float, store in the array's type.
__device__ __forceinline__ float Load(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float Load(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void Store(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void Store(__half* p, size_t i, float v) { p[i] = __float2half(v); }

struct CudaArray {
  CudaArray(const size_t size, const std::string& dtype = "float32")
      : dtype(ParseDType(dtype)), size(size) {
    cudaError_t err = cudaMalloc(&ptr, size * ElemSize(this->dtype));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  size_t itemsize() const { return ElemSize(dtype); }
  std::string dtype_name() const { return DTypeName(dtype); }

  template <typename T>
  T* as() {
    return static_cast<T*>(ptr);
  }
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(ptr);
  }

  void* ptr;
  DType dtype;
  size_t size;
};

/// Run `__VA_ARGS__` with `T` bound to the C++ element type of `dt`.
#define NEEDLE_DISPATCH(dt, T, ...)    \
  switch (dt) {                        \
    case DType::F32: {                 \
      typedef float T;                 \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
    case DType::F16: {                 \
      typedef __half T;                \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
  }

inline void CheckSameDType(const CudaArray& a, const CudaArray& b) {
  if (a.dtype != b.dtype) {
    throw std::invalid_argument(std::string("dtype mismatch: ") + a.dtype_name() + " vs " +
                                b.dtype_name());
  }
}

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill and cast
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void FillKernel(T* out, float val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, gid, val);
}

void Fill(CudaArray* out, float val) {
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(out->dtype, T, {
    FillKernel<T><<<dim.grid, dim.block>>>(out->as<T>(), val, out->size);
  });
}

template <typename TI, typename TO>
__global__ void CastKernel(const TI* a, TO* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, gid, Load(a, gid));
}

void Cast(const CudaArray& a, CudaArray* out) {
  /**
   * Convert the (compact) array a into out, which has the same size but possibly another dtype.
   */
  if (a.size != out->size) throw std::invalid_argument("cast: size mismatch");
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, TI, {
    NEEDLE_DISPATCH(out->dtype, TO, {
      CastKernel<TI, TO><<<dim.grid, dim.block>>>(a.as<TI>(), out->as<TO>(), out->size);
    });
  });
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t StridedIndex(size_t gid, const CudaVec& shape, const CudaVec& strides,
                               size_t offset) {
  int64_t index = offset;
  size_t remain = gid;
  for (int i = (int)shape.size - 1; i >= 0; i--) {
    index += (int64_t)(remain % shape.data[i]) * strides.data[i];
    remain /= shape.data[i];
  }
  return (size_t)index;
}

template <typename T>
__global__ void CompactKernel(const T* a, T* out, size_t size, CudaVec shape, CudaVec strides,
                              size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[StridedIndex(gid, shape, strides, offset)];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    CompactKernel<T><<<dim.grid, dim.block>>>(a.as<T>(), out->as<T>(), out->size,
                                              VecToCuda(shape), VecToCuda(strides), offset);
  });
}

template <typename T>
__global__ void EwiseSetitemKernel(const T* a, T* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[StridedIndex(gid, shape, strides, offset)] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(a.size);
  NEEDLE_DISPATCH(a.dtype, T, {
    EwiseSetitemKernel<T><<<dim.grid, dim.block>>>(a.as<T>(), out->as<T>(), a.size,
                                                   VecToCuda(shape), VecToCuda(strides), offset);
  });
}

template <typename T>
__global__ void ScalarSetitemKernel(float val, T* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, StridedIndex(gid, shape, strides, offset), val);
}

void ScalarSetitem(size_t size, float val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  NEEDLE_DISPATCH(out->dtype, T, {
    ScalarSetitemKernel<T><<<dim.grid, dim.block>>>(val, out->as<T>(), size, VecToCuda(shape),
                                                    VecToCuda(strides), offset);
  });
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

struct OpAdd { __device__ float operator()(float x, float y) const { return x + y; } };
struct OpMul { __device__ float operator()(float x, float y) const { return x * y; } };
struct OpDiv { __device__ float operator()(float x, float y) const { return x / y; } };
struct OpPow { __device__ float operator()(float x, float y) const { return powf(x, y); } };
struct OpMax { __device__ float operator()(float x, float y) const { return fmaxf(x, y); } };
struct OpEq { __device__ float operator()(float x, float y) const { return (float)(x == y); } };
struct OpGe { __device__ float operator()(float x, float y) const { return (float)(x >= y); } };
struct OpLog { __device__ float operator()(float x) const { return logf(x); } };
struct OpExp { __device__ float operator()(float x) const { return expf(x); } };
struct OpTanh { __device__ float operator()(float x) const { return tanhf(x); } };

template <typename T, typename Op>
__global__ void EwiseBinaryKernel(const T* a, const T* b, T* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, gid, op(Load(a, gid), Load(b, gid)));
}

template <typename T, typename Op>
__global__ void ScalarBinaryKernel(const T* a, float val, T* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, gid, op(Load(a, gid), val));
}

template <typename T, typename Op>
__global__ void EwiseUnaryKernel(const T* a, T* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) Store(out, gid, op(Load(a, gid)));
}

template <typename Op>
void EwiseBinary(const CudaArray& a, const CudaArray& b, CudaArray* out, Op op) {
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    EwiseBinaryKernel<T, Op><<<dim.grid, dim.block>>>(a.as<T>(), b.as<T>(), out->as<T>(),
                                                      out->size, op);
  });
}

template <typename Op>
void ScalarBinary(const CudaArray& a, float val, CudaArray* out, Op op) {
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    ScalarBinaryKernel<T, Op><<<dim.grid, dim.block>>>(a.as<T>(), val, out->as<T>(), out->size,
                                                       op);
  });
}

template <typename Op>
void EwiseUnary(const CudaArray& a, CudaArray* out, Op op) {
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    EwiseUnaryKernel<T, Op><<<dim.grid, dim.block>>>(a.as<T>(), out->as<T>(), out->size, op);
  });
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpAdd());
}
void ScalarAdd(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpAdd());
}
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpMul());
}
void ScalarMul(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpMul());
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpDiv());
}
void ScalarDiv(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpDiv());
}
void EwisePower(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpPow());
}
void ScalarPower(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpPow());
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpMax());
}
void ScalarMaximum(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpMax());
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpEq());
}
void ScalarEq(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpEq());
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinary(a, b, out, OpGe());
}
void ScalarGe(const CudaArray& a, float val, CudaArray* out) {
  ScalarBinary(a, val, out, OpGe());
}
void EwiseLog(const CudaArray& a, CudaArray* out) { EwiseUnary(a, out, OpLog()); }
void EwiseExp(const CudaArray& a, CudaArray* out) { EwiseUnary(a, out, OpExp()); }
void EwiseTanh(const CudaArray& a, CudaArray* out) { EwiseUnary(a, out, OpTanh()); }

////////////////////////////////////////////////////////////////////////////////
// Matmul
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void MatmulKernel(const T* a, const T* b, T* out, uint32_t M, uint32_t N,
                             uint32_t P) {
  // Shared-memory tiles and the accumulator are float regardless of T.
  __shared__ float astile[TILE][TILE];
  __shared__ float bstile[TILE][TILE];
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;
  float tmp = 0;
  for (size_t ks = 0; ks < N; ks += TILE) {
    size_t a_col = ks + threadIdx.y;
    size_t b_row = ks + threadIdx.x;
    astile[threadIdx.x][threadIdx.y] = (a_col < N && row < M) ? Load(a, row * N + a_col) : 0.0f;
    bstile[threadIdx.x][threadIdx.y] = (b_row < N && col < P) ? Load(b, b_row * P + col) : 0.0f;
    __syncthreads();
    for (size_t k = 0; k < TILE; k++) {
      tmp += astile[threadIdx.x][k] * bstile[k][threadIdx.y];
    }
    __syncthreads();
  }
  if (row < M && col < P) {
    Store(out, row * P + col, tmp);
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  dim3 new_block(TILE, TILE);
  dim3 new_grid((M + TILE - 1) / TILE, (P + TILE - 1) / TILE);
  NEEDLE_DISPATCH(a.dtype, T, {
    MatmulKernel<T><<<new_grid, new_block>>>(a.as<T>(), b.as<T>(), out->as<T>(), M, N, P);
  });
}

template <typename T>
__global__ void MatmulBatchedKernel(const T* a, const T* b, T* out, uint32_t batch, uint32_t M,
                                    uint32_t N, uint32_t P, size_t a_batch_stride,
                                    size_t b_batch_stride) {
  // Same shared-memory tiling as MatmulKernel; blockIdx.z walks the batch (grid-stride, so
  // batches larger than the grid's z limit work too). Tiles and accumulator are float.
  __shared__ float astile[TILE][TILE];
  __shared__ float bstile[TILE][TILE];
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;
  for (size_t z = blockIdx.z; z < batch; z += gridDim.z) {
    const T* az = a + z * a_batch_stride;
    const T* bz = b + z * b_batch_stride;
    T* oz = out + z * (size_t)M * P;
    float tmp = 0;
    for (size_t ks = 0; ks < N; ks += TILE) {
      size_t a_col = ks + threadIdx.y;
      size_t b_row = ks + threadIdx.x;
      astile[threadIdx.x][threadIdx.y] = (a_col < N && row < M) ? Load(az, row * N + a_col) : 0.0f;
      bstile[threadIdx.x][threadIdx.y] = (b_row < N && col < P) ? Load(bz, b_row * P + col) : 0.0f;
      __syncthreads();
      for (size_t k = 0; k < TILE; k++) {
        tmp += astile[threadIdx.x][k] * bstile[k][threadIdx.y];
      }
      __syncthreads();
    }
    if (row < M && col < P) {
      Store(oz, row * P + col, tmp);
    }
  }
}

void MatmulBatched(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t batch,
                   uint32_t M, uint32_t N, uint32_t P, size_t a_batch_stride,
                   size_t b_batch_stride) {
  /**
   * Batched matrix multiplication: out[z] = a[z] @ b[z] for z < batch.
   *
   * a / b: compact arrays; matrix z starts at z * a_batch_stride / z * b_batch_stride elements
   * (stride 0: one matrix shared by the whole batch). out: compact batch x M x P.
   */
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  if (batch == 0 || M == 0 || P == 0) return;
  dim3 block(TILE, TILE);
  dim3 grid((M + TILE - 1) / TILE, (P + TILE - 1) / TILE, std::min<uint32_t>(batch, 65535));
  NEEDLE_DISPATCH(a.dtype, T, {
    MatmulBatchedKernel<T><<<grid, block>>>(a.as<T>(), b.as<T>(), out->as<T>(), batch, M, N, P,
                                            a_batch_stride, b_batch_stride);
  });
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void ReduceMaxKernel(const T* a, T* out, size_t size, size_t reduce_size) {
  // one block per output element (grid-stride), shared-memory tree reduction in float
  size_t bid = blockIdx.x;
  size_t tid = threadIdx.x;
  __shared__ float cache[BASE_THREAD_NUM];
  for (size_t i = bid; i < size; i += gridDim.x) {
    float tmp = -INFINITY;
    for (size_t j = tid; j < reduce_size; j += blockDim.x) {
      tmp = fmaxf(tmp, Load(a, i * reduce_size + j));
    }
    cache[tid] = tmp;
    __syncthreads();
    for (size_t s = 1; s < blockDim.x; s *= 2) {
      if (tid % (s * 2) == 0 && tid + s < blockDim.x) {
        cache[tid] = fmaxf(cache[tid], cache[tid + s]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      Store(out, i, cache[0]);
    }
    __syncthreads();
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    ReduceMaxKernel<T><<<dim.grid, dim.block>>>(a.as<T>(), out->as<T>(), out->size,
                                                reduce_size);
  });
}

template <typename T>
__global__ void ReduceSumKernel(const T* a, T* out, size_t size, size_t reduce_size) {
  // one block per output element (grid-stride), shared-memory tree reduction in float
  size_t bid = blockIdx.x;
  size_t tid = threadIdx.x;
  __shared__ float cache[BASE_THREAD_NUM];
  for (size_t i = bid; i < size; i += gridDim.x) {
    float tmp = 0;
    for (size_t j = tid; j < reduce_size; j += blockDim.x) {
      tmp += Load(a, i * reduce_size + j);
    }
    cache[tid] = tmp;
    __syncthreads();
    for (size_t s = 1; s < blockDim.x; s *= 2) {
      if (tid % (s * 2) == 0 && tid + s < blockDim.x) {
        cache[tid] += cache[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      Store(out, i, cache[0]);
    }
    __syncthreads();
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks (accumulating in float).
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CheckSameDType(a, *out);
  CudaDims dim = CudaOneDim(out->size);
  NEEDLE_DISPATCH(a.dtype, T, {
    ReduceSumKernel<T><<<dim.grid, dim.block>>>(a.as<T>(), out->as<T>(), out->size,
                                                reduce_size);
  });
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;
  m.attr("__dtypes__") = py::make_tuple("float32", "float16");

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t, std::string>(), py::arg("size"), py::arg("dtype") = "float32",
           py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def_property_readonly("dtype", &CudaArray::dtype_name)
      .def_property_readonly("itemsize", &CudaArray::itemsize)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    const size_t elem = a.itemsize();
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [elem](size_t& c) { return c * elem; });

    // copy memory to host
    char* host_ptr = (char*)std::malloc(std::max<size_t>(a.size, 1) * elem);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * elem, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::free(host_ptr);
      throw std::runtime_error(cudaGetErrorString(err));
    }

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array(py::dtype(a.dtype_name()), shape, numpy_strides, host_ptr + offset * elem,
                     deallocate_buffer);
  });

  // copy numpy array to GPU; the array must be C-contiguous with out's dtype
  m.def("from_numpy", [](py::array a, CudaArray* out) {
    if (std::string(py::str(a.dtype())) != out->dtype_name()) {
      throw std::invalid_argument("from_numpy: expected " + out->dtype_name() + " array, got " +
                                  std::string(py::str(a.dtype())));
    }
    if (!(a.flags() & py::array::c_style)) {
      throw std::invalid_argument("from_numpy: array must be C-contiguous");
    }
    if ((size_t)a.size() != out->size) throw std::invalid_argument("from_numpy: size mismatch");
    cudaError_t err =
        cudaMemcpy(out->ptr, a.data(), out->size * out->itemsize(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("cast", Cast);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);
  m.def("ewise_power", EwisePower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_batched", MatmulBatched);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
