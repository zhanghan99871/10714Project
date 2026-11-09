#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8

/**
 * Supported element types.
 *
 * float32 is stored and computed natively. float16 (IEEE 754 binary16, the same format as
 * numpy.float16) is a *storage* type: every kernel loads float16 elements, widens them to
 * float32, computes in float32 and rounds the result back to float16 (round-to-nearest-even).
 * In particular matmul and reductions always accumulate in float32, which is what mixed
 * precision training needs.
 */
enum class DType { F32, F16 };

struct half_t {
  uint16_t bits;
};

inline size_t ElemSize(DType dt) { return dt == DType::F32 ? sizeof(float) : sizeof(half_t); }

inline const char* DTypeName(DType dt) { return dt == DType::F32 ? "float32" : "float16"; }

inline DType ParseDType(const std::string& name) {
  if (name == "float32") return DType::F32;
  if (name == "float16") return DType::F16;
  throw std::invalid_argument("unsupported dtype: " + name);
}

/// float32 <-> float16 conversion (round-to-nearest-even, matches numpy.float16).
/// Based on F. Giesen's public domain float_to_half_fast3_rtne / half_to_float.
inline float bits_to_float(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint32_t float_to_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

inline uint16_t FloatToHalf(float value) {
  const uint32_t f32infty = 255u << 23;
  const uint32_t f16max = (127u + 16u) << 23;
  const uint32_t denorm_magic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
  uint32_t u = float_to_bits(value);
  const uint32_t sign = u & 0x80000000u;
  u ^= sign;
  uint16_t o;
  if (u >= f16max) {
    // overflow -> inf; NaN stays (quiet) NaN
    o = (u > f32infty) ? 0x7e00 : 0x7c00;
  } else if (u < (113u << 23)) {
    // result is subnormal or zero: let float addition do the rounding
    float f = bits_to_float(u) + bits_to_float(denorm_magic);
    o = (uint16_t)(float_to_bits(f) - denorm_magic);
  } else {
    const uint32_t mant_odd = (u >> 13) & 1u;
    u += ((uint32_t)(15 - 127) << 23) + 0xfffu;  // rebias exponent, rounding bias part 1
    u += mant_odd;                                // rounding bias part 2 (ties to even)
    o = (uint16_t)(u >> 13);
  }
  return (uint16_t)(o | (sign >> 16));
}

inline float HalfToFloat(uint16_t h) {
  const uint32_t shifted_exp = 0x7c00u << 13;
  uint32_t o = ((uint32_t)(h & 0x7fffu)) << 13;
  const uint32_t exp = shifted_exp & o;
  o += (uint32_t)(127 - 15) << 23;
  if (exp == shifted_exp) {
    o += (uint32_t)(128 - 16) << 23;  // inf / NaN
  } else if (exp == 0) {
    o += 1u << 23;  // zero / subnormal: renormalize
    o = float_to_bits(bits_to_float(o) - bits_to_float(113u << 23));
  }
  return bits_to_float(o | ((uint32_t)(h & 0x8000u) << 16));
}

/// Element access: every kernel reads and writes through these, computing in float.
inline float Load(const float* p, size_t i) { return p[i]; }
inline float Load(const half_t* p, size_t i) { return HalfToFloat(p[i].bits); }
inline void Store(float* p, size_t i, float v) { p[i] = v; }
inline void Store(half_t* p, size_t i, float v) { p[i].bits = FloatToHalf(v); }

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size, const std::string& dtype = "float32")
      : dtype(ParseDType(dtype)), size(size) {
    int ret = posix_memalign(&ptr, ALIGNMENT, std::max<size_t>(size, 1) * ElemSize(this->dtype));
    if (ret != 0) throw std::bad_alloc();
  }
  ~AlignedArray() { free(ptr); }
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
      typedef half_t T;                \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
  }

inline void CheckSameDType(const AlignedArray& a, const AlignedArray& b) {
  if (a.dtype != b.dtype) {
    throw std::invalid_argument(std::string("dtype mismatch: ") + a.dtype_name() + " vs " +
                                b.dtype_name());
  }
}

void Fill(AlignedArray* out, float val) {
  /**
   * Fill the values of an aligned array with val
   */
  NEEDLE_DISPATCH(out->dtype, T, {
    T* o = out->as<T>();
    for (size_t i = 0; i < out->size; i++) Store(o, i, val);
  });
}

void Cast(const AlignedArray& a, AlignedArray* out) {
  /**
   * Convert the (compact) array a into out, which has the same size but possibly another dtype.
   */
  if (a.size != out->size) throw std::invalid_argument("cast: size mismatch");
  NEEDLE_DISPATCH(a.dtype, TI, {
    NEEDLE_DISPATCH(out->dtype, TO, {
      const TI* src = a.as<TI>();
      TO* dst = out->as<TO>();
      for (size_t i = 0; i < a.size; i++) Store(dst, i, Load(src, i));
    });
  });
}

/// Visit every element of a strided view (row-major order), calling f(compact_idx, strided_idx).
template <typename F>
void ForEachStrided(const std::vector<int32_t>& shape, const std::vector<int32_t>& strides,
                    size_t offset, F f) {
  int n = shape.size();
  std::vector<int32_t> cur(n, 0);
  size_t total = 1;
  for (int i = 0; i < n; i++) total *= shape[i];
  for (size_t i = 0; i < total; i++) {
    int64_t index = offset;
    for (int j = 0; j < n; j++) index += (int64_t)cur[j] * strides[j];
    f(i, (size_t)index);
    for (int j = n - 1; j >= 0; j--) {
      cur[j]++;
      if (cur[j] < shape[j]) break;
      cur[j] = 0;
    }
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* src = a.as<T>();
    T* dst = out->as<T>();
    ForEachStrided(shape, strides, offset, [&](size_t i, size_t idx) { dst[i] = src[idx]; });
  });
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* src = a.as<T>();
    T* dst = out->as<T>();
    ForEachStrided(shape, strides, offset, [&](size_t i, size_t idx) { dst[idx] = src[i]; });
  });
}

void ScalarSetitem(const size_t size, float val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  NEEDLE_DISPATCH(out->dtype, T, {
    T* dst = out->as<T>();
    ForEachStrided(shape, strides, offset, [&](size_t, size_t idx) { Store(dst, idx, val); });
  });
}

/// Generic element-wise kernels; the arithmetic itself is always done in float.
template <typename Op>
void EwiseBinary(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    const T* pb = b.as<T>();
    T* po = out->as<T>();
    for (size_t i = 0; i < a.size; i++) Store(po, i, op(Load(pa, i), Load(pb, i)));
  });
}

template <typename Op>
void ScalarBinary(const AlignedArray& a, float val, AlignedArray* out, Op op) {
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    T* po = out->as<T>();
    for (size_t i = 0; i < a.size; i++) Store(po, i, op(Load(pa, i), val));
  });
}

template <typename Op>
void EwiseUnary(const AlignedArray& a, AlignedArray* out, Op op) {
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    T* po = out->as<T>();
    for (size_t i = 0; i < a.size; i++) Store(po, i, op(Load(pa, i)));
  });
}

struct OpAdd { float operator()(float x, float y) const { return x + y; } };
struct OpMul { float operator()(float x, float y) const { return x * y; } };
struct OpDiv { float operator()(float x, float y) const { return x / y; } };
struct OpPow { float operator()(float x, float y) const { return std::pow(x, y); } };
struct OpMax { float operator()(float x, float y) const { return std::max(x, y); } };
struct OpEq { float operator()(float x, float y) const { return (float)(x == y); } };
struct OpGe { float operator()(float x, float y) const { return (float)(x >= y); } };
struct OpLog { float operator()(float x) const { return std::log(x); } };
struct OpExp { float operator()(float x) const { return std::exp(x); } };
struct OpTanh { float operator()(float x) const { return std::tanh(x); } };

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpAdd());
}
void ScalarAdd(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpAdd());
}
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpMul());
}
void ScalarMul(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpMul());
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpDiv());
}
void ScalarDiv(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpDiv());
}
void EwisePower(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpPow());
}
void ScalarPower(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpPow());
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpMax());
}
void ScalarMaximum(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpMax());
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpEq());
}
void ScalarEq(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpEq());
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseBinary(a, b, out, OpGe());
}
void ScalarGe(const AlignedArray& a, float val, AlignedArray* out) {
  ScalarBinary(a, val, out, OpGe());
}
void EwiseLog(const AlignedArray& a, AlignedArray* out) { EwiseUnary(a, out, OpLog()); }
void EwiseExp(const AlignedArray& a, AlignedArray* out) { EwiseUnary(a, out, OpExp()); }
void EwiseTanh(const AlignedArray& a, AlignedArray* out) { EwiseUnary(a, out, OpTanh()); }

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix, using the naive
   * three-loop algorithm. Accumulation is always done in float32.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    const T* pb = b.as<T>();
    T* po = out->as<T>();
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < p; j++) {
        float tmp = 0;
        for (size_t k = 0; k < n; k++) {
          tmp += Load(pa, i * n + k) * Load(pb, k * p + j);
        }
        Store(po, i * p + j, tmp);
      }
    }
  });
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * sizeof(float));
  b = (const float*)__builtin_assume_aligned(b, TILE * sizeof(float));
  out = (float*)__builtin_assume_aligned(out, TILE * sizeof(float));

  for (size_t i = 0; i < TILE; i++) {
    for (size_t k = 0; k < TILE; k++) {
      float a_ik = a[i * TILE + k];
      for (size_t j = 0; j < TILE; j++) {
        out[i * TILE + j] += a_ik * b[k * TILE + j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * The multiplication is done tile-by-tile with AlignedDot(). Only called when m, n, p are all
   * multiples of TILE.
   *
   * For float32 the tiles are used in place. For float16 each tile is widened into an aligned
   * float32 buffer and every output tile is accumulated in float32 before being rounded once.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */
  CheckSameDType(a, b);
  CheckSameDType(a, *out);
  const size_t TT = TILE * TILE;
  if (a.dtype == DType::F32) {
    const float* pa = a.as<float>();
    const float* pb = b.as<float>();
    float* po = out->as<float>();
    Fill(out, 0.0f);
    for (size_t i = 0; i < m / TILE; i++) {
      for (size_t j = 0; j < p / TILE; j++) {
        for (size_t k = 0; k < n / TILE; k++) {
          AlignedDot(pa + i * n * TILE + k * TT, pb + k * p * TILE + j * TT,
                     po + i * p * TILE + j * TT);
        }
      }
    }
    return;
  }

  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    const T* pb = b.as<T>();
    T* po = out->as<T>();
    alignas(ALIGNMENT) float at[TT];
    alignas(ALIGNMENT) float bt[TT];
    alignas(ALIGNMENT) float acc[TT];
    for (size_t i = 0; i < m / TILE; i++) {
      for (size_t j = 0; j < p / TILE; j++) {
        std::fill(acc, acc + TT, 0.0f);
        for (size_t k = 0; k < n / TILE; k++) {
          const size_t a_off = i * n * TILE + k * TT;
          const size_t b_off = k * p * TILE + j * TT;
          for (size_t t = 0; t < TT; t++) {
            at[t] = Load(pa, a_off + t);
            bt[t] = Load(pb, b_off + t);
          }
          AlignedDot(at, bt, acc);
        }
        const size_t o_off = i * p * TILE + j * TT;
        for (size_t t = 0; t < TT; t++) Store(po, o_off + t, acc[t]);
      }
    }
  });
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    T* po = out->as<T>();
    for (size_t outer = 0; outer < out->size; outer++) {
      float tmp = Load(pa, reduce_size * outer);
      for (size_t inner = 1; inner < reduce_size; inner++) {
        tmp = std::max(Load(pa, inner + reduce_size * outer), tmp);
      }
      Store(po, outer, tmp);
    }
  });
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks. Accumulates in float32.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  CheckSameDType(a, *out);
  NEEDLE_DISPATCH(a.dtype, T, {
    const T* pa = a.as<T>();
    T* po = out->as<T>();
    for (size_t outer = 0; outer < out->size; outer++) {
      float tmp = Load(pa, reduce_size * outer);
      for (size_t inner = 1; inner < reduce_size; inner++) {
        tmp += Load(pa, inner + reduce_size * outer);
      }
      Store(po, outer, tmp);
    }
  });
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;
  m.attr("__dtypes__") = py::make_tuple("float32", "float16");

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t, std::string>(), py::arg("size"), py::arg("dtype") = "float32",
           py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size)
      .def_property_readonly("dtype", &AlignedArray::dtype_name)
      .def_property_readonly("itemsize", &AlignedArray::itemsize);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    const size_t elem = a.itemsize();
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [elem](size_t& c) { return c * elem; });
    const char* base = static_cast<const char*>(a.ptr) + offset * elem;
    return py::array(py::dtype(a.dtype_name()), shape, numpy_strides, base);
  });

  // convert from numpy (with copying); the array must be C-contiguous with out's dtype
  m.def("from_numpy", [](py::array a, AlignedArray* out) {
    if (std::string(py::str(a.dtype())) != out->dtype_name()) {
      throw std::invalid_argument("from_numpy: expected " + out->dtype_name() + " array, got " +
                                  std::string(py::str(a.dtype())));
    }
    if (!(a.flags() & py::array::c_style)) {
      throw std::invalid_argument("from_numpy: array must be C-contiguous");
    }
    if ((size_t)a.size() != out->size) throw std::invalid_argument("from_numpy: size mismatch");
    std::memcpy(out->ptr, a.data(), out->size * out->itemsize());
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
