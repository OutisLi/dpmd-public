// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compressed graph-native DPA4C descriptor.
//
// Widths up to 32 assign one warp to each destination node. Independent
// sub-warps process multiple edges concurrently when the channel width permits.
// Widths above 32 assign one thread block to each node and one thread to each
// channel, preserving channel-local state without multiplying register usage.
//
// The complete descriptor remains center-local: radial spline lookup, first
// moments, edge feedback, second moments, and the six-family invariant readout
// execute in one forward kernel. The forward retains both center moments and
// the smooth-degree normalization; the backward therefore needs only the two
// reverse edge scans required by the feedback dependency.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <type_traits>

namespace {

constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kBasisDim = 9;
constexpr unsigned kWarpMask = 0xffffffffu;

#define DPA4C_CHECK_LAUNCH(name)                                              \
  do {                                                                        \
    const cudaError_t error = cudaGetLastError();                             \
    TORCH_CHECK(error == cudaSuccess, name, ": ", cudaGetErrorString(error)); \
  } while (0)

template <bool Canonical, typename index_t>
__device__ __forceinline__ long edge_at_position(
    long position, const index_t* destination_order) {
  if constexpr (Canonical) {
    return position;
  }
  return static_cast<long>(destination_order[position]);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kWarpMask, value, offset);
  }
  return __shfl_sync(kWarpMask, value, 0);
}

template <int Width>
__device__ __forceinline__ float reduce_channel_groups(float value) {
  if constexpr (Width < kWarpSize) {
#pragma unroll
    for (int offset = Width; offset < kWarpSize; offset <<= 1) {
      value += __shfl_xor_sync(kWarpMask, value, offset);
    }
  }
  return value;
}

template <int Width>
__device__ __forceinline__ unsigned subwarp_mask(int leader) {
  if constexpr (Width == kWarpSize) {
    return kWarpMask;
  } else {
    return ((1u << Width) - 1u) << leader;
  }
}

template <int Width>
__device__ __forceinline__ float subwarp_sum(float value, unsigned mask) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, Width);
  }
  return value;
}

struct TableLocation {
  int index;
  float coordinate;
  bool clamped;
};

__device__ __forceinline__ TableLocation locate_table(float radius,
                                                      float stride,
                                                      float table_max,
                                                      int interval_count) {
  const float coordinate = fminf(fmaxf(radius, 0.0f), table_max);
  int index = static_cast<int>(__fdividef(coordinate, stride));
  index = min(index, interval_count - 1);
  return {
      index,
      coordinate - static_cast<float>(index) * stride,
      radius >= table_max,
  };
}

__device__ __forceinline__ void load_coefficients(const float* table,
                                                  const TableLocation& location,
                                                  int channel,
                                                  int width,
                                                  float2& c01,
                                                  float2& c23,
                                                  float2& c45) {
  const long offset = static_cast<long>(location.index) * width * 6 +
                      static_cast<long>(channel) * 6;
  c01 = __ldg(reinterpret_cast<const float2*>(table + offset));
  c23 = __ldg(reinterpret_cast<const float2*>(table + offset + 2));
  c45 = __ldg(reinterpret_cast<const float2*>(table + offset + 4));
}

__device__ __forceinline__ float evaluate_table(const float* table,
                                                const TableLocation& location,
                                                int channel,
                                                int width) {
  float2 c01, c23, c45;
  load_coefficients(table, location, channel, width, c01, c23, c45);
  const float x = location.coordinate;
  return c01.x +
         (c01.y + (c23.x + (c23.y + (c45.x + c45.y * x) * x) * x) * x) * x;
}

__device__ __forceinline__ float2 evaluate_table_with_derivative(
    const float* table, const TableLocation& location, int channel, int width) {
  float2 c01, c23, c45;
  load_coefficients(table, location, channel, width, c01, c23, c45);
  const float x = location.coordinate;
  const float value =
      c01.x + (c01.y + (c23.x + (c23.y + (c45.x + c45.y * x) * x) * x) * x) * x;
  const float derivative =
      c01.y + (2.0f * c23.x +
               (3.0f * c23.y + (4.0f * c45.x + 5.0f * c45.y * x) * x) * x) *
                  x;
  return make_float2(value, location.clamped ? 0.0f : derivative);
}

__device__ __forceinline__ float c3_envelope(float radius, float rcut) {
  const float u = fminf(fmaxf(__fdividef(rcut - radius, rcut), 0.0f), 1.0f);
  const float x = 1.0f - u;
  const float series =
      1.0f + x * (4.0f + x * (10.0f + x * (20.0f + 35.0f * x)));
  const float u2 = u * u;
  return u2 * u2 * series;
}

__device__ __forceinline__ float c3_envelope_derivative(float radius,
                                                        float rcut) {
  if (radius <= 0.0f || radius >= rcut) {
    return 0.0f;
  }
  const float x = __fdividef(radius, rcut);
  const float u = 1.0f - x;
  const float x2 = x * x;
  const float x3 = x2 * x;
  const float series =
      1.0f + 4.0f * x + 10.0f * x2 + 20.0f * x3 + 35.0f * x2 * x2;
  const float series_derivative = 4.0f + 20.0f * x + 60.0f * x2 + 140.0f * x3;
  const float u2 = u * u;
  const float derivative_x =
      -4.0f * u2 * u * series + u2 * u2 * series_derivative;
  return __fdividef(derivative_x, rcut);
}

struct EdgeGeometry {
  float x;
  float y;
  float z;
  float radius;
  float envelope;
  float basis[kBasisDim];
  int source_type;
};

template <typename index_t>
__device__ __forceinline__ EdgeGeometry load_geometry(long edge,
                                                      long edge_count,
                                                      float rcut,
                                                      float eps,
                                                      const float* edge_vec,
                                                      const index_t* edge_index,
                                                      const long* atype) {
  EdgeGeometry geometry;
  const long source = static_cast<long>(edge_index[edge]);
  geometry.source_type = static_cast<int>(atype[source]);
  geometry.x = edge_vec[edge * 3 + 0];
  geometry.y = edge_vec[edge * 3 + 1];
  geometry.z = edge_vec[edge * 3 + 2];
  const float square = geometry.x * geometry.x + geometry.y * geometry.y +
                       geometry.z * geometry.z + eps * eps;
  geometry.radius = sqrtf(square);
  const float inverse_radius = __fdividef(1.0f, geometry.radius);
  const float ux = geometry.x * inverse_radius;
  const float uy = geometry.y * inverse_radius;
  const float uz = geometry.z * inverse_radius;
  const float q = ux * ux + uy * uy + uz * uz;
  constexpr float kSqrtThree = 1.7320508075688772935f;
  geometry.envelope = c3_envelope(geometry.radius, rcut);
  geometry.basis[0] = 1.0f;
  geometry.basis[1] = ux;
  geometry.basis[2] = uy;
  geometry.basis[3] = uz;
  geometry.basis[4] = kSqrtThree * ux * uy;
  geometry.basis[5] = kSqrtThree * uy * uz;
  geometry.basis[6] = 0.5f * (3.0f * uz * uz - q);
  geometry.basis[7] = kSqrtThree * ux * uz;
  geometry.basis[8] = 0.5f * kSqrtThree * (ux * ux - uy * uy);
  return geometry;
}

template <int Width>
__device__ __forceinline__ EdgeGeometry broadcast_geometry(EdgeGeometry value,
                                                           int leader,
                                                           unsigned mask) {
  value.x = __shfl_sync(mask, value.x, leader);
  value.y = __shfl_sync(mask, value.y, leader);
  value.z = __shfl_sync(mask, value.z, leader);
  value.radius = __shfl_sync(mask, value.radius, leader);
  value.envelope = __shfl_sync(mask, value.envelope, leader);
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    value.basis[k] = __shfl_sync(mask, value.basis[k], leader);
  }
  value.source_type = __shfl_sync(mask, value.source_type, leader);
  return value;
}

template <int Width>
__device__ __forceinline__ TableLocation broadcast_location(TableLocation value,
                                                            int leader,
                                                            unsigned mask) {
  value.index = __shfl_sync(mask, value.index, leader);
  value.coordinate = __shfl_sync(mask, value.coordinate, leader);
  value.clamped = __shfl_sync(mask, value.clamped, leader);
  return value;
}

template <int Width>
__device__ __forceinline__ float edge_amplitude(const EdgeGeometry& geometry,
                                                const TableLocation& location,
                                                int channel,
                                                int center_type,
                                                const float* table,
                                                const float* type_embedding) {
  const float radial = evaluate_table(table, location, channel, Width);
  const float center =
      __ldg(type_embedding + static_cast<long>(center_type) * Width + channel);
  const float source =
      __ldg(type_embedding + static_cast<long>(geometry.source_type) * Width +
            channel);
  return radial + geometry.envelope * (center + source);
}

__device__ __forceinline__ float gate_activation(float value) {
  return tanhf(value);
}

__device__ __forceinline__ float gate_derivative(float value) {
  const float activation = tanhf(value);
  return 1.0f - activation * activation;
}

template <int Width, bool Canonical, typename index_t>
__device__ __forceinline__ void compute_moment_state(
    long node,
    long edge_count,
    int center_type,
    int interval_count,
    float table_stride,
    float table_max,
    float rcut,
    float eps,
    float degree_floor,
    const float* edge_vec,
    const index_t* edge_index,
    const bool* edge_mask,
    const index_t* destination_order,
    const long* destination_row_ptr,
    const long* atype,
    const float* table,
    const float* type_embedding,
    const float* feedback_weight,
    float (&first)[kBasisDim],
    float (&second)[kBasisDim],
    float& normalizer) {
  constexpr int kSubwarps = kWarpSize / Width;
  const int lane = threadIdx.x & 31;
  const int channel = lane & (Width - 1);
  const int subgroup = lane / Width;
  const int leader = subgroup * Width;
  const unsigned mask = subwarp_mask<Width>(leader);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];
  float degree = 0.0f;

  // === Pass 1. Accumulate the unnormalized first moments and smooth degree ===
  for (long position = begin + subgroup; position < end;
       position += kSubwarps) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == leader) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<Width>(geometry, leader, mask);
    location = broadcast_location<Width>(location, leader, mask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      first[k] = fmaf(amplitude, geometry.basis[k], first[k]);
    }
    degree = fmaf(geometry.envelope, geometry.envelope, degree);
  }
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    first[k] = reduce_channel_groups<Width>(first[k]);
  }
  degree = __fdividef(warp_sum(degree), static_cast<float>(Width));
  normalizer = rsqrtf(degree + degree_floor);
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    first[k] *= normalizer;
  }

  // === Pass 2. Apply edge feedback and accumulate the second moments ===
  for (long position = begin + subgroup; position < end;
       position += kSubwarps) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == leader) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<Width>(geometry, leader, mask);
    location = broadcast_location<Width>(location, leader, mask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
    float feedback_input = 0.0f;
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      float projection = 0.0f;
      float basis_norm = 0.0f;
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        projection = fmaf(first[k], geometry.basis[k], projection);
        basis_norm = fmaf(geometry.basis[k], geometry.basis[k], basis_norm);
      }
      const float context = projection - normalizer * amplitude * basis_norm;
      feedback_input = fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
                            context, feedback_input);
    }
    const float modulated =
        amplitude * (1.0f + gate_activation(feedback_input));
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      second[k] = fmaf(modulated, geometry.basis[k], second[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    second[k] = reduce_channel_groups<Width>(second[k]) * normalizer;
  }
}

struct Matrix3 {
  float value[3][3];
};

__device__ __forceinline__ Matrix3
packed_to_stf(float q0, float q1, float q2, float q3, float q4) {
  constexpr float kInvSqrtTwo = 0.7071067811865475244f;
  constexpr float kInvSqrtSix = 0.4082482904638630164f;
  Matrix3 matrix;
  matrix.value[0][0] = -q2 * kInvSqrtSix + q4 * kInvSqrtTwo;
  matrix.value[1][1] = -q2 * kInvSqrtSix - q4 * kInvSqrtTwo;
  matrix.value[2][2] = 2.0f * q2 * kInvSqrtSix;
  matrix.value[0][1] = matrix.value[1][0] = q0 * kInvSqrtTwo;
  matrix.value[1][2] = matrix.value[2][1] = q1 * kInvSqrtTwo;
  matrix.value[0][2] = matrix.value[2][0] = q3 * kInvSqrtTwo;
  return matrix;
}

__device__ __forceinline__ void matrix_vector(const Matrix3& matrix,
                                              const float (&vector)[3],
                                              float (&output)[3]) {
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    output[row] = 0.0f;
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      output[row] =
          fmaf(matrix.value[row][column], vector[column], output[row]);
    }
  }
}

__device__ __forceinline__ Matrix3 matrix_product(const Matrix3& left,
                                                  const Matrix3& right) {
  Matrix3 output{};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
#pragma unroll
      for (int inner = 0; inner < 3; ++inner) {
        output.value[row][column] =
            fmaf(left.value[row][inner], right.value[inner][column],
                 output.value[row][column]);
      }
    }
  }
  return output;
}

__device__ __forceinline__ void assemble_invariants(float scalar,
                                                    const float (&vector)[3],
                                                    const float (&packed)[5],
                                                    Matrix3& tensor,
                                                    float (&invariants)[6]) {
  tensor = packed_to_stf(packed[0], packed[1], packed[2], packed[3], packed[4]);
  float tensor_vector[3];
  matrix_vector(tensor, vector, tensor_vector);
  const Matrix3 tensor_squared = matrix_product(tensor, tensor);
  const Matrix3 tensor_cubed = matrix_product(tensor_squared, tensor);
  invariants[0] = scalar;
  invariants[1] =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  invariants[2] = 0.0f;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      invariants[2] = fmaf(tensor.value[row][column], tensor.value[row][column],
                           invariants[2]);
    }
  }
  invariants[3] = vector[0] * tensor_vector[0] + vector[1] * tensor_vector[1] +
                  vector[2] * tensor_vector[2];
  invariants[4] = tensor_cubed.value[0][0] + tensor_cubed.value[1][1] +
                  tensor_cubed.value[2][2];
  invariants[5] = tensor_vector[0] * tensor_vector[0] +
                  tensor_vector[1] * tensor_vector[1] +
                  tensor_vector[2] * tensor_vector[2];
}

template <int Width>
__device__ __forceinline__ void project_invariants(
    long node,
    int center_type,
    const float (&second)[kBasisDim],
    const float* type_embedding,
    const float* scalar_weight,
    const float* vector_weight,
    const float* tensor_weight,
    float(&scalar),
    float (&vector)[3],
    Matrix3& tensor,
    float (&invariants)[6]) {
  const int lane = threadIdx.x & 31;
  scalar = 0.0f;
#pragma unroll
  for (int component = 0; component < 3; ++component) {
    vector[component] = 0.0f;
  }
  float packed[5] = {};
  for (int input = 0; input < Width; ++input) {
    const float scalar_input = __shfl_sync(kWarpMask, second[0], input);
    float vector_input[3];
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      vector_input[component] =
          __shfl_sync(kWarpMask, second[1 + component], input);
    }
    float tensor_input[5];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      tensor_input[component] =
          __shfl_sync(kWarpMask, second[4 + component], input);
    }
    if (lane < Width) {
      scalar =
          fmaf(scalar_input,
               __ldg(scalar_weight + static_cast<long>(input) * Width + lane),
               scalar);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        vector[component] =
            fmaf(vector_input[component],
                 __ldg(vector_weight + static_cast<long>(input) * Width + lane),
                 vector[component]);
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            fmaf(tensor_input[component],
                 __ldg(tensor_weight + static_cast<long>(input) * Width + lane),
                 packed[component]);
      }
    }
  }
  if (lane < Width) {
    scalar +=
        __ldg(type_embedding + static_cast<long>(center_type) * Width + lane);
  }
  assemble_invariants(scalar, vector, packed, tensor, invariants);
}

template <int Width>
__device__ __forceinline__ void project_invariants_wide(
    int channel,
    int center_type,
    const float* second,
    const float* type_embedding,
    const float* scalar_weight,
    const float* vector_weight,
    const float* tensor_weight,
    float& scalar,
    float (&vector)[3],
    Matrix3& tensor,
    float (&invariants)[6]) {
  static_assert(Width > kWarpSize);
  scalar = 0.0f;
#pragma unroll
  for (int component = 0; component < 3; ++component) {
    vector[component] = 0.0f;
  }
  float packed[5] = {};
  for (int input = 0; input < Width; ++input) {
    const long weight_offset = static_cast<long>(input) * Width + channel;
    scalar = fmaf(second[input], __ldg(scalar_weight + weight_offset), scalar);
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      vector[component] =
          fmaf(second[(1 + component) * Width + input],
               __ldg(vector_weight + weight_offset), vector[component]);
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] =
          fmaf(second[(4 + component) * Width + input],
               __ldg(tensor_weight + weight_offset), packed[component]);
    }
  }
  scalar +=
      __ldg(type_embedding + static_cast<long>(center_type) * Width + channel);
  assemble_invariants(scalar, vector, packed, tensor, invariants);
}

template <int Width, bool Canonical, typename index_t>
__global__ __launch_bounds__(kThreads, 2) void dpa4c_forward_kernel(
    long node_count,
    long edge_count,
    int interval_count,
    float table_stride,
    float table_max,
    float rcut,
    float eps,
    float degree_floor,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ table,
    const float* __restrict__ type_embedding,
    const float* __restrict__ feedback_weight,
    const float* __restrict__ scalar_weight,
    const float* __restrict__ vector_weight,
    const float* __restrict__ tensor_weight,
    float* __restrict__ descriptor,
    float* __restrict__ state) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x / kWarpSize;
  const long node = static_cast<long>(blockIdx.x) * warps_per_block + warp;
  if (node >= node_count) {
    return;
  }
  int center_type = lane == 0 ? static_cast<int>(atype[node]) : 0;
  center_type = __shfl_sync(kWarpMask, center_type, 0);
  float first[kBasisDim] = {};
  float second[kBasisDim] = {};
  float normalizer = 0.0f;
  compute_moment_state<Width, Canonical, index_t>(
      node, edge_count, center_type, interval_count, table_stride, table_max,
      rcut, eps, degree_floor, edge_vec, edge_index, edge_mask,
      destination_order, destination_row_ptr, atype, table, type_embedding,
      feedback_weight, first, second, normalizer);

  if (lane < Width) {
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      state[(node * 19 + k) * Width + lane] = first[k];
      state[(node * 19 + 9 + k) * Width + lane] = second[k];
    }
    state[(node * 19 + 18) * Width + lane] = normalizer;
  }

  float scalar;
  float vector[3];
  Matrix3 tensor;
  float invariants[6];
  project_invariants<Width>(node, center_type, second, type_embedding,
                            scalar_weight, vector_weight, tensor_weight, scalar,
                            vector, tensor, invariants);

  if (lane < Width) {
#pragma unroll
    for (int family = 0; family < 6; ++family) {
      descriptor[(node * Width + lane) * 6 + family] = invariants[family];
    }
  }
}

__device__ __forceinline__ void differentiate_invariants(
    const float (&vector)[3],
    const Matrix3& tensor,
    const float (&d_invariant)[6],
    float& d_scalar,
    float (&d_vector)[3],
    float (&d_packed)[5]) {
  float tensor_vector[3];
  matrix_vector(tensor, vector, tensor_vector);
  float tensor2_vector[3];
  matrix_vector(tensor, tensor_vector, tensor2_vector);
#pragma unroll
  for (int component = 0; component < 3; ++component) {
    d_vector[component] = 2.0f * d_invariant[1] * vector[component] +
                          2.0f * d_invariant[3] * tensor_vector[component] +
                          2.0f * d_invariant[5] * tensor2_vector[component];
  }

  const Matrix3 tensor_squared = matrix_product(tensor, tensor);
  Matrix3 d_tensor{};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      d_tensor.value[row][column] =
          2.0f * d_invariant[2] * tensor.value[row][column] +
          d_invariant[3] * vector[row] * vector[column] +
          3.0f * d_invariant[4] * tensor_squared.value[row][column] +
          2.0f * d_invariant[5] * tensor_vector[row] * vector[column];
    }
  }
  constexpr float kInvSqrtTwo = 0.7071067811865475244f;
  constexpr float kInvSqrtSix = 0.4082482904638630164f;
  d_packed[0] = (d_tensor.value[0][1] + d_tensor.value[1][0]) * kInvSqrtTwo;
  d_packed[1] = (d_tensor.value[1][2] + d_tensor.value[2][1]) * kInvSqrtTwo;
  d_packed[2] = (-d_tensor.value[0][0] - d_tensor.value[1][1] +
                 2.0f * d_tensor.value[2][2]) *
                kInvSqrtSix;
  d_packed[3] = (d_tensor.value[0][2] + d_tensor.value[2][0]) * kInvSqrtTwo;
  d_packed[4] = (d_tensor.value[0][0] - d_tensor.value[1][1]) * kInvSqrtTwo;
  d_scalar = d_invariant[0];
}

template <int Width>
__device__ __forceinline__ void invariant_backward(
    long node,
    const float (&second)[kBasisDim],
    int center_type,
    const float* descriptor_gradient,
    const float* type_embedding,
    const float* scalar_weight,
    const float* vector_weight,
    const float* tensor_weight,
    float (&d_second)[kBasisDim]) {
  const int lane = threadIdx.x & 31;
  float scalar;
  float vector[3];
  Matrix3 tensor;
  float invariants[6];
  project_invariants<Width>(node, center_type, second, type_embedding,
                            scalar_weight, vector_weight, tensor_weight, scalar,
                            vector, tensor, invariants);

  float d_invariant[6] = {};
  if (lane < Width) {
#pragma unroll
    for (int family = 0; family < 6; ++family) {
      d_invariant[family] =
          __ldg(descriptor_gradient + (node * Width + lane) * 6 + family);
    }
  }

  float d_vector[3];
  float d_packed[5];
  float d_scalar;
  differentiate_invariants(vector, tensor, d_invariant, d_scalar, d_vector,
                           d_packed);
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_second[k] = 0.0f;
  }
  for (int output = 0; output < Width; ++output) {
    const float scalar_gradient = __shfl_sync(kWarpMask, d_scalar, output);
    float vector_gradient[3];
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      vector_gradient[component] =
          __shfl_sync(kWarpMask, d_vector[component], output);
    }
    float tensor_gradient[5];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      tensor_gradient[component] =
          __shfl_sync(kWarpMask, d_packed[component], output);
    }
    if (lane < Width) {
      d_second[0] =
          fmaf(scalar_gradient,
               __ldg(scalar_weight + static_cast<long>(lane) * Width + output),
               d_second[0]);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_second[1 + component] = fmaf(
            vector_gradient[component],
            __ldg(vector_weight + static_cast<long>(lane) * Width + output),
            d_second[1 + component]);
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_second[4 + component] = fmaf(
            tensor_gradient[component],
            __ldg(tensor_weight + static_cast<long>(lane) * Width + output),
            d_second[4 + component]);
      }
    }
  }
  const int channel = lane & (Width - 1);
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_second[k] = __shfl_sync(kWarpMask, d_second[k], channel);
  }
}

__device__ __forceinline__ void basis_vjp(const EdgeGeometry& geometry,
                                          const float (&d_basis)[kBasisDim],
                                          float radial_gradient,
                                          float* output) {
  const float inverse_radius = __fdividef(1.0f, geometry.radius);
  const float ux = geometry.x * inverse_radius;
  const float uy = geometry.y * inverse_radius;
  const float uz = geometry.z * inverse_radius;
  constexpr float kSqrtThree = 1.7320508075688772935f;
  float dux =
      d_basis[1] +
      kSqrtThree * (d_basis[4] * uy + d_basis[7] * uz + d_basis[8] * ux) -
      d_basis[6] * ux;
  float duy =
      d_basis[2] +
      kSqrtThree * (d_basis[4] * ux + d_basis[5] * uz - d_basis[8] * uy) -
      d_basis[6] * uy;
  float duz = d_basis[3] + kSqrtThree * (d_basis[5] * uy + d_basis[7] * ux) +
              2.0f * d_basis[6] * uz;
  const float unit_dot = ux * dux + uy * duy + uz * duz;
  dux = (dux - ux * unit_dot) * inverse_radius;
  duy = (duy - uy * unit_dot) * inverse_radius;
  duz = (duz - uz * unit_dot) * inverse_radius;
  output[0] = dux + radial_gradient * ux;
  output[1] = duy + radial_gradient * uy;
  output[2] = duz + radial_gradient * uz;
}

template <int Width, bool Canonical, typename index_t>
__global__ __launch_bounds__(kThreads, 2) void dpa4c_backward_kernel(
    long node_count,
    long edge_count,
    int interval_count,
    float table_stride,
    float table_max,
    float rcut,
    float eps,
    float degree_floor,
    const float* __restrict__ descriptor_gradient,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ table,
    const float* __restrict__ type_embedding,
    const float* __restrict__ feedback_weight,
    const float* __restrict__ scalar_weight,
    const float* __restrict__ vector_weight,
    const float* __restrict__ tensor_weight,
    const float* __restrict__ state,
    float* __restrict__ edge_gradient) {
  constexpr int kSubwarps = kWarpSize / Width;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x / kWarpSize;
  const long node = static_cast<long>(blockIdx.x) * warps_per_block + warp;
  if (node >= node_count) {
    return;
  }
  const int channel = lane & (Width - 1);
  const int subgroup = lane / Width;
  const int leader = subgroup * Width;
  const unsigned mask = subwarp_mask<Width>(leader);
  int center_type = lane == 0 ? static_cast<int>(atype[node]) : 0;
  center_type = __shfl_sync(kWarpMask, center_type, 0);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];

  // === Step 1. Load the center-local state retained by the forward ===
  float first[kBasisDim] = {};
  float second[kBasisDim] = {};
  if (lane < Width) {
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      first[k] = __ldg(state + (node * 19 + k) * Width + lane);
      second[k] = __ldg(state + (node * 19 + 9 + k) * Width + lane);
    }
  }
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    first[k] = __shfl_sync(kWarpMask, first[k], channel);
    second[k] = __shfl_sync(kWarpMask, second[k], channel);
  }
  const float normalizer = __ldg(state + (node * 19 + 18) * Width);

  // === Step 2. Differentiate the node-only invariant readout ===
  float d_second[kBasisDim] = {};
  invariant_backward<Width>(node, second, center_type, descriptor_gradient,
                            type_embedding, scalar_weight, vector_weight,
                            tensor_weight, d_second);
  float d_sum_second[kBasisDim];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_sum_second[k] = normalizer * d_second[k];
  }

  // === Step 3. Accumulate feedback gradients into X and n ===
  float d_first[kBasisDim] = {};
  float d_normalizer_edges = 0.0f;
  for (long position = begin + subgroup; position < end;
       position += kSubwarps) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == leader) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<Width>(geometry, leader, mask);
    location = broadcast_location<Width>(location, leader, mask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
    float context[3] = {};
    float basis_norm[3] = {};
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        context[degree_index] =
            fmaf(first[k], geometry.basis[k], context[degree_index]);
        basis_norm[degree_index] = fmaf(geometry.basis[k], geometry.basis[k],
                                        basis_norm[degree_index]);
      }
      context[degree_index] -=
          normalizer * amplitude * basis_norm[degree_index];
    }
    float activation_input = 0.0f;
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      activation_input =
          fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
               context[degree_index], activation_input);
    }
    const float activation = gate_activation(activation_input);
    float d_modulated = 0.0f;
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_modulated = fmaf(d_sum_second[k], geometry.basis[k], d_modulated);
    }
    const float d_activation = d_modulated * amplitude;
    const float d_activation_input =
        d_activation * gate_derivative(activation_input);
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      const float d_context =
          d_activation_input *
          __ldg(feedback_weight + channel * 3 + degree_index);
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        d_first[k] = fmaf(d_context, geometry.basis[k], d_first[k]);
      }
      d_normalizer_edges -= d_context * amplitude * basis_norm[degree_index];
    }
  }
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_first[k] = reduce_channel_groups<Width>(d_first[k]);
  }
  float d_normalizer =
      warp_sum(d_normalizer_edges) +
      warp_sum(subgroup == 0
                   ? (d_first[0] * first[0] + d_first[1] * first[1] +
                      d_first[2] * first[2] + d_first[3] * first[3] +
                      d_first[4] * first[4] + d_first[5] * first[5] +
                      d_first[6] * first[6] + d_first[7] * first[7] +
                      d_first[8] * first[8] + d_second[0] * second[0] +
                      d_second[1] * second[1] + d_second[2] * second[2] +
                      d_second[3] * second[3] + d_second[4] * second[4] +
                      d_second[5] * second[5] + d_second[6] * second[6] +
                      d_second[7] * second[7] + d_second[8] * second[8]) /
                         normalizer
                   : 0.0f);
  const float d_degree =
      -0.5f * d_normalizer * normalizer * normalizer * normalizer;
  float d_sum_first[kBasisDim];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_sum_first[k] = normalizer * d_first[k];
  }

  // === Step 4. Recompute each edge and assemble its geometry VJP ===
  for (long position = begin + subgroup; position < end;
       position += kSubwarps) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      if (lane == leader) {
        edge_gradient[edge * 3 + 0] = 0.0f;
        edge_gradient[edge * 3 + 1] = 0.0f;
        edge_gradient[edge * 3 + 2] = 0.0f;
      }
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == leader) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<Width>(geometry, leader, mask);
    location = broadcast_location<Width>(location, leader, mask);
    const float2 radial =
        evaluate_table_with_derivative(table, location, channel, Width);
    const float type_value =
        __ldg(type_embedding + static_cast<long>(center_type) * Width +
              channel) +
        __ldg(type_embedding + static_cast<long>(geometry.source_type) * Width +
              channel);
    const float amplitude = radial.x + geometry.envelope * type_value;
    float context[3] = {};
    float basis_norm[3] = {};
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        context[degree_index] =
            fmaf(first[k], geometry.basis[k], context[degree_index]);
        basis_norm[degree_index] = fmaf(geometry.basis[k], geometry.basis[k],
                                        basis_norm[degree_index]);
      }
      context[degree_index] -=
          normalizer * amplitude * basis_norm[degree_index];
    }
    float activation_input = 0.0f;
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      activation_input =
          fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
               context[degree_index], activation_input);
    }
    const float activation = gate_activation(activation_input);
    const float modulated = amplitude * (1.0f + activation);
    float d_modulated = 0.0f;
    float d_basis[kBasisDim] = {};
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_modulated = fmaf(d_sum_second[k], geometry.basis[k], d_modulated);
      d_basis[k] = d_sum_second[k] * modulated;
    }
    const float d_activation = d_modulated * amplitude;
    const float d_activation_input =
        d_activation * gate_derivative(activation_input);
    float d_amplitude = d_modulated * (1.0f + activation);
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      const float d_context =
          d_activation_input *
          __ldg(feedback_weight + channel * 3 + degree_index);
      d_amplitude -= d_context * normalizer * basis_norm[degree_index];
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        d_basis[k] += d_context * (first[k] - 2.0f * normalizer * amplitude *
                                                  geometry.basis[k]);
      }
    }
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_amplitude = fmaf(d_sum_first[k], geometry.basis[k], d_amplitude);
      d_basis[k] = fmaf(d_sum_first[k], amplitude, d_basis[k]);
    }
    float radial_gradient = d_amplitude * radial.y;
    float envelope_gradient = d_amplitude * type_value;
    radial_gradient = subwarp_sum<Width>(radial_gradient, mask);
    envelope_gradient = subwarp_sum<Width>(envelope_gradient, mask);
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_basis[k] = subwarp_sum<Width>(d_basis[k], mask);
    }
    if (lane == leader) {
      envelope_gradient += 2.0f * geometry.envelope * d_degree;
      radial_gradient +=
          envelope_gradient * c3_envelope_derivative(geometry.radius, rcut);
      float output[3];
      basis_vjp(geometry, d_basis, radial_gradient, output);
      edge_gradient[edge * 3 + 0] = output[0];
      edge_gradient[edge * 3 + 1] = output[1];
      edge_gradient[edge * 3 + 2] = output[2];
    }
  }
}

template <int Width>
__device__ __forceinline__ float block_sum_wide(float value, float* workspace) {
  static_assert(Width > kWarpSize && Width % kWarpSize == 0);
  constexpr int kWarps = Width / kWarpSize;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    workspace[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    const float partial = lane < kWarps ? workspace[lane] : 0.0f;
    const float total = warp_sum(partial);
    if (lane == 0) {
      workspace[0] = total;
    }
  }
  __syncthreads();
  return workspace[0];
}

template <int Width, bool Canonical, typename index_t>
__global__ __launch_bounds__(Width, 2) void dpa4c_wide_forward_kernel(
    long node_count,
    long edge_count,
    int interval_count,
    float table_stride,
    float table_max,
    float rcut,
    float eps,
    float degree_floor,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ table,
    const float* __restrict__ type_embedding,
    const float* __restrict__ feedback_weight,
    const float* __restrict__ scalar_weight,
    const float* __restrict__ vector_weight,
    const float* __restrict__ tensor_weight,
    float* __restrict__ descriptor,
    float* __restrict__ state) {
  static_assert(Width == 64 || Width == 128);
  const int channel = threadIdx.x;
  const int lane = channel & 31;
  const long node = blockIdx.x;
  if (node >= node_count) {
    return;
  }
  const int center_type = static_cast<int>(atype[node]);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];
  float first[kBasisDim] = {};
  float second[kBasisDim] = {};
  float degree = 0.0f;

  // === Pass 1. Accumulate one moment channel per thread ===
  for (long position = begin; position < end; ++position) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == 0) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<kWarpSize>(geometry, 0, kWarpMask);
    location = broadcast_location<kWarpSize>(location, 0, kWarpMask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      first[k] = fmaf(amplitude, geometry.basis[k], first[k]);
    }
    if (channel == 0) {
      degree = fmaf(geometry.envelope, geometry.envelope, degree);
    }
  }

  __shared__ float normalizer_shared;
  if (channel == 0) {
    normalizer_shared = rsqrtf(degree + degree_floor);
  }
  __syncthreads();
  const float normalizer = normalizer_shared;
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    first[k] *= normalizer;
  }

  // === Pass 2. Apply channel-wise feedback to every edge ===
  for (long position = begin; position < end; ++position) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == 0) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<kWarpSize>(geometry, 0, kWarpMask);
    location = broadcast_location<kWarpSize>(location, 0, kWarpMask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
    float feedback_input = 0.0f;
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      float projection = 0.0f;
      float basis_norm = 0.0f;
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        projection = fmaf(first[k], geometry.basis[k], projection);
        basis_norm = fmaf(geometry.basis[k], geometry.basis[k], basis_norm);
      }
      const float context = projection - normalizer * amplitude * basis_norm;
      feedback_input = fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
                            context, feedback_input);
    }
    const float modulated =
        amplitude * (1.0f + gate_activation(feedback_input));
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      second[k] = fmaf(modulated, geometry.basis[k], second[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    second[k] *= normalizer;
    state[(node * 19 + k) * Width + channel] = first[k];
    state[(node * 19 + 9 + k) * Width + channel] = second[k];
  }
  state[(node * 19 + 18) * Width + channel] = normalizer;

  __shared__ float second_shared[kBasisDim * Width];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    second_shared[k * Width + channel] = second[k];
  }
  __syncthreads();

  float scalar;
  float vector[3];
  Matrix3 tensor;
  float invariants[6];
  project_invariants_wide<Width>(
      channel, center_type, second_shared, type_embedding, scalar_weight,
      vector_weight, tensor_weight, scalar, vector, tensor, invariants);
#pragma unroll
  for (int family = 0; family < 6; ++family) {
    descriptor[(node * Width + channel) * 6 + family] = invariants[family];
  }
}

template <int Width, bool Canonical, typename index_t>
__global__ __launch_bounds__(Width, 2) void dpa4c_wide_backward_kernel(
    long node_count,
    long edge_count,
    int interval_count,
    float table_stride,
    float table_max,
    float rcut,
    float eps,
    float degree_floor,
    const float* __restrict__ descriptor_gradient,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ table,
    const float* __restrict__ type_embedding,
    const float* __restrict__ feedback_weight,
    const float* __restrict__ scalar_weight,
    const float* __restrict__ vector_weight,
    const float* __restrict__ tensor_weight,
    const float* __restrict__ state,
    float* __restrict__ edge_gradient) {
  static_assert(Width == 64 || Width == 128);
  const int channel = threadIdx.x;
  const int lane = channel & 31;
  const int channel_warp = channel >> 5;
  const long node = blockIdx.x;
  if (node >= node_count) {
    return;
  }
  const int center_type = static_cast<int>(atype[node]);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];

  // === Step 1. Load center state and reverse the invariant projection ===
  float first[kBasisDim];
  float second[kBasisDim];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    first[k] = __ldg(state + (node * 19 + k) * Width + channel);
    second[k] = __ldg(state + (node * 19 + 9 + k) * Width + channel);
  }
  const float normalizer = __ldg(state + (node * 19 + 18) * Width);

  __shared__ float workspace[kBasisDim * Width];
  __shared__ float reduction_workspace[Width / kWarpSize];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    workspace[k * Width + channel] = second[k];
  }
  __syncthreads();

  float scalar;
  float vector[3];
  Matrix3 tensor;
  float invariants[6];
  project_invariants_wide<Width>(
      channel, center_type, workspace, type_embedding, scalar_weight,
      vector_weight, tensor_weight, scalar, vector, tensor, invariants);
  float d_invariant[6];
#pragma unroll
  for (int family = 0; family < 6; ++family) {
    d_invariant[family] =
        __ldg(descriptor_gradient + (node * Width + channel) * 6 + family);
  }
  float d_scalar;
  float d_vector[3];
  float d_packed[5];
  differentiate_invariants(vector, tensor, d_invariant, d_scalar, d_vector,
                           d_packed);

  // All threads finish reading the moments before the workspace is reused for
  // the projected-output gradients.
  __syncthreads();
  workspace[channel] = d_scalar;
#pragma unroll
  for (int component = 0; component < 3; ++component) {
    workspace[(1 + component) * Width + channel] = d_vector[component];
  }
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    workspace[(4 + component) * Width + channel] = d_packed[component];
  }
  __syncthreads();

  float d_second[kBasisDim] = {};
  for (int output = 0; output < Width; ++output) {
    const long weight_offset = static_cast<long>(channel) * Width + output;
    d_second[0] = fmaf(workspace[output], __ldg(scalar_weight + weight_offset),
                       d_second[0]);
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      d_second[1 + component] =
          fmaf(workspace[(1 + component) * Width + output],
               __ldg(vector_weight + weight_offset), d_second[1 + component]);
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      d_second[4 + component] =
          fmaf(workspace[(4 + component) * Width + output],
               __ldg(tensor_weight + weight_offset), d_second[4 + component]);
    }
  }
  float d_sum_second[kBasisDim];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_sum_second[k] = normalizer * d_second[k];
  }

  // === Step 2. Accumulate feedback derivatives into the first moments ===
  float d_first[kBasisDim] = {};
  float d_normalizer_edges = 0.0f;
  for (long position = begin; position < end; ++position) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == 0) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<kWarpSize>(geometry, 0, kWarpMask);
    location = broadcast_location<kWarpSize>(location, 0, kWarpMask);
    const float amplitude = edge_amplitude<Width>(
        geometry, location, channel, center_type, table, type_embedding);
    float context[3] = {};
    float basis_norm[3] = {};
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        context[degree_index] =
            fmaf(first[k], geometry.basis[k], context[degree_index]);
        basis_norm[degree_index] = fmaf(geometry.basis[k], geometry.basis[k],
                                        basis_norm[degree_index]);
      }
      context[degree_index] -=
          normalizer * amplitude * basis_norm[degree_index];
    }
    float activation_input = 0.0f;
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      activation_input =
          fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
               context[degree_index], activation_input);
    }
    const float activation = gate_activation(activation_input);
    float d_modulated = 0.0f;
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_modulated = fmaf(d_sum_second[k], geometry.basis[k], d_modulated);
    }
    const float d_activation_input =
        d_modulated * amplitude * gate_derivative(activation_input);
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      const float d_context =
          d_activation_input *
          __ldg(feedback_weight + channel * 3 + degree_index);
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        d_first[k] = fmaf(d_context, geometry.basis[k], d_first[k]);
      }
      d_normalizer_edges -= d_context * amplitude * basis_norm[degree_index];
    }
  }

  float normalization_dot = 0.0f;
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    normalization_dot = fmaf(d_first[k], first[k], normalization_dot);
    normalization_dot = fmaf(d_second[k], second[k], normalization_dot);
  }
  float d_normalizer =
      d_normalizer_edges + __fdividef(normalization_dot, normalizer);
  d_normalizer = block_sum_wide<Width>(d_normalizer, reduction_workspace);
  const float d_degree =
      -0.5f * d_normalizer * normalizer * normalizer * normalizer;
  float d_sum_first[kBasisDim];
#pragma unroll
  for (int k = 0; k < kBasisDim; ++k) {
    d_sum_first[k] = normalizer * d_first[k];
  }

  // === Step 3. Recompute edges and reduce channel VJPs within each warp ===
  for (long position = begin; position < end; ++position) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (edge_mask != nullptr && !edge_mask[edge]) {
      continue;
    }
    EdgeGeometry geometry{};
    TableLocation location{};
    if (lane == 0) {
      geometry = load_geometry<index_t>(edge, edge_count, rcut, eps, edge_vec,
                                        edge_index, atype);
      location = locate_table(geometry.radius, table_stride, table_max,
                              interval_count);
    }
    geometry = broadcast_geometry<kWarpSize>(geometry, 0, kWarpMask);
    location = broadcast_location<kWarpSize>(location, 0, kWarpMask);
    const float2 radial =
        evaluate_table_with_derivative(table, location, channel, Width);
    const float type_value =
        __ldg(type_embedding + static_cast<long>(center_type) * Width +
              channel) +
        __ldg(type_embedding + static_cast<long>(geometry.source_type) * Width +
              channel);
    const float amplitude = radial.x + geometry.envelope * type_value;
    float context[3] = {};
    float basis_norm[3] = {};
    const int begins[3] = {0, 1, 4};
    const int ends[3] = {1, 4, 9};
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        context[degree_index] =
            fmaf(first[k], geometry.basis[k], context[degree_index]);
        basis_norm[degree_index] = fmaf(geometry.basis[k], geometry.basis[k],
                                        basis_norm[degree_index]);
      }
      context[degree_index] -=
          normalizer * amplitude * basis_norm[degree_index];
    }
    float activation_input = 0.0f;
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      activation_input =
          fmaf(__ldg(feedback_weight + channel * 3 + degree_index),
               context[degree_index], activation_input);
    }
    const float activation = gate_activation(activation_input);
    const float modulated = amplitude * (1.0f + activation);
    float d_modulated = 0.0f;
    float d_basis[kBasisDim] = {};
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_modulated = fmaf(d_sum_second[k], geometry.basis[k], d_modulated);
      d_basis[k] = d_sum_second[k] * modulated;
    }
    const float d_activation_input =
        d_modulated * amplitude * gate_derivative(activation_input);
    float d_amplitude = d_modulated * (1.0f + activation);
#pragma unroll
    for (int degree_index = 0; degree_index < 3; ++degree_index) {
      const float d_context =
          d_activation_input *
          __ldg(feedback_weight + channel * 3 + degree_index);
      d_amplitude -= d_context * normalizer * basis_norm[degree_index];
#pragma unroll
      for (int k = begins[degree_index]; k < ends[degree_index]; ++k) {
        d_basis[k] += d_context * (first[k] - 2.0f * normalizer * amplitude *
                                                  geometry.basis[k]);
      }
    }
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_amplitude = fmaf(d_sum_first[k], geometry.basis[k], d_amplitude);
      d_basis[k] = fmaf(d_sum_first[k], amplitude, d_basis[k]);
    }
    float radial_gradient = warp_sum(d_amplitude * radial.y);
    float envelope_gradient = warp_sum(d_amplitude * type_value);
#pragma unroll
    for (int k = 0; k < kBasisDim; ++k) {
      d_basis[k] = warp_sum(d_basis[k]);
    }
    if (lane == 0) {
      if (channel_warp == 0) {
        envelope_gradient += 2.0f * geometry.envelope * d_degree;
      }
      radial_gradient +=
          envelope_gradient * c3_envelope_derivative(geometry.radius, rcut);
      float output[3];
      basis_vjp(geometry, d_basis, radial_gradient, output);
      atomicAdd(edge_gradient + edge * 3 + 0, output[0]);
      atomicAdd(edge_gradient + edge * 3 + 1, output[1]);
      atomicAdd(edge_gradient + edge * 3 + 2, output[2]);
    }
  }
}

template <bool Canonical, typename index_t>
__global__ void zero_padding_kernel(long node_count,
                                    long edge_count,
                                    const index_t* destination_order,
                                    const long* destination_row_ptr,
                                    float* edge_gradient) {
  const long valid_edge_count = destination_row_ptr[node_count];
  for (long position = valid_edge_count + blockIdx.x * blockDim.x + threadIdx.x;
       position < edge_count;
       position += static_cast<long>(blockDim.x) * gridDim.x) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    edge_gradient[edge * 3 + 0] = 0.0f;
    edge_gradient[edge * 3 + 1] = 0.0f;
    edge_gradient[edge * 3 + 2] = 0.0f;
  }
}

template <int Width, bool Canonical, typename index_t>
void launch_forward(long node_count,
                    long edge_count,
                    int interval_count,
                    float table_stride,
                    float table_max,
                    float rcut,
                    float eps,
                    float degree_floor,
                    const torch::Tensor& edge_vec,
                    const torch::Tensor& edge_index,
                    const torch::Tensor& edge_mask,
                    const torch::Tensor& destination_order,
                    const torch::Tensor& destination_row_ptr,
                    const torch::Tensor& atype,
                    const torch::Tensor& table,
                    const torch::Tensor& type_embedding,
                    const torch::Tensor& feedback_weight,
                    const torch::Tensor& scalar_weight,
                    const torch::Tensor& vector_weight,
                    const torch::Tensor& tensor_weight,
                    torch::Tensor& descriptor,
                    torch::Tensor& state,
                    cudaStream_t stream) {
  if constexpr (Width <= kWarpSize) {
    const int warps_per_block = kThreads / kWarpSize;
    const int blocks =
        static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
    dpa4c_forward_kernel<Width, Canonical, index_t>
        <<<blocks, kThreads, 0, stream>>>(
            node_count, edge_count, interval_count, table_stride, table_max,
            rcut, eps, degree_floor, edge_vec.data_ptr<float>(),
            edge_index.data_ptr<index_t>(),
            edge_mask.numel() ? edge_mask.data_ptr<bool>() : nullptr,
            destination_order.numel() ? destination_order.data_ptr<index_t>()
                                      : nullptr,
            destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
            table.data_ptr<float>(), type_embedding.data_ptr<float>(),
            feedback_weight.data_ptr<float>(), scalar_weight.data_ptr<float>(),
            vector_weight.data_ptr<float>(), tensor_weight.data_ptr<float>(),
            descriptor.data_ptr<float>(), state.data_ptr<float>());
  } else {
    dpa4c_wide_forward_kernel<Width, Canonical, index_t>
        <<<static_cast<int>(node_count), Width, 0, stream>>>(
            node_count, edge_count, interval_count, table_stride, table_max,
            rcut, eps, degree_floor, edge_vec.data_ptr<float>(),
            edge_index.data_ptr<index_t>(),
            edge_mask.numel() ? edge_mask.data_ptr<bool>() : nullptr,
            destination_order.numel() ? destination_order.data_ptr<index_t>()
                                      : nullptr,
            destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
            table.data_ptr<float>(), type_embedding.data_ptr<float>(),
            feedback_weight.data_ptr<float>(), scalar_weight.data_ptr<float>(),
            vector_weight.data_ptr<float>(), tensor_weight.data_ptr<float>(),
            descriptor.data_ptr<float>(), state.data_ptr<float>());
  }
  DPA4C_CHECK_LAUNCH("dpa4c_graph_compress forward");
}

template <int Width, bool Canonical, typename index_t>
void launch_backward(long node_count,
                     long edge_count,
                     int interval_count,
                     float table_stride,
                     float table_max,
                     float rcut,
                     float eps,
                     float degree_floor,
                     const torch::Tensor& descriptor_gradient,
                     const torch::Tensor& edge_vec,
                     const torch::Tensor& edge_index,
                     const torch::Tensor& edge_mask,
                     const torch::Tensor& destination_order,
                     const torch::Tensor& destination_row_ptr,
                     const torch::Tensor& atype,
                     const torch::Tensor& table,
                     const torch::Tensor& type_embedding,
                     const torch::Tensor& feedback_weight,
                     const torch::Tensor& scalar_weight,
                     const torch::Tensor& vector_weight,
                     const torch::Tensor& tensor_weight,
                     const torch::Tensor& state,
                     torch::Tensor& edge_gradient,
                     cudaStream_t stream) {
  if constexpr (Width <= kWarpSize) {
    const int warps_per_block = kThreads / kWarpSize;
    const int blocks =
        static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
    dpa4c_backward_kernel<Width, Canonical, index_t>
        <<<blocks, kThreads, 0, stream>>>(
            node_count, edge_count, interval_count, table_stride, table_max,
            rcut, eps, degree_floor, descriptor_gradient.data_ptr<float>(),
            edge_vec.data_ptr<float>(), edge_index.data_ptr<index_t>(),
            edge_mask.numel() ? edge_mask.data_ptr<bool>() : nullptr,
            destination_order.numel() ? destination_order.data_ptr<index_t>()
                                      : nullptr,
            destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
            table.data_ptr<float>(), type_embedding.data_ptr<float>(),
            feedback_weight.data_ptr<float>(), scalar_weight.data_ptr<float>(),
            vector_weight.data_ptr<float>(), tensor_weight.data_ptr<float>(),
            state.data_ptr<float>(), edge_gradient.data_ptr<float>());
  } else {
    const cudaError_t memset_error =
        cudaMemsetAsync(edge_gradient.data_ptr<float>(), 0,
                        edge_gradient.numel() * sizeof(float), stream);
    TORCH_CHECK(memset_error == cudaSuccess,
                "dpa4c_graph_compress backward initialization: ",
                cudaGetErrorString(memset_error));
    dpa4c_wide_backward_kernel<Width, Canonical, index_t>
        <<<static_cast<int>(node_count), Width, 0, stream>>>(
            node_count, edge_count, interval_count, table_stride, table_max,
            rcut, eps, degree_floor, descriptor_gradient.data_ptr<float>(),
            edge_vec.data_ptr<float>(), edge_index.data_ptr<index_t>(),
            edge_mask.numel() ? edge_mask.data_ptr<bool>() : nullptr,
            destination_order.numel() ? destination_order.data_ptr<index_t>()
                                      : nullptr,
            destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
            table.data_ptr<float>(), type_embedding.data_ptr<float>(),
            feedback_weight.data_ptr<float>(), scalar_weight.data_ptr<float>(),
            vector_weight.data_ptr<float>(), tensor_weight.data_ptr<float>(),
            state.data_ptr<float>(), edge_gradient.data_ptr<float>());
  }
  DPA4C_CHECK_LAUNCH("dpa4c_graph_compress backward");
  if constexpr (Width <= kWarpSize) {
    zero_padding_kernel<Canonical, index_t><<<1, kThreads, 0, stream>>>(
        node_count, edge_count,
        destination_order.numel() ? destination_order.data_ptr<index_t>()
                                  : nullptr,
        destination_row_ptr.data_ptr<long>(), edge_gradient.data_ptr<float>());
    DPA4C_CHECK_LAUNCH("dpa4c_graph_compress padding");
  }
}

void validate_inputs(const torch::Tensor& edge_vec,
                     const torch::Tensor& edge_index,
                     const torch::Tensor& edge_mask,
                     const torch::Tensor& destination_order,
                     const torch::Tensor& destination_row_ptr,
                     const torch::Tensor& atype,
                     const torch::Tensor& table,
                     const torch::Tensor& type_embedding,
                     const torch::Tensor& feedback_weight,
                     const torch::Tensor& scalar_weight,
                     const torch::Tensor& vector_weight,
                     const torch::Tensor& tensor_weight) {
  TORCH_CHECK(edge_vec.is_cuda() && edge_index.is_cuda() &&
                  edge_mask.is_cuda() && destination_order.is_cuda() &&
                  destination_row_ptr.is_cuda() && atype.is_cuda() &&
                  table.is_cuda() && type_embedding.is_cuda() &&
                  feedback_weight.is_cuda() && scalar_weight.is_cuda() &&
                  vector_weight.is_cuda() && tensor_weight.is_cuda(),
              "dpa4c_graph_compress: all tensor inputs must be CUDA tensors");
  TORCH_CHECK(
      edge_vec.is_contiguous() && edge_index.is_contiguous() &&
          edge_mask.is_contiguous() && destination_order.is_contiguous() &&
          destination_row_ptr.is_contiguous() && atype.is_contiguous() &&
          table.is_contiguous() && type_embedding.is_contiguous() &&
          feedback_weight.is_contiguous() && scalar_weight.is_contiguous() &&
          vector_weight.is_contiguous() && tensor_weight.is_contiguous(),
      "dpa4c_graph_compress: all tensor inputs must be contiguous");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt32 ||
                  edge_index.scalar_type() == torch::kInt64,
              "dpa4c_graph_compress: edge indices must be int32 or int64");
  TORCH_CHECK(destination_order.scalar_type() == edge_index.scalar_type(),
              "dpa4c_graph_compress: destination_order dtype must match "
              "edge_index");
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool,
              "dpa4c_graph_compress: edge_mask must be bool");
  TORCH_CHECK(
      destination_row_ptr.scalar_type() == torch::kInt64 &&
          atype.scalar_type() == torch::kInt64,
      "dpa4c_graph_compress: row pointers and atom types must be int64");
  for (const torch::Tensor* tensor :
       {&table, &type_embedding, &feedback_weight, &scalar_weight,
        &vector_weight, &tensor_weight}) {
    TORCH_CHECK(tensor->scalar_type() == torch::kFloat32,
                "dpa4c_graph_compress: tables and weights must be fp32");
  }
}

template <typename index_t>
void dispatch_forward(int width,
                      bool canonical,
                      long node_count,
                      long edge_count,
                      int interval_count,
                      float table_stride,
                      float table_max,
                      float rcut,
                      float eps,
                      float degree_floor,
                      const torch::Tensor& edge_vec,
                      const torch::Tensor& edge_index,
                      const torch::Tensor& edge_mask,
                      const torch::Tensor& destination_order,
                      const torch::Tensor& destination_row_ptr,
                      const torch::Tensor& atype,
                      const torch::Tensor& table,
                      const torch::Tensor& type_embedding,
                      const torch::Tensor& feedback_weight,
                      const torch::Tensor& scalar_weight,
                      const torch::Tensor& vector_weight,
                      const torch::Tensor& tensor_weight,
                      torch::Tensor& descriptor,
                      torch::Tensor& state,
                      cudaStream_t stream) {
#define DISPATCH_WIDTH(value)                                              \
  if (width == value) {                                                    \
    if (canonical) {                                                       \
      launch_forward<value, true, index_t>(                                \
          node_count, edge_count, interval_count, table_stride, table_max, \
          rcut, eps, degree_floor, edge_vec, edge_index, edge_mask,        \
          destination_order, destination_row_ptr, atype, table,            \
          type_embedding, feedback_weight, scalar_weight, vector_weight,   \
          tensor_weight, descriptor, state, stream);                       \
    } else {                                                               \
      launch_forward<value, false, index_t>(                               \
          node_count, edge_count, interval_count, table_stride, table_max, \
          rcut, eps, degree_floor, edge_vec, edge_index, edge_mask,        \
          destination_order, destination_row_ptr, atype, table,            \
          type_embedding, feedback_weight, scalar_weight, vector_weight,   \
          tensor_weight, descriptor, state, stream);                       \
    }                                                                      \
    return;                                                                \
  }
  DISPATCH_WIDTH(4)
  DISPATCH_WIDTH(8)
  DISPATCH_WIDTH(16)
  DISPATCH_WIDTH(32)
  DISPATCH_WIDTH(64)
  DISPATCH_WIDTH(128)
#undef DISPATCH_WIDTH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channel width ", width);
}

template <typename index_t>
void dispatch_backward(int width,
                       bool canonical,
                       long node_count,
                       long edge_count,
                       int interval_count,
                       float table_stride,
                       float table_max,
                       float rcut,
                       float eps,
                       float degree_floor,
                       const torch::Tensor& descriptor_gradient,
                       const torch::Tensor& edge_vec,
                       const torch::Tensor& edge_index,
                       const torch::Tensor& edge_mask,
                       const torch::Tensor& destination_order,
                       const torch::Tensor& destination_row_ptr,
                       const torch::Tensor& atype,
                       const torch::Tensor& table,
                       const torch::Tensor& type_embedding,
                       const torch::Tensor& feedback_weight,
                       const torch::Tensor& scalar_weight,
                       const torch::Tensor& vector_weight,
                       const torch::Tensor& tensor_weight,
                       const torch::Tensor& state,
                       torch::Tensor& edge_gradient,
                       cudaStream_t stream) {
#define DISPATCH_WIDTH(value)                                                 \
  if (width == value) {                                                       \
    if (canonical) {                                                          \
      launch_backward<value, true, index_t>(                                  \
          node_count, edge_count, interval_count, table_stride, table_max,    \
          rcut, eps, degree_floor, descriptor_gradient, edge_vec, edge_index, \
          edge_mask, destination_order, destination_row_ptr, atype, table,    \
          type_embedding, feedback_weight, scalar_weight, vector_weight,      \
          tensor_weight, state, edge_gradient, stream);                       \
    } else {                                                                  \
      launch_backward<value, false, index_t>(                                 \
          node_count, edge_count, interval_count, table_stride, table_max,    \
          rcut, eps, degree_floor, descriptor_gradient, edge_vec, edge_index, \
          edge_mask, destination_order, destination_row_ptr, atype, table,    \
          type_embedding, feedback_weight, scalar_weight, vector_weight,      \
          tensor_weight, state, edge_gradient, stream);                       \
    }                                                                         \
    return;                                                                   \
  }
  DISPATCH_WIDTH(4)
  DISPATCH_WIDTH(8)
  DISPATCH_WIDTH(16)
  DISPATCH_WIDTH(32)
  DISPATCH_WIDTH(64)
  DISPATCH_WIDTH(128)
#undef DISPATCH_WIDTH
  TORCH_CHECK(false,
              "dpa4c_graph_compress_backward: unsupported channel width ",
              width);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dpa4c_graph_compress(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor type_embedding,
    torch::Tensor feedback_weight,
    torch::Tensor scalar_weight,
    torch::Tensor vector_weight,
    torch::Tensor tensor_weight,
    bool canonical,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, table, type_embedding,
                  feedback_weight, scalar_weight, vector_weight, tensor_weight);
  const long node_count = atype.size(0);
  const long edge_count = edge_vec.size(0);
  const int width = static_cast<int>(type_embedding.size(1));
  TORCH_CHECK(table.size(1) == 6 * width,
              "dpa4c_graph_compress: table width must equal 6 * channels");
  TORCH_CHECK(feedback_weight.sizes() == torch::IntArrayRef({width, 3}),
              "dpa4c_graph_compress: feedback weight must have shape "
              "(channels, 3)");
  TORCH_CHECK(scalar_weight.sizes() == torch::IntArrayRef({width, width}) &&
                  vector_weight.sizes() == torch::IntArrayRef({width, width}) &&
                  tensor_weight.sizes() == torch::IntArrayRef({width, width}),
              "dpa4c_graph_compress: invalid projection shape");
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1,
              "dpa4c_graph_compress: destination_row_ptr must have N + 1 "
              "entries");
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor = torch::empty({node_count, 6 * width}, options);
  auto state = torch::empty({node_count, 19, width}, options);
  if (node_count == 0) {
    return {descriptor, state};
  }
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int interval_count = static_cast<int>(table.size(0));
  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    dispatch_forward<index_t>(
        width, canonical, node_count, edge_count, interval_count,
        static_cast<float>(table_stride), static_cast<float>(table_max),
        static_cast<float>(rcut), static_cast<float>(eps),
        static_cast<float>(degree_floor), edge_vec_float, edge_index, edge_mask,
        destination_order, destination_row_ptr, atype, table, type_embedding,
        feedback_weight, scalar_weight, vector_weight, tensor_weight,
        descriptor, state, stream);
  };
  if (edge_index.scalar_type() == torch::kInt32) {
    launch(int{});
  } else {
    launch(long{});
  }
  return {descriptor, state};
}

torch::Tensor dpa4c_graph_compress_backward(torch::Tensor descriptor_gradient,
                                            torch::Tensor state,
                                            torch::Tensor edge_vec,
                                            torch::Tensor edge_index,
                                            torch::Tensor edge_mask,
                                            torch::Tensor destination_order,
                                            torch::Tensor destination_row_ptr,
                                            torch::Tensor atype,
                                            torch::Tensor table,
                                            torch::Tensor type_embedding,
                                            torch::Tensor feedback_weight,
                                            torch::Tensor scalar_weight,
                                            torch::Tensor vector_weight,
                                            torch::Tensor tensor_weight,
                                            bool canonical,
                                            double table_stride,
                                            double table_max,
                                            double rcut,
                                            double eps,
                                            double degree_floor) {
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, table, type_embedding,
                  feedback_weight, scalar_weight, vector_weight, tensor_weight);
  const long node_count = atype.size(0);
  const long edge_count = edge_vec.size(0);
  const int width = static_cast<int>(type_embedding.size(1));
  TORCH_CHECK(state.is_cuda() && state.is_contiguous() &&
                  state.scalar_type() == torch::kFloat32 &&
                  state.sizes() == torch::IntArrayRef({node_count, 19, width}),
              "dpa4c_graph_compress_backward: invalid saved state");
  if (node_count == 0) {
    return torch::zeros_like(edge_vec);
  }
  auto descriptor_gradient_float =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto edge_gradient = torch::empty_like(edge_vec_float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int interval_count = static_cast<int>(table.size(0));
  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    dispatch_backward<index_t>(
        width, canonical, node_count, edge_count, interval_count,
        static_cast<float>(table_stride), static_cast<float>(table_max),
        static_cast<float>(rcut), static_cast<float>(eps),
        static_cast<float>(degree_floor), descriptor_gradient_float,
        edge_vec_float, edge_index, edge_mask, destination_order,
        destination_row_ptr, atype, table, type_embedding, feedback_weight,
        scalar_weight, vector_weight, tensor_weight, state, edge_gradient,
        stream);
  };
  if (edge_index.scalar_type() == torch::kInt32) {
    launch(int{});
  } else {
    launch(long{});
  }
  return edge_gradient.to(edge_vec.scalar_type());
}

std::tuple<torch::Tensor, torch::Tensor> dpa4c_canonical_compress(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor type_embedding,
    torch::Tensor feedback_weight,
    torch::Tensor scalar_weight,
    torch::Tensor vector_weight,
    torch::Tensor tensor_weight,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress: source and edge_vec must share the "
              "edge axis");
  TORCH_CHECK(destination_row_ptr.numel() == atype.size(0) + 1,
              "dpa4c_canonical_compress: destination_row_ptr must have N + 1 "
              "entries");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress(edge_vec, source, edge_mask, destination_order,
                              destination_row_ptr, atype, table, type_embedding,
                              feedback_weight, scalar_weight, vector_weight,
                              tensor_weight, true, table_stride, table_max,
                              rcut, eps, degree_floor);
}

torch::Tensor dpa4c_canonical_compress_backward(
    torch::Tensor descriptor_gradient,
    torch::Tensor state,
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor type_embedding,
    torch::Tensor feedback_weight,
    torch::Tensor scalar_weight,
    torch::Tensor vector_weight,
    torch::Tensor tensor_weight,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_backward: source and edge_vec must "
              "share the edge axis");
  TORCH_CHECK(destination_row_ptr.numel() == atype.size(0) + 1,
              "dpa4c_canonical_compress_backward: destination_row_ptr must "
              "have N + 1 entries");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress_backward(
      descriptor_gradient, state, edge_vec, source, edge_mask,
      destination_order, destination_row_ptr, atype, table, type_embedding,
      feedback_weight, scalar_weight, vector_weight, tensor_weight, true,
      table_stride, table_max, rcut, eps, degree_floor);
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa4c_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor type_embedding, Tensor feedback_weight, "
      "Tensor scalar_weight, Tensor vector_weight, Tensor tensor_weight, "
      "bool canonical, float table_stride, "
      "float table_max, float rcut, float eps, float degree_floor) "
      "-> (Tensor descriptor, Tensor state)");
  library.impl("dpa4c_graph_compress", torch::kCUDA, &dpa4c_graph_compress);
  library.def(
      "dpa4c_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, "
      "Tensor destination_order, Tensor destination_row_ptr, Tensor atype, "
      "Tensor table, Tensor type_embedding, Tensor feedback_weight, "
      "Tensor scalar_weight, Tensor vector_weight, Tensor tensor_weight, "
      "bool canonical, float table_stride, "
      "float table_max, float rcut, float eps, float degree_floor) -> Tensor");
  library.impl("dpa4c_graph_compress_backward", torch::kCUDA,
               &dpa4c_graph_compress_backward);
  library.def(
      "dpa4c_canonical_compress(Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor type_embedding, Tensor feedback_weight, "
      "Tensor scalar_weight, Tensor vector_weight, Tensor tensor_weight, "
      "float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) -> (Tensor descriptor, Tensor state)");
  library.impl("dpa4c_canonical_compress", torch::kCUDA,
               &dpa4c_canonical_compress);
  library.def(
      "dpa4c_canonical_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor type_embedding, Tensor feedback_weight, "
      "Tensor scalar_weight, Tensor vector_weight, Tensor tensor_weight, "
      "float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) -> Tensor");
  library.impl("dpa4c_canonical_compress_backward", torch::kCUDA,
               &dpa4c_canonical_compress_backward);
}
