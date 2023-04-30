// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/ops/arm/base/group_conv2d_3x3.h"

#include <arm_neon.h>

#include <memory>

#include "mace/ops/delegator/group_conv2d.h"

namespace mace {
namespace ops {
namespace arm {

template <>
MaceStatus GroupConv2dK3x3S1<float>::DoCompute(const GroupConvComputeParam &p,
                                               const float *filter_data,
                                               const float *input_data,
                                               float *output_data) {
  p.thread_pool.Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t g = 0; g < p.group_size; g++) {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t m = start1 + g * p.group_size;
                 m < std::min(end1, (g + 1) * p.group_size); m += step1) {
              if (m + 1 < (g + 1) * p.group_size && m + 1 < p.out_channels) {
                float *out_ptr0_base =
                    output_data + b * p.out_batch_size + m * p.out_image_size;
                float *out_ptr1_base = output_data + b * p.out_batch_size +
                                       (m + 1) * p.out_image_size;
                for (index_t c = g * p.in_channels / p.group_size;
                     c < (g + 1) * p.in_channels / p.group_size; ++c) {
                  const float *in_ptr0 =
                      input_data + b * p.in_batch_size + c * p.in_image_size;
                  const float *filter_ptr0 = filter_data +
                                             (m - g * p.group_size) *
                                                 p.in_channels / p.group_size *
                                                 9 +
                                             c * 9;

                  float *out_ptr1 = out_ptr1_base;
                  const float *in_ptr1 = input_data + b * p.in_batch_size +
                                         c * p.in_image_size + 1 * p.in_width;
                  const float *in_ptr2 = input_data + b * p.in_batch_size +
                                         c * p.in_image_size + 2 * p.in_width;
                  const float *in_ptr3 = input_data + b * p.in_batch_size +
                                         c * p.in_image_size + 3 * p.in_width;
                  const float *filter_ptr1 = filter_data +
                                             (m - g * p.group_size + 1) *
                                                 p.in_channels / p.group_size *
                                                 9 +
                                             c * 9;

                  float *out_ptr0 = out_ptr0_base;


                  // load filter (2 outch x 3 height x 3 width): vf_outch_height
                  float32x4_t vf00, vf01, vf02;
                  float32x4_t vf10, vf11, vf12;
                  vf00 = vld1q_f32(filter_ptr0);
                  vf01 = vld1q_f32(filter_ptr0 + 3);
                  vf02 = vld1q_f32(filter_ptr0 + 6);

                  vf10 = vld1q_f32(filter_ptr1);
                  vf11 = vld1q_f32(filter_ptr1 + 3);
                  vf12 = vld1q_f32(filter_ptr1 + 6);

                  for (index_t h = 0; h + 1 < p.out_height; h += 2) {
                    for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                      // input (4 height x 3 slide): vi_height_slide
                      float32x4_t vi00, vi01, vi02;  // reg count: 14
                      float32x4_t vi10, vi11, vi12;
                      float32x4_t vi20, vi21, vi22;
                      float32x4_t vi30, vi31, vi32;
                      float32x4_t vo20, vo30;  // tmp use

                      // output (4 outch x 2 height x 4 width): vo_outch_height
                      float32x4_t vo00, vo01;
                      float32x4_t vo10, vo11;

                      // load input
                      vi00 = vld1q_f32(in_ptr0);
                      vo00 = vld1q_f32(in_ptr0 + 4);  // reuse vo00: vi0n
                      vi10 = vld1q_f32(in_ptr1);
                      vo10 = vld1q_f32(in_ptr1 + 4);
                      vi20 = vld1q_f32(in_ptr2);
                      vo20 = vld1q_f32(in_ptr2 + 4);
                      vi30 = vld1q_f32(in_ptr3);
                      vo30 = vld1q_f32(in_ptr3 + 4);

                      vi01 = vextq_f32(vi00, vo00, 1);
                      vi02 = vextq_f32(vi00, vo00, 2);
                      vi11 = vextq_f32(vi10, vo10, 1);
                      vi12 = vextq_f32(vi10, vo10, 2);
                      vi21 = vextq_f32(vi20, vo20, 1);
                      vi22 = vextq_f32(vi20, vo20, 2);
                      vi31 = vextq_f32(vi30, vo30, 1);
                      vi32 = vextq_f32(vi30, vo30, 2);

                      // load ouptut
                      vo00 = vld1q_f32(out_ptr0);
                      vo01 = vld1q_f32(out_ptr0 + p.out_width);
                      vo10 = vld1q_f32(out_ptr1);
                      vo11 = vld1q_f32(out_ptr1 + p.out_width);

                      // outch 0, height 0
                      vo00 = vfmaq_laneq_f32(vo00, vi00, vf00,
                                             0);  // reg count: 18
                      vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                      vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                      vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                      vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                      vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                      vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 0);
                      vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 1);
                      vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 2);

                      // outch 0, height 1
                      vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                      vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                      vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                      vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                      vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                      vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                      vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 0);
                      vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 1);
                      vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 2);

                      // outch 1, height 0
                      vo10 = vfmaq_laneq_f32(vo10, vi00, vf10, 0);
                      vo10 = vfmaq_laneq_f32(vo10, vi01, vf10, 1);
                      vo10 = vfmaq_laneq_f32(vo10, vi02, vf10, 2);
                      vo10 = vfmaq_laneq_f32(vo10, vi10, vf11, 0);
                      vo10 = vfmaq_laneq_f32(vo10, vi11, vf11, 1);
                      vo10 = vfmaq_laneq_f32(vo10, vi12, vf11, 2);
                      vo10 = vfmaq_laneq_f32(vo10, vi20, vf12, 0);
                      vo10 = vfmaq_laneq_f32(vo10, vi21, vf12, 1);
                      vo10 = vfmaq_laneq_f32(vo10, vi22, vf12, 2);

                      // outch 1, height 1
                      vo11 = vfmaq_laneq_f32(vo11, vi10, vf10, 0);
                      vo11 = vfmaq_laneq_f32(vo11, vi11, vf10, 1);
                      vo11 = vfmaq_laneq_f32(vo11, vi12, vf10, 2);
                      vo11 = vfmaq_laneq_f32(vo11, vi20, vf11, 0);
                      vo11 = vfmaq_laneq_f32(vo11, vi21, vf11, 1);
                      vo11 = vfmaq_laneq_f32(vo11, vi22, vf11, 2);
                      vo11 = vfmaq_laneq_f32(vo11, vi30, vf12, 0);
                      vo11 = vfmaq_laneq_f32(vo11, vi31, vf12, 1);
                      vo11 = vfmaq_laneq_f32(vo11, vi32, vf12, 2);

                      vst1q_f32(out_ptr0, vo00);
                      vst1q_f32(out_ptr0 + p.out_width, vo01);
                      vst1q_f32(out_ptr1, vo10);
                      vst1q_f32(out_ptr1 + p.out_width, vo11);

                      in_ptr0 += 4;
                      in_ptr1 += 4;
                      in_ptr2 += 4;
                      in_ptr3 += 4;

                      out_ptr0 += 4;
                      out_ptr1 += 4;
                    }  // w

                    in_ptr0 += 2 + p.in_width;
                    in_ptr1 += 2 + p.in_width;
                    in_ptr2 += 2 + p.in_width;
                    in_ptr3 += 2 + p.in_width;

                    out_ptr0 += p.out_width;
                    out_ptr1 += p.out_width;
                  }  // h
                }    // c
              } else {
                for (index_t mm = g * p.out_channels / p.group_size;
                     mm < (g + 1) * p.out_channels / p.group_size; ++mm) {
                  float *out_ptr0_base = output_data + b * p.out_batch_size +
                                         mm * p.out_image_size;

                  for (index_t c = g * p.in_channels / p.group_size;
                       c < (g + 1) * p.in_channels / p.group_size; ++c) {
                    const float *in_ptr0 =
                        input_data + b * p.in_batch_size + c * p.in_image_size;
                    const float *in_ptr1 = input_data + b * p.in_batch_size +
                                           c * p.in_image_size + 1 * p.in_width;
                    const float *in_ptr2 = input_data + b * p.in_batch_size +
                                           c * p.in_image_size + 2 * p.in_width;
                    const float *in_ptr3 = input_data + b * p.in_batch_size +
                                           c * p.in_image_size + 3 * p.in_width;
                    const float *filter_ptr0 =
                        filter_data + mm * (p.in_channels / p.group_size) * 9 +
                        c * 9;

                    float *out_ptr0 = out_ptr0_base;

                    // load filter (1 outch x 3 height x 3 width):
                    // vf_outch_height
                    float32x4_t vf00, vf01, vf02;
                    vf00 = vld1q_f32(filter_ptr0);
                    vf01 = vld1q_f32(filter_ptr0 + 3);
                    vf02 = vld1q_f32(filter_ptr0 + 5);

                    for (index_t h = 0; h + 1 < p.out_height; h += 2) {
                      for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                        // input (4 height x 3 slide): vi_height_slide
                        float32x4_t vi00, vi01, vi02, vi0n;
                        float32x4_t vi10, vi11, vi12, vi1n;
                        float32x4_t vi20, vi21, vi22, vi2n;
                        float32x4_t vi30, vi31, vi32, vi3n;

                        // output (1 outch x 2 height x 4 width):
                        // vo_outch_height
                        float32x4_t vo00, vo01;

                        // load input
                        vi00 = vld1q_f32(in_ptr0);
                        vi0n = vld1q_f32(in_ptr0 + 4);
                        vi10 = vld1q_f32(in_ptr1);
                        vi1n = vld1q_f32(in_ptr1 + 4);
                        vi20 = vld1q_f32(in_ptr2);
                        vi2n = vld1q_f32(in_ptr2 + 4);
                        vi30 = vld1q_f32(in_ptr3);
                        vi3n = vld1q_f32(in_ptr3 + 4);

                        vi01 = vextq_f32(vi00, vi0n, 1);
                        vi02 = vextq_f32(vi00, vi0n, 2);
                        vi11 = vextq_f32(vi10, vi1n, 1);
                        vi12 = vextq_f32(vi10, vi1n, 2);
                        vi21 = vextq_f32(vi20, vi2n, 1);
                        vi22 = vextq_f32(vi20, vi2n, 2);
                        vi31 = vextq_f32(vi30, vi3n, 1);
                        vi32 = vextq_f32(vi30, vi3n, 2);

                        // load ouptut
                        vo00 = vld1q_f32(out_ptr0);
                        vo01 = vld1q_f32(out_ptr0 + p.out_width);

                        // outch 0, height 0
                        vo00 = vfmaq_laneq_f32(vo00, vi00, vf00, 0);
                        vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                        vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                        vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                        vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                        vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                        vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 1);
                        vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 2);
                        vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 3);

                        // outch 0, height 1
                        vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                        vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                        vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                        vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                        vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                        vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                        vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 1);
                        vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 2);
                        vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 3);

                        vst1q_f32(out_ptr0, vo00);
                        vst1q_f32(out_ptr0 + p.out_width, vo01);

                        in_ptr0 += 4;
                        in_ptr1 += 4;
                        in_ptr2 += 4;
                        in_ptr3 += 4;

                        out_ptr0 += 4;
                      }  // w

                      in_ptr0 += 2 + p.in_width;
                      in_ptr1 += 2 + p.in_width;
                      in_ptr2 += 2 + p.in_width;
                      in_ptr3 += 2 + p.in_width;

                      out_ptr0 += p.out_width;
                    }  // h
                  }    // c
                }      // mm
              }        // if
            }          // m
          }            // b
        }              // g
      },
      0, p.batch, 1, 0, p.out_channels, 2);

  return MaceStatus::MACE_SUCCESS;
}

template <>
MaceStatus GroupConv2dK3x3S2<float>::DoCompute(const GroupConvComputeParam &p,
                                               const float *filter_data,
                                               const float *input_data,
                                               float *output_data) {
  p.thread_pool.Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t g = start1; g < end1; g += step1) {
            for (index_t m = g * p.out_channels / p.group_size;
                 m < (g + 1) * p.out_channels / p.group_size; ++m) {
              for (index_t c = g * p.in_channels / p.group_size;
                   c < (g + 1) * p.in_channels / p.group_size; ++c) {
                const float *in_base =
                    input_data + b * p.in_batch_size + c * p.in_image_size;
                const float *filter_ptr =
                    filter_data + m * p.in_channels * 9 + c * 9;
                float *out_base =
                    output_data + b * p.out_batch_size + m * p.out_image_size;

                // load filter (1 outch x 3 height x 3 width):
                // vf_outch_height
                float32x4_t vf00, vf01, vf02;
                vf00 = vld1q_f32(filter_ptr);
                vf01 = vld1q_f32(filter_ptr + 3);
                vf02 = vld1q_f32(filter_ptr + 5);

                for (index_t h = 0; h < p.out_height; ++h) {
                  for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                    float32x4x2_t vi0, vi1, vi2;
                    float32x4_t vi0n, vi1n, vi2n;

                    // input (3 height x 3 slide): vi_height_slide
                    float32x4_t vi00, vi01, vi02;
                    float32x4_t vi10, vi11, vi12;
                    float32x4_t vi20, vi21, vi22;

                    // output (1 outch x 1 height x 4 width): vo
                    float32x4_t vo;

                    // load input
                    index_t in_h = h * 2;
                    index_t in_w = w * 2;
                    index_t in_offset = in_h * p.in_width + in_w;
                    vi0 =
                        vld2q_f32(in_base + in_offset);  // [ 0.2.4.6, 1.3.5.7 ]
                    vi1 = vld2q_f32(in_base + in_offset + p.in_width);
                    vi2 = vld2q_f32(in_base + in_offset + 2 * p.in_width);

                    vi0n = vld1q_f32(in_base + in_offset + 8);  // [8.9.10.11]
                    vi1n = vld1q_f32(in_base + in_offset + p.in_width + 8);
                    vi2n = (float32x4_t){
                        (in_base + in_offset + 2 * p.in_width + 8)[0], 0.0, 0.0,
                        0.0};

                    // load ouptut
                    index_t out_offset = h * p.out_width + w;
                    vo = vld1q_f32(out_base + out_offset);

                    vi00 = vi0.val[0];                // [0.2.4.6]
                    vi01 = vi0.val[1];                // [1.3.5.7]
                    vi02 = vextq_f32(vi00, vi0n, 1);  // [2.4.6.8]
                    vi10 = vi1.val[0];
                    vi11 = vi1.val[1];
                    vi12 = vextq_f32(vi10, vi1n, 1);
                    vi20 = vi2.val[0];
                    vi21 = vi2.val[1];
                    vi22 = vextq_f32(vi20, vi2n, 1);

                    // outch 0, height 0
                    vo = vfmaq_laneq_f32(vo, vi00, vf00, 0);
                    vo = vfmaq_laneq_f32(vo, vi01, vf00, 1);
                    vo = vfmaq_laneq_f32(vo, vi02, vf00, 2);
                    vo = vfmaq_laneq_f32(vo, vi10, vf01, 0);
                    vo = vfmaq_laneq_f32(vo, vi11, vf01, 1);
                    vo = vfmaq_laneq_f32(vo, vi12, vf01, 2);
                    vo = vfmaq_laneq_f32(vo, vi20, vf02, 1);
                    vo = vfmaq_laneq_f32(vo, vi21, vf02, 2);
                    vo = vfmaq_laneq_f32(vo, vi22, vf02, 3);

                    vst1q_f32(out_base + out_offset, vo);
                  }  // w
                }    // h
              }      // c
            }        // m
          }          // g
        }            // b
      },
      0, p.batch, 1, 0, p.group_size, 1);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
