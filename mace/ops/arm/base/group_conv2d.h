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

#ifndef MACE_OPS_ARM_BASE_GROUP_CONV2D_H_
#define MACE_OPS_ARM_BASE_GROUP_CONV2D_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/gemm.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/group_conv2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

struct GroupConvComputeParam {
  const index_t batch;
  const index_t in_channels;
  const index_t in_height;
  const index_t in_width;
  const index_t out_channels;
  const index_t out_height;
  const index_t out_width;

  const index_t in_image_size;
  const index_t out_image_size;
  const index_t in_batch_size;
  const index_t out_batch_size;

  const index_t group_size;

  utils::ThreadPool &thread_pool;

  GroupConvComputeParam(const index_t b,
                        const index_t in_c,
                        const index_t in_h,
                        const index_t in_w,
                        const index_t out_c,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t in_size,
                        const index_t out_size,
                        const index_t in_b_size,
                        const index_t out_b_size,
                        const index_t groups,
                        utils::ThreadPool *thrd_pool)
      : batch(b),
        in_channels(in_c),
        in_height(in_h),
        in_width(in_w),
        out_channels(out_c),
        out_height(out_h),
        out_width(out_w),
        in_image_size(in_size),
        out_image_size(out_size),
        in_batch_size(in_b_size),
        out_batch_size(out_b_size),
        group_size(groups),
        thread_pool(*thrd_pool) {}
};

class GroupConv2dBase : public delegator::GroupConv2d {
 public:
  explicit GroupConv2dBase(const delegator::GroupConv2dParam &param,
                           int type_size)
      : delegator::GroupConv2d(param), type_size_(type_size) {}

  virtual ~GroupConv2dBase() = default;

 protected:
  void CalOutputShapeAndInputPadSize(const std::vector<index_t> &input_shape,
                                     const std::vector<index_t> &filter_shape,
                                     std::vector<index_t> *output_shape,
                                     std::vector<int> *in_pad_size);

  void CalOutputBoundaryWithoutUsingInputPad(
      const std::vector<index_t> &output_shape,
      const std::vector<int> in_pad_size,
      std::vector<index_t> *out_bound);

  void CalOutputShapeAndPadSize(const Tensor *input,
                                const Tensor *filter,
                                const int out_tile_height,
                                const int out_tile_width,
                                std::vector<index_t> *output_shape,
                                std::vector<int> *in_pad_size,
                                std::vector<int> *out_pad_size);

  MaceStatus ResizeOutAndPadInOut(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output,
                                  const int out_tile_height,
                                  const int out_tile_width,
                                  std::unique_ptr<const Tensor> *padded_input,
                                  std::unique_ptr<Tensor> *padded_output);

  void PadInput(const Tensor &src,
                const int pad_top,
                const int pad_left,
                Tensor *dst);
  void UnPadOutput(const Tensor &src, Tensor *dst);

  GroupConvComputeParam PreWorkAndGetConv2DParam(const OpContext *context,
                                                 const Tensor *in_tensor,
                                                 Tensor *out_tensor);

 private:
  int type_size_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_GROUP_CONV2D_H_
