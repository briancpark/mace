// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#ifndef MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
#define MACE_OPS_OPENCL_IMAGE_CONV_2D_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/group_conv2d.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

extern MaceStatus GroupConv2d(OpContext *context,
                              cl::Kernel *kernel,
                              const Tensor *input,
                              const Tensor *filter,
                              const Tensor *bias,
                              const int stride_h,
                              const int stride_w,
                              const int *padding,
                              const int *dilations,
                              const ActivationType activation,
                              const float relux_max_limit,
                              const float activation_coefficient,
                              std::vector<index_t> *prev_input_shape,
                              Tensor *output,
                              uint32_t *kwg_size,
                              const int groups);

class GroupConv2dKernel : public OpenCLGroupConv2dKernel {
 public:
  // TODO: (bcp) Wheter or not to really implement winograd should be determined
  // later
  bool CheckUseWinograd(OpenclExecutor *executor,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<index_t> &output_shape,
                        const int *strides,
                        const int *dilations,
                        int *wino_block_size) override;

  MaceStatus Compute(OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *bias,
                     const int *strides,
                     const Padding &padding_type,
                     const std::vector<int> &padding_data,
                     const int *dilations,
                     const ActivationType activation,
                     const float relux_max_limit,
                     const float activation_coefficient,
                     const int wino_blk_size,
                     const int groups,
                     Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
