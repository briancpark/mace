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

#include "mace/ops/opencl/image/group_conv2d.h"

#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/utils.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"
#include "mace/runtimes/opencl/opencl_runtime.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

bool GroupConv2dKernel::CheckUseWinograd(
    OpenclExecutor *executor,
    const std::vector<mace::index_t> &filter_shape,
    const std::vector<mace::index_t> &output_shape,
    const int *strides,
    const int *dilations,
    int *wino_blk_size) {
  if (filter_shape[2] != 3 || filter_shape[3] != 3 || strides[0] > 1 ||
      strides[1] > 1 || dilations[0] > 1 || dilations[1] > 1) {
    return false;
  }
  index_t out_channels = filter_shape[0];
  index_t in_channels = filter_shape[1];
  auto opencl_image_max_size = executor->GetMaxImage2DSize();
  auto check_opencl_limit = [&](int block_size) -> bool {
    int sqr_block = (block_size + 2) * (block_size + 2);
    uint64_t transformed_width = static_cast<uint64_t>(
        output_shape[0] * ((output_shape[1] + block_size - 1) / block_size) *
        ((output_shape[2] + block_size - 1) / block_size));
    return (transformed_width < opencl_image_max_size[0] &&
            static_cast<uint64_t>(sqr_block * in_channels) <
                opencl_image_max_size[1] &&
            static_cast<uint64_t>(sqr_block * out_channels) <
                opencl_image_max_size[1]);
  };
  // GPU only supports 4x4 and 2x2 gpu winograd convolution
  if (*wino_blk_size == 4) {
    // if block size == 4 exceed OpenCL image size limitation, fallback to 2
    if (!check_opencl_limit(4)) {
      *wino_blk_size = 2;
    } else {
      return true;
    }
  }
  return check_opencl_limit(2);
}


MaceStatus GroupConv2dKernel::Compute(OpContext *context,
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
                                      Tensor *output) {
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);

  // Reshape output
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), filter->shape().data(),
                   padding_data.data(), dilations, strides, RoundType::FLOOR,
                   output_shape.data());
  }

  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  std::function<MaceStatus()> groupconv_func;

  if (wino_blk_size) {
    VLOG(3) << "wino_blk_size is not 0";
    // Winograd for group conv2d not implemented yet
    // return not supported error
    MACE_NOT_IMPLEMENTED;
    // Group Conv for 1x1 not implemented yet
    //   } else if (kernel_h == 1 && kernel_w == 1) {
  } else if (kernel_h == 3 && kernel_w == 3) {
    groupconv_func = [&]() -> MaceStatus {
      return GroupConv2dK3x3(context, &kernel_, input, filter, bias, strides[0],
                             strides[1], paddings.data(), dilations, activation,
                             relux_max_limit, activation_coefficient,
                             &input_shape_, output, &kwg_size_, groups);
    };
  } else {
    groupconv_func = [&]() -> MaceStatus {
      return GroupConv2d(context, &kernel_, input, filter, bias, strides[0],
                         strides[1], paddings.data(), dilations, activation,
                         relux_max_limit, activation_coefficient, &input_shape_,
                         output, &kwg_size_, groups);
    };
  }
  return groupconv_func();
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
