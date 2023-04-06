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

MaceStatus GroupConv2d(OpContext *context,
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
                       const int groups) {
  VLOG(3) << "GroupConv2d OPENCL OP";
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);
  VLOG(3) << "groups: " << groups;

  auto executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    VLOG(3) << "kernel is null";
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("group_conv2d");
    built_options.emplace("-Dgroup_conv2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    common::utils::FillBuiltOptions(&built_options, activation);

    VLOG(3) << "time to execute";

    MACE_RETURN_IF_ERROR(executor->BuildKernel("group_conv2d", kernel_name,
                                               built_options, kernel));
    VLOG(3) << "executed";
    *kwg_size =
        static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(*kernel));
  }

  VLOG(3) << "REGISTERED";
  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  MACE_OUT_OF_RANGE_INIT(*kernel);

  // Support different input size
  if (IsResetArgsNeeded(context, *prev_input_shape, input->shape())) {
    VLOG(3) << "RESET ARGS";
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(*kernel);
    MACE_SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(input->memory<cl::Image>()));
    kernel->setArg(idx++, *(filter->memory<cl::Image>()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->memory<cl::Image>()));
    }
    kernel->setArg(idx++, *(output->mutable_memory<cl::Image>()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, activation_coefficient);
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<uint32_t>(height));
    kernel->setArg(idx++, static_cast<uint32_t>(width));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(3)));
    kernel->setArg(idx++, static_cast<uint32_t>(stride_h));
    kernel->setArg(idx++, static_cast<uint32_t>(stride_w));
    kernel->setArg(idx++, padding[0] / 2);
    kernel->setArg(idx++, padding[1] / 2);
    kernel->setArg(idx++, dilations[0]);
    kernel->setArg(idx++, dilations[1]);
    kernel->setArg(idx++, groups);

    *prev_input_shape = input->shape();
    VLOG(3) << "RESET ARGS SUCCESS";
  }

  std::string tuning_key = Concat(
      "group_conv2dgeneral_opencl_kernel", output->dim(0), output->dim(1),
      output->dim(2), output->dim(3), filter->dim(2), filter->dim(3));
  //   std::vector<uint32_t> lws =
  //       LocalWS(executor, gws, filter->dim(2) * filter->dim(3), *kwg_size);
  //   MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, *kernel, tuning_key,
  //   gws,
  //                                            lws, context->future(),
  //                                            context));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
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
  VLOG(3) << "GroupConv2dKernel::Compute";
  index_t kernel_h = filter->dim(2);
  VLOG(3) << "GroupConv2dKernel::Compute kernel_h = " << kernel_h << "";
  index_t kernel_w = filter->dim(3);
  VLOG(3) << "GroupConv2dKernel::Compute kernel_w = " << kernel_w << "";
  //   if (dilations[0] > 1 && (strides[0] > 1 || kernel_h == 1)) {
  //     LOG(WARNING) << "OpenCL conv2d kernel with "
  //                  << "filter" << kernel_h << "x" << kernel_w << ","
  //                  << " stride " << strides[0] << "x" << strides[1]
  //                  << ",dilations " << dilations[0] << "x" << dilations[1]
  //                  << " is not implemented yet.";
  //     MACE_NOT_IMPLEMENTED;
  //   }
  VLOG(3) << "GroupConv2dKernel::Compute 1";
  // Reshape output
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    VLOG(3) << "Padding data is empty";
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    VLOG(3) << "Padding data is not empty";
    paddings = padding_data;
    // Print the contents of padding_data.data()
    // Get the size of the array
    VLOG(3) << "padding_data.size() (kernekl) = " << padding_data.size();
    // for (int i = 0; i < 2; i++) {
    //   VLOG(3) << "padding_data.data() = " << padding_data.data()[i];
    // }
    // CalcOutputSize(input->shape().data(), filter->shape().data(),
    //                padding_data.data(), dilations, strides, RoundType::FLOOR,
    //                output_shape.data());
    // Calculate the outputsize given that this is a group convolution,
    // Where the input is split into groups, and each group is convolved
    // separately.
    // The output is then concatenated.
    // Print the outputshape
    // Output is in the form of [batch, output_height, output_width,
    // output_channels

    output_shape[0] = input->dim(0);
    output_shape[1] = (input->dim(2) + paddings[1] -
                       dilations[1] * (filter->dim(2) - 1) - 1) /
                          strides[1] +
                      1;
    output_shape[2] = (input->dim(2) + paddings[1] -
                       dilations[1] * (filter->dim(2) - 1) - 1) /
                          strides[1] +
                      1;
    output_shape[3] = input->dim(3);

    // Print the outputshape
    for (int i = 0; i < 4; i++) {
      VLOG(3) << "output_shape.data() = " << output_shape.data()[i];
    }
  }
  VLOG(3) << "GroupConv2dKernel::Compute 2";
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  VLOG(3) << "GroupConv2dKernel::Compute 3";
  std::function<MaceStatus()> conv_func;
  VLOG(3) << "GROUOP_CONV2D";
  if (wino_blk_size) {
    VLOG(3) << "Use winograd kernel";
  }

  //   if (relux_max_limit > 0) {
  //     VLOG(3) << "Use ReluX kernel";
  //   }
  //   if (bias != nullptr) {
  //     VLOG(3) << "Use bias kernel";
  //   }
  //   if (activation == ActivationType::NOOP) {
  //     VLOG(3) << "Use no activation kernel";
  //   }
  //   if (activation_coefficient != 1) {
  //     VLOG(3) << "Use activation coefficient kernel";
  //   }
  //   if (context != nullptr) {
  //     VLOG(3) << "Use context kernel";
  //   }
  //   if (kwg_size_ != 0) {
  //     VLOG(3) << "Use kwg_size kernel";
  //   }
  //   Conv2d, which this is copied from, is implmeentedn in conv2d_general

  // TODO: (bcp) If I happen to add more microkernels, remember to add kernel_
  //   to kernel_[i] and kwg_size_ to kwg_size_[i] conv_func = [&]() ->
  VLOG(3) << "GroupConv2dKernel::Compute 4";
  conv_func = [&]() -> MaceStatus {
    return GroupConv2d(context, &kernel_, input, filter, bias, strides[0],
                       strides[1], paddings.data(), dilations, activation,
                       relux_max_limit, activation_coefficient, &input_shape_,
                       output, &kwg_size_, groups);
  };
  VLOG(3) << "GroupConv2dKernel::Compute 5";
  return conv_func();
  //   return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
