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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/delegator/activation.h"
#include "mace/ops/delegator/bias_add.h"
// #include "mace/ops/delegator/conv_2d.h"
#include "mace/ops/delegator/group_conv2d.h"
#include "mace/ops/group_conv2d.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/group_conv2d.h"
#include "mace/ops/opencl/image/group_conv2d.h"
#include "mace/runtimes/opencl/opencl_runtime.h"
#include "mace/runtimes/opencl/transform/buffer_transformer.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <RuntimeType D, class T>
class GroupConv2dOp;

template <class T>
class GroupConv2dOp<RuntimeType::RT_CPU, T> : public GroupConv2dOpBase {
 public:
  explicit GroupConv2dOp(OpConstructContext *context)
      : GroupConv2dOpBase(context),
        activation_delegator_(delegator::Activation::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(
                Activation, RuntimeType::RT_CPU, T, kCpuImplType),
            /* Delegator copied from conv_2d.cc */
            delegator::ActivationParam(
                ops::StringToActivationType(
                    Operation::GetOptionalArg<std::string>("activation",
                                                           "NOOP")),
                Operation::GetOptionalArg<float>("max_limit", 0.0f),
                Operation::GetOptionalArg<float>("activation_coefficient",
                                                 0.0f),
                Operation::GetOptionalArg<float>("hardsigmoid_alpha", 0.f),
                Operation::GetOptionalArg<float>("hardsigmoid_beta", 0.f)))),
        bias_add_delegator_(delegator::BiasAdd::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(BiasAdd, RuntimeType::RT_CPU, T, kCpuImplType),
            DelegatorParam())) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    // printf("BEGIN RUNNING GroupConv2d\n");
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    // TODO: (bcp) DEBUGGING

    // Print the input, filter, bias, and output shapes
    // printf("input shape: %lu %lu %lu %lu\n", input->dim(0), input->dim(1),
    //        input->dim(2), input->dim(3));
    // printf("filter shape: %lu %lu %lu %lu\n", filter->dim(0), filter->dim(1),
    //        filter->dim(2), filter->dim(3));
    // printf("bias shape: %lu\n", bias->dim(0));
    // printf("output shape: %lu %lu %lu %lu\n", output->dim(0), output->dim(1),
    //        output->dim(2), output->dim(3));

    // Print the format of the input, filter, bias, and output


    // print out the group size
    VLOG(3) << "group size: " << group_;

    // if (group_conv2d_delegator_ == nullptr) {
    //   VLOG(3) << "Creating group conv2d delegator";
    //   auto tag =
    //       MACE_DELEGATOR_KEY(GroupConv2d, RuntimeType::RT_CPU, T,
    //       kCpuImplType);

    // } else {
    //   VLOG(3) << "Reusing group conv2d delegator";
    // }

    // This is the function signature:
    // void naive_group_conv2d(const T *input,
    //                     const T *filter,
    //                     const index_t *in_shape,
    //                     const index_t *filter_shape,
    //                     const index_t *out_shape,
    //                     const std::vector<int> &strides,
    //                     const std::vector<int> &paddings,
    //                     const int group,
    //                     T *output)

    naive_group_conv2d(input->data<T>(), filter->data<T>(),
                       input->shape().data(), filter->shape().data(),
                       output->shape().data(), strides_, paddings_, group_,
                       output->mutable_data<T>());

    // if (conv2d_delegator_ == nullptr) {
    //   auto tag =
    //       MACE_DELEGATOR_KEY(Conv2d, RuntimeType::RT_CPU, T,
    //       kCpuImplType);
    //   if (kCpuImplType == NEON) {
    //     // the following params are used to decide which conv delegator
    //     to use const index_t stride_h = strides_[0]; const index_t
    //     stride_w = strides_[1]; const index_t dilation_h = dilations_[0];
    //     const index_t dilation_w = dilations_[1]; const index_t filter_h
    //     = filter->dim(2); const index_t filter_w = filter->dim(3); const
    //     index_t input_channels = input->dim(1); const index_t channels =
    //     filter->dim(0);
    //     // NOTE: delegator is fixed after first round of running,
    //     // although winograd depends on input params.
    //     // We do not support changeable filter for now.
    //     if (filter_h == 1 && filter_w == 1 && stride_h == 1 && stride_w
    //     == 1
    //     &&
    //         dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K1x1);
    //     } else if (filter_h == 3 && filter_w == 3 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       if (input_channels >= 8 && channels >= 8) {
    //         tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                     kCpuImplType, K3x3Winograd);
    //       } else {
    //         tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                     kCpuImplType, K3x3S1);
    //       }
    //     } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 &&
    //                stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K3x3S2);
    //     } else if (filter_h == 5 && filter_w == 5 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K5x5S1);
    //     } else if (filter_h == 7 && filter_w == 7 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K7x7S1);
    //     } else if (filter_h == 7 && filter_w == 7 && stride_h == 2 &&
    //                stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K7x7S2);
    //     } else if (filter_h == 7 && filter_w == 7 && stride_h == 3 &&
    //                stride_w == 3 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K7x7S3);
    //     } else if (filter_h == 1 && filter_w == 7 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K1x7S1);
    //     } else if (filter_h == 7 && filter_w == 1 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K7x1S1);
    //     } else if (filter_h == 1 && filter_w == 15 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K1x15S1);
    //     } else if (filter_h == 15 && filter_w == 1 && stride_h == 1 &&
    //                stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    //       tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
    //                                   kCpuImplType, K15x1S1);
    //     }
    //   }
    // This is a temporary tag for now:
    auto tag = MACE_DELEGATOR_KEY_EX(GroupConv2d, RuntimeType::RT_CPU, T,
                                     kCpuImplType, K3x3Winograd);

    delegator::GroupConv2dParam param(strides_, dilations_, paddings_,
                                      padding_type_, group_);
    group_conv2d_delegator_ =
        delegator::GroupConv2d::Create(context->workspace(), tag, param);
    group_conv2d_delegator_->Compute(context, input, filter, output);
    bias_add_delegator_->Compute(context, output, bias, output);
    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }


 private:
  std::unique_ptr<delegator::Activation> activation_delegator_;
  std::unique_ptr<delegator::BiasAdd> bias_add_delegator_;
  std::unique_ptr<delegator::GroupConv2d> group_conv2d_delegator_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template <>
class GroupConv2dOp<RuntimeType::RT_OPENCL, float> : public GroupConv2dOpBase {
 public:
  explicit GroupConv2dOp(OpConstructContext *context)
      : GroupConv2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        activation_coefficient_(
            Operation::GetOptionalArg<float>("activation_coefficient", 0.0f)),
        wino_block_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)) {
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      // TODO: support group conv2d with image
      VLOG(1) << "IMAGEBUFF";
      kernel_ = make_unique<opencl::image::GroupConv2dKernel>();
    } else {
      VLOG(1) << "IMAGEBUFFBUFF";
      mem_type = MemoryType::GPU_BUFFER;
      kernel_ = make_unique<opencl::image::GroupConv2dKernel>();
      //   kernel_ = make_unique<opencl::buffer::GroupConv2dKernel>();
    }
    // Transform input tensor to target format
    auto *input_tensor =
        context->workspace()->GetTensor(operator_def_->input(INPUT));
    if (input_tensor != nullptr && input_tensor->is_weight()) {
      MACE_CHECK(TransformFilter(context, operator_def_.get(), 0,
                                 BufferContentType::IN_OUT_CHANNEL,
                                 mem_type) == MaceStatus::MACE_SUCCESS);
    }
    // Transform filter tensor to target format
    auto *filter_tensor =
        context->workspace()->GetTensor(operator_def_->input(FILTER));
    if (filter_tensor != nullptr && filter_tensor->is_weight()) {
      if ((wino_block_size_ == 2 || wino_block_size_ == 4) &&
          (kernel_->CheckUseWinograd(
              OpenclRuntime::Get(context)->GetOpenclExecutor(),
              filter_tensor->shape(),
              std::vector<index_t>(
                  operator_def_->output_shape(0).dims().begin(),
                  operator_def_->output_shape(0).dims().end()),
              strides_.data(), dilations_.data(), &wino_block_size_))) {
        MACE_CHECK(TransformFilter(context, operator_def_.get(), 1,
                                   BufferContentType::WINOGRAD_FILTER, mem_type,
                                   wino_block_size_) ==
                   MaceStatus::MACE_SUCCESS);
      } else {
        wino_block_size_ = 0;
        MACE_CHECK(TransformFilter(context, operator_def_.get(), 1,
                                   BufferContentType::CONV2D_FILTER,
                                   mem_type) == MaceStatus::MACE_SUCCESS);
      }
    } else {
      // we don't know whether the kernal support winograd, so disable it.
      wino_block_size_ = 0;
    }

    if (operator_def_->input_size() > 2) {
      auto ret = TransformFilter(context, operator_def_.get(), 2,
                                 BufferContentType::ARGUMENT, mem_type);
      MACE_CHECK(ret == MaceStatus::MACE_SUCCESS);
    }
    VLOG(1) << "GCONV INIT DONE";
  }

  MaceStatus Run(OpContext *context) override {
    VLOG(1) << "RUNNING GPU GCONV";
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    // Print out the contents of the padding
    // printout the size of the paddings
    // print the filter shape, input shape, output shape
    VLOG(1) << "Filter shape is " << filter->dim(0) << " " << filter->dim(1)
            << " " << filter->dim(2) << " " << filter->dim(3);
    VLOG(1) << "Input shape is " << input->dim(0) << " " << input->dim(1) << " "
            << input->dim(2) << " " << input->dim(3);
    VLOG(1) << "Output shape is " << output->dim(0) << " " << output->dim(1)
            << " " << output->dim(2) << " " << output->dim(3);
    VLOG(1) << "Paddings size is " << paddings_.size();
    for (int i = 0; i < 2; i++) {
      VLOG(1) << "Padding " << i << " is " << paddings_[i];
    }
    VLOG(1) << "RUNNING GPU GCONV INIT";
    return kernel_->Compute(
        context, input, filter, bias, strides_.data(), padding_type_, paddings_,
        dilations_.data(), activation_, relux_max_limit_,
        activation_coefficient_, wino_block_size_, group_, output);
  }

 protected:
  BufferContentType GetInputTensorContentType(size_t idx) const override {
    if (idx == FILTER) {
      if (wino_block_size_ == 0) {
        return BufferContentType::CONV2D_FILTER;
      } else {
        return BufferContentType::WINOGRAD_FILTER;
      }
    }
    return Operation::GetInputTensorContentType(idx);
  }

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float activation_coefficient_;
  std::unique_ptr<OpenCLGroupConv2dKernel> kernel_;
  int wino_block_size_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterGroupConv2d(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "GroupConv2d", GroupConv2dOp,
                   RuntimeType::RT_CPU, float);

  MACE_REGISTER_GPU_OP(op_registry, "GroupConv2d", GroupConv2dOp);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("GroupConv2D")
          .SetInputMemoryTypeSetter([](OpConditionContext *context) -> void {
            SetFilterMemoryType(context, BufferContentType::CONV2D_FILTER);
          }));
#endif  // MACE_ENABLE_OPENCL

  RegisterFilterDataFormat(op_registry, "GroupConv2d");
}

}  // namespace ops
}  // namespace mace