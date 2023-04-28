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
    // TODO: (bcp) The naive implemenation does not work with the onnx file
    // anymore because ops are fused with activations
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    if (group_conv2d_delegator_ == nullptr) {
      auto tag =
          MACE_DELEGATOR_KEY(GroupConv2d, RuntimeType::RT_CPU, T, kCpuImplType);
      if (kCpuImplType == NEON) {
        // the following params are used to decide which conv delegator to use
        const index_t stride_h = strides_[0];
        const index_t stride_w = strides_[1];
        const index_t dilation_h = dilations_[0];
        const index_t dilation_w = dilations_[1];
        const index_t filter_h = filter->dim(2);
        const index_t filter_w = filter->dim(3);

        // TODO: (bcp) Only 3x3S1 and 3x3S2 are supported for now to complete
        // RegNet support

        VLOG(3) << "ENTERING CPU GROUP CONV2D DELEGATOR";
        if (filter_h == 3 && filter_w == 3 && stride_h == 1 && stride_w == 1 &&
            dilation_h == 1 && dilation_w == 1) {
          VLOG(3) << "ENTERING CPU GROUP CONV2D K3x3S1 DELEGATOR";
          tag = MACE_DELEGATOR_KEY_EX(GroupConv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K3x3S1);

        } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 &&
                   stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
          VLOG(3) << "ENTERING CPU GROUP CONV2D K1x1S1 DELEGATOR";
          tag = MACE_DELEGATOR_KEY_EX(GroupConv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K3x3S2);
        }
      }
      delegator::GroupConv2dParam param(strides_, dilations_, paddings_,
                                        padding_type_, group_);
      group_conv2d_delegator_ =
          delegator::GroupConv2d::Create(context->workspace(), tag, param);
    }


    // naive_group_conv2d(input->data<T>(), filter->data<T>(),
    //                    input->shape().data(), filter->shape().data(),
    //                    output->shape().data(), strides_, paddings_, group_,
    //                    output->mutable_data<T>());

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
      kernel_ = make_unique<opencl::image::GroupConv2dKernel>();
    } else {
      // TODO (bcp): support group conv2d with buffer (the ONNX files only
      // require image)
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
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
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