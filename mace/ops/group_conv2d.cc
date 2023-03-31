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
#include "mace/ops/delegator/conv_2d.h"
#include "mace/ops/group_conv2d.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <RuntimeType D, class T>
class GroupConv2d;

template <class T>
class GroupConv2d<RuntimeType::RT_CPU, T> : public GroupConv2dOpBase {
 public:
  explicit GroupConv2d(OpConstructContext *context)
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
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    if (conv2d_delegator_ == nullptr) {
      auto tag =
          MACE_DELEGATOR_KEY(Conv2d, RuntimeType::RT_CPU, T, kCpuImplType);
      if (kCpuImplType == NEON) {
        // the following params are used to decide which conv delegator to use
        const index_t stride_h = strides_[0];
        const index_t stride_w = strides_[1];
        const index_t dilation_h = dilations_[0];
        const index_t dilation_w = dilations_[1];
        const index_t filter_h = filter->dim(2);
        const index_t filter_w = filter->dim(3);
        const index_t input_channels = input->dim(1);
        const index_t channels = filter->dim(0);
        // NOTE: delegator is fixed after first round of running,
        // although winograd depends on input params.
        // We do not support changeable filter for now.
        if (filter_h == 1 && filter_w == 1 && stride_h == 1 && stride_w == 1 &&
            dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K1x1);
        } else if (filter_h == 3 && filter_w == 3 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          if (input_channels >= 8 && channels >= 8) {
            tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                        kCpuImplType, K3x3Winograd);
          } else {
            tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                        kCpuImplType, K3x3S1);
          }
        } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 &&
                   stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K3x3S2);
        } else if (filter_h == 5 && filter_w == 5 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K5x5S1);
        } else if (filter_h == 7 && filter_w == 7 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K7x7S1);
        } else if (filter_h == 7 && filter_w == 7 && stride_h == 2 &&
                   stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K7x7S2);
        } else if (filter_h == 7 && filter_w == 7 && stride_h == 3 &&
                   stride_w == 3 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K7x7S3);
        } else if (filter_h == 1 && filter_w == 7 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K1x7S1);
        } else if (filter_h == 7 && filter_w == 1 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K7x1S1);
        } else if (filter_h == 1 && filter_w == 15 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K1x15S1);
        } else if (filter_h == 15 && filter_w == 1 && stride_h == 1 &&
                   stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
          tag = MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU, T,
                                      kCpuImplType, K15x1S1);
        }
      }
      delegator::Conv2dParam param(strides_, dilations_, paddings_,
                                   padding_type_);
      conv2d_delegator_ =
          delegator::Conv2d::Create(context->workspace(), tag, param);
    }

    conv2d_delegator_->Compute(context, input, filter, output);
    bias_add_delegator_->Compute(context, output, bias, output);
    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }


 private:
  std::unique_ptr<delegator::Activation> activation_delegator_;
  std::unique_ptr<delegator::BiasAdd> bias_add_delegator_;
  std::unique_ptr<delegator::Conv2d> conv2d_delegator_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

// #ifdef MACE_ENABLE_OPENCL
// template <>
// class GroupConv2d<RuntimeType::RT_OPENCL, float> : public Operation {};
// #endif  // MACE_ENABLE_OPENCL

void RegisterGroupConv2d(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "GroupConv2d", GroupConv2d, RuntimeType::RT_CPU,
                   float);

  //   MACE_REGISTER_GPU_OP(op_registry, "GroupConv2d", GroupConv2d);

  RegisterFilterDataFormat(op_registry, "GroupConv2d");
}

}  // namespace ops
}  // namespace mace