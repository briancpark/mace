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


#ifndef MACE_OPS_DELEGATOR_GROUP_CONV_2D_H_
#define MACE_OPS_DELEGATOR_GROUP_CONV_2D_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

// enum ConvType {
//   K1x1,
//   K1x7S1,
//   K7x1S1,
//   K1x15S1,
//   K15x1S1,
//   K3x3S1,
//   K3x3S2,
//   K3x3Winograd,
//   K5x5S1,
//   K7x7S1,
//   K7x7S2,
//   K7x7S3,
// };

namespace delegator {

struct GroupConv2dParam : public DelegatorParam {
  explicit GroupConv2dParam(const std::vector<int> &strides,
                            const std::vector<int> &dilations,
                            const std::vector<int> &paddings,
                            const Padding padding_type,
                            const int group)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        group_(group) {}

  const std::vector<int> &strides_;
  const std::vector<int> &dilations_;
  const std::vector<int> &paddings_;
  const Padding padding_type_;
  const int group_;
};

class GroupConv2d : public OpDelegator {
 public:
  explicit GroupConv2d(const delegator::GroupConv2dParam &param)
      : OpDelegator(param),
        strides_(param.strides_),
        dilations_(param.dilations_),
        paddings_(param.paddings_),
        padding_type_(param.padding_type_),
        group_(param.group_) {}
  virtual ~GroupConv2d() = default;

  MACE_DEFINE_DELEGATOR_CREATOR(GroupConv2d)

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *filter,
                             Tensor *output) = 0;

 protected:
  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  const int group_;
};

}  // namespace delegator
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DELEGATOR_GROUP_CONV_2D_H_
