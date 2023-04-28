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

#ifndef MACE_OPS_ARM_BASE_GROUP_CONV2D_3X3_H_
#define MACE_OPS_ARM_BASE_GROUP_CONV2D_3X3_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/group_conv2d_mxn.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template <typename T>
class GroupConv2dK3x3S1 : public GroupConv2dKMxN<T> {
 public:
  explicit GroupConv2dK3x3S1(const delegator::GroupConv2dParam &param)
      : GroupConv2dKMxN<T>(param, 2, 4) {}
  virtual ~GroupConv2dK3x3S1() {}

  MaceStatus DoCompute(const ConvComputeParam &p,
                       const T *filter,
                       const T *input_data,
                       T *output_data) override;
};

template <typename T>
class GroupConv2dK3x3S2 : public GroupConv2dKMxN<T> {
 public:
  explicit GroupConv2dK3x3S2(const delegator::GroupConv2dParam &param)
      : GroupConv2dKMxN<T>(param, 1, 4) {}
  virtual ~GroupConv2dK3x3S2() {}

  MaceStatus DoCompute(const ConvComputeParam &p,
                       const T *filter,
                       const T *input_data,
                       T *output_data) override;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_GROUP_CONV2D_3X3_H_
