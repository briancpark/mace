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

#ifndef MACE_OPS_GROUPCONV_2D_H_
#define MACE_OPS_GROUPCONV_2D_H_

#include <algorithm>
#include <string>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/types.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

class GroupConv2dOpBase : public Operation {
 public:
  explicit GroupConv2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(
            Operation::GetOptionalArg<int>("padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        dilations_(Operation::GetRepeatedArgs<int>("dilations", {1, 1})),
        group_(Operation::GetOptionalArg<int>("group", 1)) {}

 protected:
  std::vector<int> strides_;  // [stride_h, stride_w]
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int group_;
};


// This is the naive group conv2d implementation, used for reference
template <typename T>
void naive_group_conv2d(const T *input,
                        const T *filter,
                        const index_t *in_shape,
                        const index_t *filter_shape,
                        const index_t *out_shape,
                        const std::vector<int> &strides,
                        const std::vector<int> &paddings,
                        const int group,
                        T *output) {
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t g = 0; g < group; ++g) {
      for (index_t oc = 0; oc < out_shape[1] / group; ++oc) {
        for (index_t oh = 0; oh < out_shape[2]; ++oh) {
          for (index_t ow = 0; ow < out_shape[3]; ++ow) {
            index_t out_idx =
                b * out_shape[1] * out_shape[2] * out_shape[3] +
                g * out_shape[1] / group * out_shape[2] * out_shape[3] +
                oc * out_shape[2] * out_shape[3] + oh * out_shape[3] + ow;
            T sum = 0;
            for (index_t ic = 0; ic < in_shape[1] / group; ++ic) {
              for (index_t kh = 0; kh < filter_shape[2]; ++kh) {
                for (index_t kw = 0; kw < filter_shape[3]; ++kw) {
                  index_t ih = oh * strides[0] - paddings[0] + kh;
                  index_t iw = ow * strides[1] - paddings[1] + kw;
                  if (ih >= 0 && ih < in_shape[2] && iw >= 0 &&
                      iw < in_shape[3]) {
                    index_t in_idx =
                        b * in_shape[1] * in_shape[2] * in_shape[3] +
                        g * in_shape[1] / group * in_shape[2] * in_shape[3] +
                        ic * in_shape[2] * in_shape[3] + ih * in_shape[3] + iw;
                    index_t filter_idx =
                        g * filter_shape[1] / group * filter_shape[2] *
                            filter_shape[3] +
                        oc * filter_shape[2] * filter_shape[3] +
                        kh * filter_shape[3] + kw;
                    sum += input[in_idx] * filter[filter_idx];
                  }
                }
              }
            }
            output[out_idx] = sum;
          }
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_GROUPCONV_2D_H_
