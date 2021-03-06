/* file: softmax_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//++
//  Declaration of template function that calculate softmax.
//--


#ifndef __SOFTMAX_KERNEL_H__
#define __SOFTMAX_KERNEL_H__

#include "math/softmax.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "threading.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace softmax
{
namespace internal
{
/**
 *  \brief Kernel for softmax calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SoftmaxKernel : public Kernel
{
public:
    Status compute(const NumericTable *inputTable, NumericTable *resultTable);

private:
    const size_t _nRowsInBlock = 5000;

    inline Status processBlock(const NumericTable &inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                               NumericTable &resultTable);
};

} // namespace daal::internal
} // namespace softmax
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
