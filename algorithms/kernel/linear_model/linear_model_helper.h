/* file: linear_model_helper.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef _LINEAR_MODEL_HELPER_H__
#define _LINEAR_MODEL_HELPER_H__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_defines.h"
#include "algorithms/kernel/service_error_handling.h"

#include "algorithms/threading/threading.h"
#include "externals/service_blas.h"
#include "externals/service_spblas.h"
#include "service/kernel/service_data_utils.h"
#include "service/kernel/service_environment.h"
//#include <iostream>

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct BSHelper
{
    static size_t getBlockSize(const size_t nRows, const size_t nFeatures, const size_t nResponses)
    {

        const double cacheFullness     = 0.8;
        const size_t maxRowsPerBlock   = 512;
        const size_t minRowsPerBlockL1 = 256;
        const size_t minRowsPerBlockL2 = 8;
        const size_t rowsFitL1         = (getL1CacheSize() / sizeof(algorithmFPType) * cacheFullness - (nFeatures * nResponses)) / (nFeatures + nResponses);
        const size_t rowsFitL2         = (getL2CacheSize() / sizeof(algorithmFPType) * cacheFullness - (nFeatures * nResponses)) / (nFeatures + nResponses);
        size_t blockSize               = 96;

        if (rowsFitL1 >= minRowsPerBlockL1 && rowsFitL1 <= maxRowsPerBlock)
        {
            blockSize = rowsFitL1;
        }
        else if (rowsFitL2 >= minRowsPerBlockL2 && rowsFitL2 <= maxRowsPerBlock)
        {
            blockSize = rowsFitL2;
        }
        else if (rowsFitL2 >= maxRowsPerBlock)
        {
            blockSize = maxRowsPerBlock;
        }
        // std::cout << "rowsFitL1: " << rowsFitL1 << std::endl;
        // std::cout << "rowsFitL2: " << rowsFitL2 << std::endl;
        // std::cout << "BlockSize: " << blockSize << std::endl;
        return blockSize;
    }
};

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
