/* file: df_classification_predict_dense_default_batch_container.h */
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

/*
//++
//  Implementation of decision forest algorithm container -- a class
//  that contains fast decision forest prediction kernels
//  for supported architectures.
//--
*/

#include "decision_forest_classification_predict.h"
#include "df_classification_predict_dense_default_batch.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace prediction
{
namespace interface1
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    decision_forest::classification::Model *m = static_cast<decision_forest::classification::Model *>(input->get(classifier::prediction::model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());

    const classifier::interface1::Parameter *par = static_cast<classifier::interface1::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), a, m, r, par->nClasses);
}

}
namespace interface2
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    decision_forest::classification::Model *m = static_cast<decision_forest::classification::Model *>(input->get(classifier::prediction::model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());

    const classifier::Parameter *par = static_cast<classifier::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
        daal::services::internal::hostApp(*input), a, m, r, par->nClasses);
}

}
}
}
}
}
} // namespace daal
