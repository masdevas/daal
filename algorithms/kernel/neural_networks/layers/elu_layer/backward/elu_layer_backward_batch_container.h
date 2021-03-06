/* file: elu_layer_backward_batch_container.h */
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
//  Implementation of ELU calculation algorithm container.
//--
*/

#ifndef __ELU_LAYER_BACKWARD_BATCH_CONTAINER_H__
#define __ELU_LAYER_BACKWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/elu/elu_layer.h"
#include "elu_layer_backward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace backward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ELUKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const Parameter *par = static_cast<const Parameter *>(_par);
    if (!par->propagateGradient) { return services::Status(); }

    elu::backward::Input *input = static_cast<elu::backward::Input *>(_in);
    elu::backward::Result *result = static_cast<elu::backward::Result *>(_res);

    Tensor *inputGradientTensor = input->get(layers::backward::inputGradient).get();
    Tensor *forwardDataTensor   = input->get(elu::auxData).get();
    Tensor *auxValueTensor      = input->get(elu::auxIntermediateValue).get();
    Tensor *gradientTensor      = result->get(layers::backward::gradient).get();

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::ELUKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       *par, *inputGradientTensor, *forwardDataTensor, auxValueTensor, *gradientTensor);
}
} // namespace interface1
} // namespace backward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
