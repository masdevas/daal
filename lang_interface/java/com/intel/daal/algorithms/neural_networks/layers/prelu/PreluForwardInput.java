/* file: PreluForwardInput.java */
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

/**
 * @defgroup prelu_forward Forward Parametric Rectifier Linear Unit (pReLU) Layer
 * @brief Contains classes for the forward prelu layer
 * @ingroup prelu
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELUFORWARDINPUT"></a>
 * @brief %Input object for the forward prelu layer
 */
public class PreluForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PreluForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns dimensions of weights tensor
     * @param parameter  Layer parameter
     * @return Dimensions of weights tensor
     */
    public long[] getWeightsSizes(PreluParameter parameter)
    {
        return cGetWeightsSizes(cObject, parameter.getCObject());
    }

    private native long[] cGetWeightsSizes(long cObject, long cParameter);
}
/** @} */
