/* file: LcnBackwardInput.java */
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
 * @defgroup lcn_layers_backward Backward Local contrast normalization (LCN) Layer
 * @brief Contains classes for the backward local contrast normalization layer
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNBACKWARDINPUT"></a>
 * @brief Input object for the backward local contrast normalization layer
 */
public final class LcnBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LcnBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward local contrast normalization layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(LcnLayerDataId id, Tensor val) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LcnBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward local contrast normalization layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LcnLayerDataId id) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
