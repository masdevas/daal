/* file: BatchNormalizationForwardInputLayerDataId.java */
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
 * @ingroup batch_normalization_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONFORWARDINPUTLAYERDATAID"></a>
 * \brief Available identifiers of input objects for the forward batch normalization layer
 */
public final class BatchNormalizationForwardInputLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public BatchNormalizationForwardInputLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int populationMeanId = 3;
    @Native private static final int populationVarianceId = 4;

    public static final BatchNormalizationForwardInputLayerDataId populationMean     = new BatchNormalizationForwardInputLayerDataId(populationMeanId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population mean computed on the previous stage */
    public static final BatchNormalizationForwardInputLayerDataId populationVariance = new BatchNormalizationForwardInputLayerDataId(populationVarianceId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population variance computed on the previous stage */
}
/** @} */
