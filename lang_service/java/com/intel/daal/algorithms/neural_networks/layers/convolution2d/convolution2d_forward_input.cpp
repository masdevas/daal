/* file: convolution2d_forward_input.cpp */
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

#include <jni.h>
#include "com_intel_daal_algorithms_neural_networks_layers_convolution2d_Convolution2dForwardInput.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Convolution2dForwardInput
 * Method:    cGetWeightsSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Convolution2dForwardInput_cGetWeightsSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    Collection<size_t> dims = ((convolution2d::forward::Input *)inputAddr)->getWeightsSizes((convolution2d::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Convolution2dForwardInput
 * Method:    cGetBiasesSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Convolution2dForwardInput_cGetBiasesSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    Collection<size_t> dims = ((convolution2d::forward::Input *)inputAddr)->getBiasesSizes((convolution2d::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}
