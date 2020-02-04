/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "daal.h"
#include "common_helpers.h"
#include "com_intel_daal_algorithms_classifier_Parameter.h"

USING_COMMON_NAMESPACES()

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_prediction_Parameter_cSetNClasses(JNIEnv * env, jobject thisObj, jlong self, jint votingMethod)
{
    unpack<decision_forest::classification::prediction::Parameter>(self).votingMethod = votingMethod;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_prediction_Parameter_cGetNClasses(JNIEnv * env, jobject thisObj, jlong self)
{
    return (jlong)(unpack<classifier::Parameter>(self).votingMethod);
}
