/* file: env_detect.cpp */
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

/*
//++
//  Definitions of structures used for environment detection.
//--
*/

#include <immintrin.h>

#include "services/env_detect.h"
#include "services/daal_defines.h"
#include "service/kernel/service_defines.h"
#include "externals/service_service.h"
#include "algorithms/threading/threading.h"
#include "services/error_indexes.h"

#include "service/kernel/service_topo.h"
#include "algorithms/threading/service_thread_pinner.h"

static daal::services::Environment::LibraryThreadingType daal_thr_set = (daal::services::Environment::LibraryThreadingType)-1;
static bool isInit                                                    = false;

namespace daal
{
namespace services
{
void daal_free_buffers();
}
} // namespace daal

void getCpuidInfo(daal::services::internal::CPUIDinfo * info, unsigned int eax, unsigned int ecx)
{
    #if defined(_MSC_VER)
    __cpuidex((int *)info, eax, ecx);
    #else
    unsigned int ebx, edx;
        #if defined(__i386__) && defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__("movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
        #else
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
        #endif
    info->EAX = eax;
    info->EBX = ebx;
    info->ECX = ecx;
    info->EDX = edx;
    #endif
}

// Here is a section of sequences of chars "Genu", "ineI", "ntel" is interpreted like uint32 values
#define DAAL_INTERNAL_GENU 0x756e6547
#define DAAL_INTERNAL_INEI 0x49656e69
#define DAAL_INTERNAL_NTEL 0x6c65746e
DAAL_EXPORT bool daal::services::Environment::isIntel() {
    daal::services::internal::CPUIDinfo info{0, 0, 0, 0};
    getCpuidInfo(&info, 0, 0);
    return info.EBX == DAAL_INTERNAL_GENU &&
           info.ECX == DAAL_INTERNAL_INEI &&
           info.EDX == DAAL_INTERNAL_NTEL;
}
#undef DAAL_INTERNAL_GENU
#undef DAAL_INTERNAL_INEI
#undef DAAL_INTERNAL_NTEL

DAAL_EXPORT daal::services::Environment * daal::services::Environment::getInstance()
{
    static daal::services::Environment instance;
    return &instance;
}

DAAL_EXPORT int daal::services::Environment::freeInstance()
{
    return 0;
}

DAAL_EXPORT int daal::services::Environment::getCpuId(int enable)
{
    initNumberOfThreads();
    if (!_env.cpuid_init_flag)
    {
        _cpu_detect(enable);
        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

DAAL_EXPORT int daal::services::Environment::enableInstructionsSet(int enable)
{
    initNumberOfThreads();
    if (!_env.cpuid_init_flag)
    {
        _cpu_detect(enable);
        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

DAAL_EXPORT int daal::services::Environment::setCpuId(int cpuid)
{
    initNumberOfThreads();
    int host_cpuid = __daal_serv_cpu_detect(daal::services::Environment::avx512_mic_e1);

    if (!_env.cpuid_init_flag)
    {
        if (~size_t(0) == _env.cpuid)
        {
            if (cpuid > daal::lastCpuType || cpuid < 0) return daal::services::ErrorCpuIsInvalid;

            if (cpuid > host_cpuid)
            {
                _cpu_detect(daal::services::Environment::avx512_mic_e1);
            }
            else
            {
                if (daal::avx512_mic == cpuid && daal::avx512_mic != host_cpuid)
                {
                    _cpu_detect(daal::services::Environment::avx512_mic_e1);
                }
                else
                {
                    _env.cpuid = cpuid;
                }
            }
        }

        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set()
{
    return daal_thr_set;
}

DAAL_EXPORT void daal::services::Environment::setDynamicLibraryThreadingTypeOnWindows(daal::services::Environment::LibraryThreadingType thr)
{
    daal_thr_set = thr;
    initNumberOfThreads();
}

DAAL_EXPORT daal::services::Environment::Environment() : _init(0)
{
    _env.cpuid_init_flag = false;
    _env.cpuid           = -1;
    this->setDefaultExecutionContext(CpuExecutionContext());
}

DAAL_EXPORT daal::services::Environment::Environment(const Environment & e) : daal::services::Environment::Environment() {}

DAAL_EXPORT void daal::services::Environment::initNumberOfThreads()
{
    if (isInit) return;

    /* if HT enabled - set _numThreads to physical cores num */
    if (daal::internal::Service<>::serv_get_ht())
    {
        /* Number of cores = number of cpu packages * number of cores per cpu package */
        int ncores = daal::internal::Service<>::serv_get_ncpus() * daal::internal::Service<>::serv_get_ncorespercpu();

        /*  Re-set number of threads if ncores is valid and different to _numThreads */
        if ((ncores > 0) && (ncores < _daal_threader_get_max_threads()))
        {
            daal::services::Environment::setNumberOfThreads(ncores);
        }
    }
    isInit = true;
}

DAAL_EXPORT daal::services::Environment::~Environment()
{
    daal::services::daal_free_buffers();
    _daal_tbb_task_scheduler_free(_init);
}

void daal::services::Environment::_cpu_detect(int enable)
{
    initNumberOfThreads();
    if (~size_t(0) == _env.cpuid)
    {
        _env.cpuid = __daal_serv_cpu_detect(enable);
    }
}

DAAL_EXPORT void daal::services::Environment::setNumberOfThreads(const size_t numThreads)
{
    isInit = true;
    daal::setNumberOfThreads(numThreads, &_init);
}

DAAL_EXPORT size_t daal::services::Environment::getNumberOfThreads() const
{
    return daal::threader_get_threads_number();
}

DAAL_EXPORT int daal::services::Environment::setMemoryLimit(MemType type, size_t limit)
{
    initNumberOfThreads();
    return daal::internal::Service<>::serv_set_memory_limit(type, limit);
}

DAAL_EXPORT void daal::services::Environment::enableThreadPinning(const bool enableThreadPinningFlag)
{
    initNumberOfThreads();
#if !(defined DAAL_THREAD_PINNING_DISABLED)
    daal::services::internal::thread_pinner_t * thread_pinner = daal::services::internal::getThreadPinner(true, read_topology, delete_topology);

    if (thread_pinner != NULL)
    {
        thread_pinner->set_pinning(enableThreadPinningFlag);
    }
#endif
    return;
}
