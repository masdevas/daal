#include "daal.h"
#include "service.h"
#include "src/externals/service_ittnotify.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

template <typename algorithmFPType, typename Algorithm>
services::Status process_data_for_kernel_func(Algorithm& algorithm, FileDataSource<CSVFeatureManager>& leftDataSource, FileDataSource<CSVFeatureManager>& rightDataSource) {
    services::Status status;
    data_management::BlockDescriptor<algorithmFPType> block;

    // ******************************************************************
    // ##### Create and set result
    NumericTablePtr x = leftDataSource.getNumericTable();
    size_t linesNumberX = x->getNumberOfRows();
    size_t columnsNumberX = x->getNumberOfColumns();
    //std::cout << "X: " << linesNumberX << '*' << columnsNumberX << std::endl;
    size_t soaResultColumnsNumber = 2048;
    services::SharedPtr<HomogenNumericTable<algorithmFPType> > buffer = HomogenNumericTable<algorithmFPType>::create(soaResultColumnsNumber * 2, linesNumberX, data_management::NumericTableIface::doAllocate, &status);
    DAAL_CHECK_STATUS_VAR(status);
    algorithmFPType* bufferData = buffer->getArray();
    
    services::SharedPtr<SOANumericTable> kernelComputeTable = SOANumericTable::create(soaResultColumnsNumber, linesNumberX, data_management::DictionaryIface::equal, &status);
    DAAL_CHECK_STATUS_VAR(status);

    for (size_t i = 0; i < soaResultColumnsNumber; ++i)
    {
        algorithmFPType* localDataPtr = bufferData + 2 * i * linesNumberX;
        DAAL_CHECK_STATUS(status, kernelComputeTable->setArray<algorithmFPType>(localDataPtr, i));
    }

    kernel_function::ResultPtr shRes(new kernel_function::Result());
    shRes->set(kernel_function::values, kernelComputeTable);
    algorithm.setResult(shRes);
   
    // ##### Creating and setting result finished
    // ##### Create Y
    NumericTablePtr yFull = rightDataSource.getNumericTable();
    size_t linesNumberYFull = yFull->getNumberOfRows();
    size_t columnsNumberYFull = yFull->getNumberOfColumns();
    //std::cout << "Y: " << linesNumberYFull << '*' << columnsNumberYFull << std::endl;
    
    status |= yFull->getBlockOfRows(0, soaResultColumnsNumber, data_management::readOnly, block);
    DAAL_CHECK_STATUS_VAR(status);
    algorithmFPType * yFullData = block.getBlockPtr();
    services::SharedPtr<HomogenNumericTable<algorithmFPType> > y = HomogenNumericTable<algorithmFPType>::create(yFullData, columnsNumberYFull, soaResultColumnsNumber);
    //std::cout << "Yred: " << y->getNumberOfRows() << '*' << y->getNumberOfColumns() << std::endl;
    // ##### Creating Y finished

    algorithm.input.set(kernel_function::X, x);
    algorithm.input.set(kernel_function::Y, y);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(Kernel);
        algorithm.compute();
    }
    
    status |= yFull->releaseBlockOfRows(block);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}