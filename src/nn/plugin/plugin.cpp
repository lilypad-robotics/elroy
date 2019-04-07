#include "nn/plugin/plugin.h"

static const int OUTPUT_CLS_SIZE = 91;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const char* OUTPUT_BLOB_NAME0 = "NMS";

const int concatAxis[2] = {1, 1};
const bool ignoreBatch[2] = {false, false};

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 200, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);
    CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

    std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

    assert(d == a + length);
}

FlattenConcat::~FlattenConcat()
{
    CHECK(cudaFreeHost(mInputConcatAxis));
    CHECK(cudaFreeHost(mCopySize));
}

int FlattenConcat::getNbOutputs() const override { 
    return 1; 
}

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
{
    assert(nbInputDims >= 1);
    assert(index == 0);
    mNumInputs = nbInputDims;
    CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    mOutputConcatAxis = 0;
#ifdef SSD_INT8_DEBUG
    std::cout << " Concat nbInputs " << nbInputDims << "\n";
    std::cout << " Concat axis " << mConcatAxisID << "\n";
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j)
            std::cout << " Concat InputDims[" << i << "]"
                      << "d[" << j << " is " << inputs[i].d[j] << "\n";
#endif
    for (int i = 0; i < nbInputDims; ++i)
    {
        int flattenInput = 0;
        assert(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1) assert(inputs[i].d[0] == inputs[0].d[0]);
        if (mConcatAxisID != 2) assert(inputs[i].d[1] == inputs[0].d[1]);
        if (mConcatAxisID != 3) assert(inputs[i].d[2] == inputs[0].d[2]);
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                   mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                   mConcatAxisID == 3 ? mOutputConcatAxis : 1);
}

int FlattenConcat::initialize() override
{
    CHECK(cublasCreate(&mCublas));
    return 0;
}

void FlattenConcat::terminate() override
{
    CHECK(cublasDestroy(mCublas));
}

size_t FlattenConcat::getWorkspaceSize(int) const override { 
    return 0; 
}

int FlattenConcat::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
{
    int numConcats = 1;
    assert(mConcatAxisID != 0);
    numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

    if (!mIgnoreBatch) numConcats *= batchSize;

    float* output = reinterpret_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const float* input = reinterpret_cast<const float*>(inputs[i]);
        float* inputTemp;
        CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

        CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

        for (int n = 0; n < numConcats; ++n)
        {
            CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                              inputTemp + n * mInputConcatAxis[i], 1,
                              output + (n * mOutputConcatAxis + offset), 1));
        }
        CHECK(cudaFree(inputTemp));
        offset += mInputConcatAxis[i];
    }

    return 0;
}

size_t FlattenConcat::getSerializationSize() override
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
}

void FlattenConcat::serialize(void* buffer) override
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    assert(d == a + getSerializationSize());
}

void FlattenConcat::configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
{
    assert(nbOutputs == 1);
    mCHW = inputs[0];
    assert(inputs[0].nbDims == 3);
    CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
    }
}
