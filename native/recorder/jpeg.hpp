#pragma once
#include "common.hpp"
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <npp.h>
#include <nppi_color_conversion.h>

namespace recorder {
inline void cuda_check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
inline void jpeg_check(nvjpegStatus_t code) { require(code == NVJPEG_STATUS_SUCCESS, "nvJPEG error " + std::to_string(code)); }
class JpegEncoder {
    int width, height;
    cudaStream_t stream = nullptr;
    nvjpegHandle_t handle = nullptr;
    nvjpegEncoderState_t state = nullptr;
    nvjpegEncoderParams_t params = nullptr;
    uint8_t *staging = nullptr, *bayer = nullptr, *rgb = nullptr;
    NppStreamContext context{};
    std::vector<uint8_t> output;
public:
    JpegEncoder(int w, int h, int quality) : width(w), height(h) {
        try {
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            require(nppGetStreamContext(&context)==NPP_SUCCESS,"Cannot get NPP device context");
            context.hStream=stream; cuda_check(cudaStreamGetFlags(stream,&context.nStreamFlags));
            jpeg_check(nvjpegCreateSimple(&handle));
            jpeg_check(nvjpegEncoderStateCreate(handle,&state,stream));
            jpeg_check(nvjpegEncoderParamsCreate(handle,&params,stream));
            jpeg_check(nvjpegEncoderParamsSetQuality(params,quality,stream));
            jpeg_check(nvjpegEncoderParamsSetSamplingFactors(params,NVJPEG_CSS_420,stream));
            jpeg_check(nvjpegEncoderParamsSetOptimizedHuffman(params,0,stream));
            size_t capacity;
            jpeg_check(nvjpegEncodeGetBufferSize(handle,params,width,height,&capacity));
            output.resize(capacity);
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&staging),size_t(w)*h));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&bayer),size_t(w)*h));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&rgb),size_t(w)*h*3));
            std::vector<uint8_t> warm(size_t(w)*h,128); encode(warm.data());
        } catch (...) { release(); throw; }
    }
    ~JpegEncoder() { release(); }
    void release() {
        if(stream)cudaStreamSynchronize(stream);
        if(params)nvjpegEncoderParamsDestroy(params);
        if(state)nvjpegEncoderStateDestroy(state);
        if(handle)nvjpegDestroy(handle);
        if(staging)cudaFreeHost(staging);
        if(bayer)cudaFree(bayer);
        if(rgb)cudaFree(rgb);
        if(stream)cudaStreamDestroy(stream);
    }
    std::vector<uint8_t> encode(const uint8_t* input) {
        size_t size=size_t(width)*height;
        std::memcpy(staging,input,size);
        cuda_check(cudaMemcpyAsync(bayer,staging,size,cudaMemcpyHostToDevice,stream));
        // Explicit RGGB phase, full sensor geometry, no additional camera colour
        // matrix/gamma/white balance. Record this transform in the take manifest.
        auto code=nppiCFAToRGB_8u_C1C3R_Ctx(bayer,width,{width,height},{0,0,width,height},
                    rgb,width*3,NPPI_BAYER_RGGB,NPPI_INTER_UNDEFINED,context);
        require(code==NPP_SUCCESS,"NPP debayer error " + std::to_string(code));
        nvjpegImage_t image{};image.channel[0]=rgb;image.pitch[0]=width*3;
        jpeg_check(nvjpegEncodeImage(handle,state,params,&image,NVJPEG_INPUT_RGBI,width,height,stream));
        size_t length=output.size();
        jpeg_check(nvjpegEncodeRetrieveBitstream(handle,state,output.data(),&length,stream));
        cuda_check(cudaStreamSynchronize(stream));
        require(length<=output.size(),"JPEG exceeded allocated output");
        return std::vector<uint8_t>(output.begin(),output.begin()+length);
    }
};
} // namespace recorder
