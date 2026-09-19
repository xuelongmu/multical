// Replay-only JPEG throughput experiment. No camera ownership or production writes.
// Build instructions and measurement limitations: RECORDING_BENCHMARK_RESULTS.md.
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <turbojpeg.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr int W = 1440, H = 1080;
constexpr size_t Y = W * H, FRAME = Y * 3 / 2;
using Clock = std::chrono::steady_clock;
void cu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void nj(nvjpegStatus_t status) {
    if (status != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvJPEG error " + std::to_string(status));
}

struct Encoder {
    virtual size_t encode(const unsigned char* input) = 0;
    virtual const unsigned char* bytes() const = 0;
    virtual ~Encoder() = default;
};

struct CpuEncoder : Encoder {
    tjhandle handle = nullptr;
    unsigned char* output = nullptr;
    unsigned long capacity;
    int quality;
    explicit CpuEncoder(int q) : quality(q) {
        handle = tjInitCompress();
        capacity = tjBufSize(W, H, TJSAMP_420);
        output = tjAlloc(capacity);
        if (!handle || !output) throw std::runtime_error("TurboJPEG allocation failed");
    }
    ~CpuEncoder() override { if (handle) tjDestroy(handle); if (output) tjFree(output); }
    size_t encode(const unsigned char* input) override {
        const unsigned char* planes[] = {input, input + Y, input + Y + Y / 4};
        const int strides[] = {W, W / 2, W / 2};
        unsigned long length = capacity;
        if (tjCompressFromYUVPlanes(handle, planes, W, strides, H, TJSAMP_420,
                &output, &length, quality, TJFLAG_FASTDCT | TJFLAG_NOREALLOC))
            throw std::runtime_error(tjGetErrorStr2(handle));
        return length;
    }
    const unsigned char* bytes() const override { return output; }
};

struct GpuEncoder : Encoder {
    nvjpegHandle_t handle = nullptr;
    nvjpegEncoderState_t state = nullptr;
    nvjpegEncoderParams_t params = nullptr;
    cudaStream_t stream = nullptr;
    unsigned char *device = nullptr, *staging = nullptr;
    std::vector<unsigned char> output;
    explicit GpuEncoder(int quality) {
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        nj(nvjpegCreateSimple(&handle));
        nj(nvjpegEncoderStateCreate(handle, &state, stream));
        nj(nvjpegEncoderParamsCreate(handle, &params, stream));
        nj(nvjpegEncoderParamsSetQuality(params, quality, stream));
        nj(nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_420, stream));
        nj(nvjpegEncoderParamsSetOptimizedHuffman(params, 0, stream));
        size_t capacity;
        nj(nvjpegEncodeGetBufferSize(handle, params, W, H, &capacity));
        output.resize(capacity);
        cu(cudaMalloc(reinterpret_cast<void**>(&device), FRAME));
        cu(cudaMallocHost(reinterpret_cast<void**>(&staging), FRAME));
    }
    ~GpuEncoder() override {
        if (stream) cudaStreamSynchronize(stream);
        if (params) nvjpegEncoderParamsDestroy(params);
        if (state) nvjpegEncoderStateDestroy(state);
        if (handle) nvjpegDestroy(handle);
        if (device) cudaFree(device);
        if (staging) cudaFreeHost(staging);
        if (stream) cudaStreamDestroy(stream);
    }
    size_t encode(const unsigned char* input) override {
        std::memcpy(staging, input, FRAME);
        cu(cudaMemcpyAsync(device, staging, FRAME, cudaMemcpyHostToDevice, stream));
        nvjpegImage_t image{};
        image.channel[0] = device;
        image.channel[1] = device + Y;
        image.channel[2] = device + Y + Y / 4;
        image.pitch[0] = W; image.pitch[1] = image.pitch[2] = W / 2;
        nj(nvjpegEncodeYUV(handle, state, params, &image, NVJPEG_CSS_420, W, H, stream));
        size_t length = output.size();
        nj(nvjpegEncodeRetrieveBitstream(handle, state, output.data(), &length, stream));
        cu(cudaStreamSynchronize(stream));
        if (length > output.size()) throw std::runtime_error("JPEG output exceeded buffer");
        return length;
    }
    const unsigned char* bytes() const override { return output.data(); }
};

int main(int argc, char** argv) try {
    if (argc < 7) throw std::runtime_error(
        "usage: jpeg-bench cpu|gpu QUALITY WORKERS FRAMES OUTPUT_DIR clips... [--paced] [--export]");
    const std::string backend = argv[1];
    const int quality = std::stoi(argv[2]), workers = std::stoi(argv[3]), jobs = std::stoi(argv[4]);
    if ((backend != "cpu" && backend != "gpu") || quality < 1 || quality > 100 ||
        workers < 1 || workers > 64 || jobs < 1) throw std::runtime_error("Invalid parameters");
    const std::filesystem::path output_dir(argv[5]);
    std::filesystem::create_directories(output_dir);
    bool paced = false, export_clips = false;
    std::vector<std::vector<unsigned char>> clips;
    for (int i = 6; i < argc; ++i) {
        if (std::string(argv[i]) == "--paced") { paced = true; continue; }
        if (std::string(argv[i]) == "--export") { export_clips = true; continue; }
        std::ifstream input(argv[i], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open input");
        auto size = input.tellg();
        if (size <= 0 || size % FRAME) throw std::runtime_error("Invalid YUV420 frame size");
        clips.emplace_back(static_cast<size_t>(size));
        input.seekg(0); input.read(reinterpret_cast<char*>(clips.back().data()), size);
        if (!input) throw std::runtime_error("Incomplete input read");
    }
    if (clips.empty()) throw std::runtime_error("No clips");
    std::atomic<int> next{0}, completed{0};
    std::atomic<unsigned long long> total_bytes{0};
    std::vector<double> latencies(jobs);
    std::vector<double> delivery_ms(jobs);
    std::vector<int> worker_frames(workers, 0);
    std::mutex mutex;
    std::condition_variable condition;
    bool start = false, failed = false;
    std::atomic<bool> cancelled{false};
    int ready = 0;
    std::string error;
    Clock::time_point before;
    std::vector<std::thread> threads;
    for (int t = 0; t < workers; ++t) threads.emplace_back([&, t] {
        try {
            std::unique_ptr<Encoder> encoder = backend == "gpu"
                ? std::unique_ptr<Encoder>(new GpuEncoder(quality))
                : std::unique_ptr<Encoder>(new CpuEncoder(quality));
            encoder->encode(clips[0].data()); // Warm allocations before the timed interval.
            if (t == 0) for (size_t s = 0; s < clips.size(); ++s) {
                auto length = encoder->encode(clips[s].data());
                std::ofstream sample(output_dir / ("sample" + std::to_string(s) + ".jpg"), std::ios::binary);
                sample.write(reinterpret_cast<const char*>(encoder->bytes()), length);
                if (!sample) throw std::runtime_error("Cannot write JPEG sample");
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready; condition.notify_all();
                condition.wait(lock, [&] { return start || failed; });
                if (failed) return;
            }
            unsigned long long local_bytes = 0;
            while (!cancelled) {
                const int job = next.fetch_add(1);
                if (job >= jobs) break;
                const auto& clip = clips[(job % 25) % clips.size()];
                const size_t frame_index = (job / 25) % (clip.size() / FRAME);
                const auto due = before + std::chrono::nanoseconds((job / 25) * 1000000000LL / 60);
                if (paced) std::this_thread::sleep_until(due);
                const auto encode_start = Clock::now();
                local_bytes += encoder->encode(clip.data() + frame_index * FRAME);
                latencies[job] = std::chrono::duration<double, std::milli>(Clock::now() - encode_start).count();
                delivery_ms[job] = std::chrono::duration<double, std::milli>(Clock::now() - due).count();
                ++worker_frames[t]; ++completed;
            }
            total_bytes += local_bytes;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex);
            failed = true; cancelled = true; error = e.what(); condition.notify_all();
        }
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return ready == workers || failed; });
        before = Clock::now(); start = true; condition.notify_all();
    }
    for (auto& thread : threads) thread.join();
    const double seconds = std::chrono::duration<double>(Clock::now() - before).count();
    if (failed) throw std::runtime_error(error);
    if (completed != jobs) throw std::runtime_error("Missing jobs");
    std::sort(latencies.begin(), latencies.end());
    std::sort(delivery_ms.begin(), delivery_ms.end());
    std::cout << "{\"backend\":\"" << backend << "\",\"quality\":" << quality
              << ",\"workers\":" << workers << ",\"frames\":" << completed
              << ",\"seconds\":" << seconds << ",\"fps\":" << jobs / seconds
              << ",\"p50_ms\":" << latencies[jobs / 2]
              << ",\"p95_ms\":" << latencies[static_cast<size_t>(jobs * .95)]
              << ",\"paced\":" << (paced ? "true" : "false")
              << ",\"delivery_p99_ms\":" << (paced ? delivery_ms[static_cast<size_t>(jobs * .99)] : 0)
              << ",\"delivery_max_ms\":" << (paced ? delivery_ms.back() : 0)
              << ",\"bytes_per_frame\":" << double(total_bytes) / jobs
              << ",\"projected_25cam_MB_s\":" << double(total_bytes) / jobs * 1500 / 1e6
              << ",\"projected_120s_GB\":" << double(total_bytes) / jobs * 180000 / 1e9
              << ",\"scope\":\"Predecoded YUV420, warm worker pool; GPU includes host upload and bitstream download. Excludes camera, debayer, mux and disk.\"}\n";
    if (export_clips) {
        std::unique_ptr<Encoder> encoder = backend == "gpu"
            ? std::unique_ptr<Encoder>(new GpuEncoder(quality))
            : std::unique_ptr<Encoder>(new CpuEncoder(quality));
        // Export after timing so quality comparisons don't change throughput.
        for (size_t s = 0; s < clips.size(); ++s) {
            std::ofstream output(output_dir / ("clip" + std::to_string(s) + ".mjpg"), std::ios::binary);
            for (size_t offset = 0; offset < clips[s].size(); offset += FRAME) {
                auto length = encoder->encode(clips[s].data() + offset);
                output.write(reinterpret_cast<const char*>(encoder->bytes()), length);
            }
            if (!output) throw std::runtime_error("Cannot export JPEG clip");
        }
    }
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
