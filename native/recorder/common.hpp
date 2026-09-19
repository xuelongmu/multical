#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace recorder {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
inline void write_all(int fd, const void* data, size_t length) {
    auto p = static_cast<const uint8_t*>(data);
    while (length) {
        auto n = ::write(fd, p, length);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "Write failed: " + std::string(strerror(errno)));
        p += n; length -= n;
    }
}
inline void sync_path(const fs::path& path, bool directory = false) {
    int fd = ::open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
    require(fd >= 0, "Cannot open for flush: " + path.string());
    int result = ::fsync(fd); ::close(fd);
    require(result == 0, "Flush failed: " + path.string());
}
inline void atomic_json(const fs::path& path, const Json& value) {
    auto temp = path.string() + ".tmp";
    int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    require(fd >= 0, "Cannot write " + temp);
    try { auto text = value.dump(2) + "\n"; write_all(fd, text.data(), text.size());
          require(::fsync(fd) == 0, "Manifest flush failed"); }
    catch (...) { ::close(fd); throw; }
    ::close(fd); fs::rename(temp, path); sync_path(path.parent_path(), true);
}
inline uint64_t free_bytes(const fs::path& path) {
    struct statvfs info{};
    require(::statvfs(path.c_str(), &info) == 0, "Cannot inspect recording storage");
    return uint64_t(info.f_bavail) * info.f_frsize;
}
inline void emit(const Json& value) {
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << value.dump() << '\n' << std::flush;
}

template<class T> class Queue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<T> values;
    size_t capacity;
    bool closed = false;
public:
    explicit Queue(size_t limit) : capacity(limit) {}
    bool push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed || values.size() >= capacity) return false;
        values.push_back(std::move(value)); changed.notify_one(); return true;
    }
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&]{return closed || !values.empty();});
        if (values.empty()) return false;
        value = std::move(values.front()); values.pop_front(); return true;
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex);
        if(values.empty())return false;
        value=std::move(values.front());values.pop_front();return true;
    }
    void close() { std::lock_guard<std::mutex> lock(mutex); closed = true; changed.notify_all(); }
    size_t size() { std::lock_guard<std::mutex> lock(mutex); return values.size(); }
};

struct CameraInfo {
    std::string serial, mac;
    int width = 1440, height = 1080;
    double exposure_us = 1600, gain_db = 0;
    Json settings = Json::object();
    Json json() const { return {{"serial",serial},{"mac",mac},{"width",width},{"height",height},
        {"exposure_us",exposure_us},{"gain_db",gain_db},{"settings",settings}}; }
};

// Two full-resolution Bayer slots per camera. Header is exactly eight uint64s:
// seqlock, preview sequence, global sequence, scheduled ns, exposure-end ns,
// hardware frame ID, width, height. Never expose a buffer still owned by the SDK.
class PreviewMemory {
    uint8_t* memory = nullptr;
    size_t length = 0, frame_size = 0;
public:
    PreviewMemory(const std::string& path, int cameras, int width, int height) {
        frame_size = size_t(width) * height;
        length = size_t(cameras) * 2 * (64 + frame_size);
        int fd = ::open(path.c_str(), O_RDWR);
        require(fd >= 0, "Cannot open preview shared memory");
        memory = static_cast<uint8_t*>(::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        ::close(fd); require(memory != MAP_FAILED, "Cannot map preview memory");
    }
    ~PreviewMemory() { if (memory && memory != MAP_FAILED) ::munmap(memory, length); }
    void publish(int camera, uint64_t preview, uint64_t sequence, uint64_t scheduled,
                 uint64_t timestamp, uint64_t frame_id, int width, int height, const uint8_t* data, size_t stride) {
        auto slot = memory + (size_t(camera) * 2 + preview % 2) * (64 + frame_size);
        auto header = reinterpret_cast<uint64_t*>(slot);
        uint64_t version = __atomic_load_n(header, __ATOMIC_RELAXED);
        __atomic_store_n(header, version + 1, __ATOMIC_SEQ_CST);
        header[1]=preview; header[2]=sequence; header[3]=scheduled; header[4]=timestamp;
        header[5]=frame_id; header[6]=width; header[7]=height;
        for (int row=0; row<height; ++row) std::memcpy(slot+64+size_t(row)*width, data+size_t(row)*stride, width);
        __atomic_store_n(header, version + 2, __ATOMIC_RELEASE);
    }
};
} // namespace recorder
