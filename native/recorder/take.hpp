#pragma once
#include "jpeg.hpp"
#include "mux.hpp"
#include <limits>

namespace recorder {
struct RawJob { int camera=0,index=0,buffer=0; uint64_t scheduled=0,timestamp=0,frame_id=0; };
struct EncodedJob { RawJob frame; std::vector<uint8_t> bytes; };
struct Counts { std::atomic<uint64_t> received{0},encoded{0},written{0}; };
struct Timing { int64_t start_min=INT64_MAX,start_max=0,mid_min=INT64_MAX,mid_max=0;int count=0; };

class Take {
    static constexpr int pool_size=512, encoder_count=4;
    const uint64_t encoded_limit=1024ULL*1024*1024;
    std::vector<CameraInfo> cameras;
    std::unique_ptr<Counts[]> counts;
    std::vector<std::vector<uint8_t>> buffers;
    Queue<int> free_buffers{pool_size};
    Queue<RawJob> input{pool_size};
    Queue<EncodedJob> output{8192};
    std::vector<std::thread> workers;
    std::thread writer;
    std::atomic<int> workers_alive{encoder_count};
    std::atomic<uint64_t> encoded_bytes{0}, output_bytes{0}, max_queue{0}, max_encoded_bytes{0};
    std::atomic<bool> input_closed{false},finished{false};
    std::mutex ingest_mutex,status_mutex,ready_mutex;
    std::condition_variable ready_changed;
    int encoders_ready=0;
    std::vector<int> next_received;
    std::vector<Timing> timing;
    std::atomic<int64_t> max_start_spread{0},max_mid_spread{0};
    std::string error,phase="arming";
    Json segments=Json::object();
    int64_t drain_deadline=0;
    std::atomic<int64_t> started_at{now_ns()};
    Json configuration;
    std::vector<std::unique_ptr<SegmentMuxer>> muxers;
    std::vector<std::ofstream> ledgers;
    std::vector<int> segment_number,segment_packets;
    std::vector<uint64_t> segment_bytes;

    fs::path camera_dir(int c) const { return directory/"cameras"/cameras[c].serial; }
    fs::path segment_path(int c,int segment) const {
        char name[40];std::snprintf(name,sizeof(name),"segment_%05d.mkv",segment);
        return camera_dir(c)/name;
    }
    void close_segment(int c) {
        if(!muxers[c])return;
        muxers[c]->finish();muxers[c].reset();
        segments[cameras[c].serial].push_back({{"file",fs::relative(segment_path(c,segment_number[c]),directory).string()},
            {"segment",segment_number[c]},{"first_scheduled_index",segment_number[c]*fps*2},
            {"frames",segment_packets[c]},{"jpeg_bytes",segment_bytes[c]},{"finalized",true}});
        ledgers[c].flush();require(bool(ledgers[c]),"Frame ledger flush failed");
        sync_path(camera_dir(c)/"frames.csv");
        atomic_json(segment_path(c,segment_number[c]).string()+".json",segments[cameras[c].serial].back());
    }
    void persist(EncodedJob& job) {
        if(configuration.value("simulated",false)&&configuration.contains("test_fail_write_after"))
            require(output_bytes<uint64_t(configuration["test_fail_write_after"]),"Injected recording storage failure");
        const auto& f=job.frame;int c=f.camera,seg=f.index/(fps*2);
        if(seg!=segment_number[c]) {
            close_segment(c);
            require(free_bytes(directory)>2ULL*1024*1024*1024,"Recording storage reserve exhausted");
            segment_number[c]=seg;segment_packets[c]=0;segment_bytes[c]=0;
            muxers[c]=std::make_unique<SegmentMuxer>(segment_path(c,seg),cameras[c].width,cameras[c].height,fps);
        }
        muxers[c]->write(job.bytes,f.index-seg*fps*2);
        int64_t exposure=std::llround(cameras[c].exposure_us*1000);
        auto& ledger=ledgers[c];
        ledger << f.index << ',' << f.scheduled << ',' << f.timestamp << ',' << (f.timestamp-exposure)
               << ',' << (f.timestamp-exposure/2) << ',' << f.frame_id << ',' << seg << ','
               << segment_packets[c] << ',' << job.bytes.size() << '\n';
        require(bool(ledger),"Frame ledger write failed");
        ++segment_packets[c];segment_bytes[c]+=job.bytes.size();
        output_bytes+=job.bytes.size();++counts[c].written;
    }
    void write_loop() {
        std::vector<std::map<int,EncodedJob>> waiting(cameras.size());
        std::vector<int> next(cameras.size(),0);
        bool disk_ok=true;
        EncodedJob job;
        while(output.pop(job)) {
            auto c=job.frame.camera;
            if(!disk_ok){encoded_bytes-=job.bytes.size();continue;}
            auto result=waiting[c].emplace(job.frame.index,std::move(job));
            if(!result.second){fail("Duplicate encoded frame index");continue;}
            try {
                while(waiting[c].count(next[c])) {
                    auto it=waiting[c].find(next[c]);persist(it->second);
                    encoded_bytes-=it->second.bytes.size();waiting[c].erase(it);++next[c];
                }
            } catch(const std::exception& e){fail(e.what());disk_ok=false;}
        }
        try {
            // Preserve later real frames when a prior frame failed, with the gap
            // explicit in the ledger; never insert a replacement image.
            for(size_t c=0;c<cameras.size();++c) {
                if(!waiting[c].empty())fail("Frame gap in encoded stream " + cameras[c].serial);
                try {
                    if(disk_ok)for(auto& entry:waiting[c])persist(entry.second);
                } catch(const std::exception& e){fail(e.what());disk_ok=false;}
                for(auto& entry:waiting[c])encoded_bytes-=entry.second.bytes.size();
                waiting[c].clear();
                // Attempt every camera even if an earlier disk operation failed.
                try {close_segment(int(c));}catch(const std::exception& e){fail(e.what());}
                try {
                    ledgers[c].flush();require(bool(ledgers[c]),"Frame ledger flush failed");ledgers[c].close();
                    sync_path(camera_dir(c)/"frames.csv");sync_path(camera_dir(c),true);
                }catch(const std::exception& e){fail(e.what());}
            }
            sync_path(directory/"cameras",true);
            bool complete=scheduled_count.load()>0;
            for(size_t c=0;c<cameras.size();++c)
                complete=complete&&counts[c].received==uint64_t(scheduled_count)&&counts[c].written==uint64_t(scheduled_count);
            if(!complete && scheduled_count>0)fail("Received/written counts do not match scheduled frames");
            std::string terminal;
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                terminal=error.empty()?(scheduled_count>0?"saved":"cancelled"):"failed";
            }
            Json manifest=snapshot();manifest["configuration"]=configuration;manifest["segments"]=segments;
            manifest["state"]=terminal;
            manifest["complete"]=complete&&manifest["error"].get<std::string>().empty();
            manifest["timing_review"]=max_start_spread>20000||max_mid_spread>20000;
            manifest["timestamp_model"]="Spinnaker image timestamp is exposure end; start and midpoint subtract the frozen exposure";
            atomic_json(directory/"take.json",manifest);
            {std::lock_guard<std::mutex> lock(status_mutex);phase=terminal;}
        } catch(const std::exception& e) {
            fail(e.what());
            {std::lock_guard<std::mutex> lock(status_mutex);phase="failed";}
            try {auto m=snapshot();m["complete"]=false;m["configuration"]=configuration;m["segments"]=segments;atomic_json(directory/"take.json",m);}catch(...){ }
        }
        finished=true;emit({{"event","recording"},{"status",snapshot()}});
    }
    void encode_loop() {
        std::unique_ptr<JpegEncoder> encoder;
        try {encoder=std::make_unique<JpegEncoder>(cameras[0].width,cameras[0].height,quality);}
        catch(const std::exception& e){fail(e.what());}
        {std::lock_guard<std::mutex> lock(ready_mutex);++encoders_ready;ready_changed.notify_all();}
        RawJob frame;
        while(input.pop(frame)) {
            try {
                require(bool(encoder),"JPEG encoder unavailable");
                auto bytes=encoder->encode(buffers[frame.buffer].data());
                ++counts[frame.camera].encoded;
                auto size=bytes.size();
                auto queued=encoded_bytes.fetch_add(size)+size;
                auto peak=max_encoded_bytes.load();
                while(peak<queued&&!max_encoded_bytes.compare_exchange_weak(peak,queued)){}
                if(queued>encoded_limit){encoded_bytes-=size;fail("Encoded queue exceeded 1 GiB");}
                else if(!output.push({frame,std::move(bytes)})){encoded_bytes-=size;fail("Encoded queue overflow");}
            }catch(const std::exception& e){fail(e.what());}
            free_buffers.push(frame.buffer);
        }
        if(--workers_alive==0)output.close();
    }
public:
    const fs::path directory;
    const int fps,planned_frames,quality;
    std::atomic<int> scheduled_count{0};
    std::atomic<int> stop_kind{0}; // 0 limit, 1 operator, 2 shutdown, 3 fault
    std::atomic<bool> stop_requested{false},scheduling_done{false};
    Take(const fs::path& path,const std::vector<CameraInfo>& info,const Json& request)
        : cameras(info),counts(new Counts[info.size()]),next_received(info.size(),0),
          configuration(request),directory(path),
          fps(request.at("fps")),planned_frames(request.at("frames")),quality(request.value("quality",85)) {
        require(fps>=1&&fps<=60&&planned_frames>=1&&planned_frames<=fps*120,"Take must be 1–60 fps and at most 120 seconds");
        require(quality>=50&&quality<=95,"JPEG quality must be 50–95");
        timing.resize(planned_frames);
        require(!cameras.empty(),"No cameras");
        for(auto& camera:cameras)require(camera.width==cameras[0].width&&camera.height==cameras[0].height,"Mixed sensor geometry is unsupported");
        require(fs::is_directory(directory),"Take directory must be created by the controller");
        int lockfd=::open((directory/".take.lock").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        require(lockfd>=0,"Take directory was already used");::close(lockfd);
        uint64_t estimate=uint64_t(cameras[0].width)*cameras[0].height*cameras.size()*planned_frames*35/100;
        require(free_bytes(directory)>estimate+2ULL*1024*1024*1024,"Insufficient recording space including reserve");
        configuration["cameras"]=Json::array();for(auto& c:cameras)configuration["cameras"].push_back(c.json());
        configuration["codec"]="mjpeg";configuration["container"]="matroska";configuration["chroma"]="4:2:0";
        configuration["color_range"]="full";configuration["input"]="BayerRG8";
        configuration["color_processing"]="NPP RGGB CFA-to-RGB; nvJPEG RGB-to-YCbCr; no added white balance, gamma or colour matrix";
        configuration["raw_pool_frames"]=pool_size;configuration["jpeg_workers"]=encoder_count;
        configuration["segment_frames"]=fps*2;configuration["encoded_queue_capacity_bytes"]=encoded_limit;
        auto initial=snapshot();initial["configuration"]=configuration;initial["complete"]=false;
        atomic_json(directory/"take.json",initial);
        try {
            size_t size=size_t(cameras[0].width)*cameras[0].height;
            buffers.resize(pool_size);for(int i=0;i<pool_size;++i){buffers[i].resize(size);free_buffers.push(i);}
            muxers.resize(cameras.size());ledgers.resize(cameras.size());
            segment_number.resize(cameras.size(),0);segment_packets.resize(cameras.size(),0);segment_bytes.resize(cameras.size(),0);
            for(size_t c=0;c<cameras.size();++c) {
                require(cameras[c].serial.find_first_not_of("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_")==std::string::npos,"Unsafe camera serial");
                fs::create_directories(camera_dir(int(c)));
                segments[cameras[c].serial]=Json::array();
                muxers[c]=std::make_unique<SegmentMuxer>(segment_path(int(c),0),cameras[c].width,cameras[c].height,fps);
                ledgers[c].open(camera_dir(int(c))/"frames.csv");require(bool(ledgers[c]),"Cannot open frame ledger");
                ledgers[c]<<"frame_index,scheduled_ns,timestamp_ns,exposure_start_ns,exposure_midpoint_ns,frame_id,segment,packet,bytes\n";
            }
            writer=std::thread([this]{write_loop();});
            for(int i=0;i<encoder_count;++i)workers.emplace_back([this]{encode_loop();});
            std::unique_lock<std::mutex> lock(ready_mutex);ready_changed.wait(lock,[&]{return encoders_ready==encoder_count;});
            require(!stop_requested,"Cannot arm JPEG encoders");
            {std::lock_guard<std::mutex> status(status_mutex);phase="armed";}
        } catch(...) {
            stop_requested=true;scheduling_done=true;input.close();
            for(auto& worker:workers)if(worker.joinable())worker.join();
            output.close();
            if(writer.joinable())writer.join();
            throw;
        }
    }
    ~Take(){input.close();for(auto& w:workers)if(w.joinable())w.join();output.close();if(writer.joinable())writer.join();}
    void fail(const std::string& message) {
        {std::lock_guard<std::mutex> lock(status_mutex);if(error.empty())error=message;}
        stop_kind=3;
        stop_requested=true;
    }
    void begin() {started_at=now_ns();std::lock_guard<std::mutex> lock(status_mutex);phase="recording";}
    void submit(int camera,int index,uint64_t scheduled,uint64_t timestamp,uint64_t id,const uint8_t* data,size_t stride) {
        std::lock_guard<std::mutex> lock(ingest_mutex);
        if(input_closed||index<0||index>=planned_frames)return;
        if(index!=next_received[camera])fail("Missing or repeated received frame on " + cameras[camera].serial + ": expected " + std::to_string(next_received[camera])+", got "+std::to_string(index));
        next_received[camera]=index+1;++counts[camera].received;
        auto& t=timing[index];int64_t exposure=std::llround(cameras[camera].exposure_us*1000);
        int64_t start=int64_t(timestamp)-exposure,mid=int64_t(timestamp)-exposure/2;
        t.start_min=std::min(t.start_min,start);t.start_max=std::max(t.start_max,start);
        t.mid_min=std::min(t.mid_min,mid);t.mid_max=std::max(t.mid_max,mid);++t.count;
        max_start_spread=std::max(max_start_spread.load(),t.start_max-t.start_min);
        max_mid_spread=std::max(max_mid_spread.load(),t.mid_max-t.mid_min);
        int buffer;
        if(!free_buffers.try_pop(buffer)){fail("Raw recording pool exhausted");return;}
        int w=cameras[camera].width,h=cameras[camera].height;
        for(int y=0;y<h;++y)std::memcpy(buffers[buffer].data()+size_t(y)*w,data+size_t(y)*stride,w);
        if(!input.push({camera,index,buffer,scheduled,timestamp,id})){free_buffers.push(buffer);fail("Raw recording queue overflow");}
        max_queue=std::max<uint64_t>(max_queue.load(),input.size());
    }
    void drain(int64_t last_due) {
        if(scheduling_done)return;
        drain_deadline=std::max(now_ns(),last_due)+3'000'000'000LL;
        {std::lock_guard<std::mutex> lock(status_mutex);phase="draining";}
        scheduling_done=true;
    }
    void poll() {
        if(!scheduling_done||input_closed)return;
        std::lock_guard<std::mutex> lock(ingest_mutex);
        bool all=true;for(size_t c=0;c<cameras.size();++c)all=all&&counts[c].received>=uint64_t(scheduled_count);
        if(all||now_ns()>drain_deadline){if(!all)fail("Timed out waiting for scheduled camera frames");input_closed=true;input.close();}
    }
    bool done()const{return finished;}
    Json snapshot() {
        Json result;
        {std::lock_guard<std::mutex> lock(status_mutex);result={{"state",phase},{"error",error}};}
        result["directory"]=directory.string();result["fps"]=fps;result["planned_frames"]=planned_frames;
        result["scheduled_frames"]=scheduled_count.load();result["elapsed_seconds"]=(now_ns()-started_at.load())/1e9;
        result["stop_reason"]=scheduling_done?std::vector<std::string>{"frame_limit","operator","shutdown","fault"}.at(stop_kind.load()):"";
        result["raw_queue"]=input.size();result["raw_queue_capacity"]=pool_size;result["max_raw_queue"]=max_queue.load();
        result["encoded_queue_bytes"]=encoded_bytes.load();result["max_encoded_queue_bytes"]=max_encoded_bytes.load();result["encoded_queue_capacity_bytes"]=encoded_limit;result["jpeg_bytes_written"]=output_bytes.load();
        result["max_start_spread_us"]=max_start_spread.load()/1000.;result["max_midpoint_spread_us"]=max_mid_spread.load()/1000.;
        result["cameras"]=Json::object();
        for(size_t c=0;c<cameras.size();++c)result["cameras"][cameras[c].serial]={{"received",counts[c].received.load()},
            {"encoded",counts[c].encoded.load()},{"written",counts[c].written.load()}};
        return result;
    }
};
} // namespace recorder
