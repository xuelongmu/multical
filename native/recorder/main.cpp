#include "cameras.hpp"
#include "take.hpp"
#include <csignal>
#include <poll.h>

using namespace recorder;
static volatile sig_atomic_t interrupted=0;
static void interrupt_handler(int){interrupted=1;}

struct Scheduled {
    uint64_t sequence=0,ptp=0,preview=0;
    int64_t host=0;
    std::shared_ptr<Take> take;
    int index=-1;
    std::mutex mutex;
    std::vector<bool> present;
    bool emitted=false;
};

class Service {
    Json config;
    CameraRig rig;
    PreviewMemory preview;
    std::atomic<bool> stopping{false};
    std::atomic<int64_t> offset;
    std::thread scheduler;
    std::vector<std::thread> receivers;
    std::mutex history_mutex,take_mutex;
    std::condition_variable history_changed;
    std::deque<std::shared_ptr<Scheduled>> history;
    std::shared_ptr<Take> active;
    Json ptp_state;
    std::atomic<uint64_t> unexpected{0},incomplete{0};
    double preview_fps;
    int64_t last_status=0,last_clock=0;

    std::shared_ptr<Take> current() {std::lock_guard<std::mutex> lock(take_mutex);return active;}
    void fault(const std::string& message) {
        auto take=current();if(take&&!take->done())take->fail(message);
        emit({{"event","camera_warning"},{"error",message}});
    }
    std::shared_ptr<Scheduled> match(uint64_t timestamp,int camera) {
        int64_t start=int64_t(timestamp)-std::llround(rig.info[camera].exposure_us*1000);
        std::lock_guard<std::mutex> lock(history_mutex);
        std::shared_ptr<Scheduled> best;
        int64_t distance=1'000'001;
        for(auto it=history.rbegin();it!=history.rend();++it) {
            int64_t delta=std::llabs(start-int64_t((*it)->ptp));
            if(delta<distance){best=*it;distance=delta;}
            if(start>int64_t((*it)->ptp)+1'000'000)break;
        }
        return best;
    }
    void deliver(int camera,const std::shared_ptr<Scheduled>& slot,uint64_t timestamp,uint64_t frame_id,const uint8_t* data,size_t stride) {
        if(slot->take)slot->take->submit(camera,slot->index,slot->ptp,timestamp,frame_id,data,stride);
        if(slot->preview) {
            const auto& c=rig.info[camera];
            preview.publish(camera,slot->preview,slot->sequence,slot->ptp,timestamp,frame_id,c.width,c.height,data,stride);
            std::lock_guard<std::mutex> lock(slot->mutex);slot->present[camera]=true;
        }
    }
    void receive(int camera) {
        auto& hardware=*rig.cameras[camera];
        while(!stopping) {
            ImagePtr image;
            try {
                image=hardware.camera->GetNextImage(500);
                if(image->IsIncomplete()) {
                    ++incomplete;fault("Incomplete image on "+hardware.info.serial+": "+std::to_string(image->GetImageStatus()));
                } else {
                    uint64_t id=image->GetFrameID(),timestamp=image->GetTimeStamp();
                    require(id>hardware.last_frame,"Repeated camera frame ID on "+hardware.info.serial);
                    hardware.last_frame=id;
                    require(image->GetPixelFormat()==PixelFormat_BayerRG8&&image->GetWidth()==1440&&image->GetHeight()==1080,"Camera pixel format or geometry changed");
                    auto slot=match(timestamp,camera);
                    if(slot)deliver(camera,slot,timestamp,id,static_cast<const uint8_t*>(image->GetData()),image->GetStride());
                    else {++unexpected;auto take=current();if(take&&!take->done()&&!take->scheduling_done)take->fail("Frame does not match a scheduled exposure on "+hardware.info.serial);}
                }
                image->Release();image=nullptr;
            } catch(const Spinnaker::Exception& e) {
                if(image)image->Release();
                if(e.GetError()!=SPINNAKER_ERR_TIMEOUT&&!stopping){fault(e.what());std::this_thread::sleep_for(std::chrono::milliseconds(100));}
            } catch(const std::exception& e) {
                if(image)image->Release();
                if(!stopping)fault(e.what());
            }
        }
    }
    void simulate(int camera) {
        auto& c=rig.info[camera];std::vector<uint8_t> pixels(size_t(c.width)*c.height);
        for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x) {
            int channel=(y%2==0)?(x%2==0?0:1):(x%2==0?1:2);
            int stripe=std::min(2,x/(c.width/3));
            pixels[size_t(y)*c.width+x]=uint8_t(channel==stripe?200:35);
        }
        uint64_t sequence=0;
        while(!stopping) {
            std::shared_ptr<Scheduled> slot;
            {
                std::unique_lock<std::mutex> lock(history_mutex);
                history_changed.wait_for(lock,std::chrono::milliseconds(100),[&]{return stopping||(!history.empty()&&history.back()->sequence>sequence);});
                for(auto& item:history)if(item->sequence>sequence){slot=item;break;}
            }
            if(!slot)continue;
            std::this_thread::sleep_until(Clock::time_point(std::chrono::nanoseconds(slot->host)));
            sequence=slot->sequence;
            if(slot->take&&camera==config.value("test_drop_camera",-1)&&slot->index==config.value("test_drop_frame",-1))continue;
            pixels[0]=uint8_t(sequence%256);
            uint64_t timestamp=slot->ptp+std::llround(c.exposure_us*1000)+camera*100;
            deliver(camera,slot,timestamp,sequence,pixels.data(),c.width);
        }
    }
    void schedule() {
        uint64_t sequence=0,preview_sequence=0,last_ptp=0;
        int64_t last_host=now_ns();
        int64_t next_host=now_ns()+250'000'000;
        uint64_t next_ptp=next_host+offset;
        std::shared_ptr<Take> taking;
        uint64_t epoch=0;
        int index=0;
        try {
            while(!stopping) {
                auto candidate=current();
                if(candidate&&!candidate->done()&&!candidate->scheduling_done&&candidate!=taking) {
                    taking=candidate;taking->begin();index=0;
                    epoch=std::max<uint64_t>(last_ptp+20'000'000,now_ns()+offset+150'000'000);
                    next_ptp=epoch;next_host=int64_t(epoch)-offset;
                }
                if(taking&&(taking->stop_requested||index>=taking->planned_frames)) {
                    taking->drain(last_host);taking.reset();
                    next_host=std::max<int64_t>(now_ns()+100'000'000,last_host+int64_t(1e9/preview_fps));
                    next_ptp=next_host+offset;
                }
                if(now_ns()<next_host-80'000'000){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
                if(now_ns()>next_host-2'000'000) {
                    if(taking){taking->fail("Scheduler missed its future-action lead time");continue;}
                    next_host=now_ns()+150'000'000;next_ptp=next_host+offset;continue;
                }
                auto slot=std::make_shared<Scheduled>();
                slot->sequence=++sequence;slot->ptp=next_ptp;slot->host=next_host;slot->take=taking;slot->index=taking?index:-1;
                if(!taking||index%std::max(1,taking->fps/2)==0){slot->preview=++preview_sequence;slot->present.resize(rig.info.size(),false);}
                {
                    std::lock_guard<std::mutex> lock(history_mutex);
                    history.push_back(slot);
                    while(!history.empty()&&history.front()->host<now_ns()-5'000'000'000LL)history.pop_front();
                }
                // A frame scheduled on only some interfaces is still expected;
                // missing replies must appear as missing frames, never a shorter take.
                if(taking)taking->scheduled_count=index+1;
                rig.action(next_ptp);history_changed.notify_all();
                last_ptp=next_ptp;last_host=next_host;
                if(taking){++index;next_ptp=epoch+uint64_t(index)*1'000'000'000ULL/taking->fps;next_host=int64_t(next_ptp)-offset;}
                else {next_host+=int64_t(1e9/preview_fps);next_ptp=next_host+offset;}
            }
        }catch(const std::exception& e){fault(std::string("Scheduler: ")+e.what());stopping=true;}
        if(taking)taking->drain(last_host);
    }
    void publish_preview() {
        std::shared_ptr<Scheduled> selected;
        std::vector<bool> present;
        {
            std::lock_guard<std::mutex> lock(history_mutex);
            for(auto it=history.rbegin();it!=history.rend();++it) {
                auto& s=*it;if(!s->preview||s->emitted)continue;
                std::lock_guard<std::mutex> state(s->mutex);
                bool all=std::all_of(s->present.begin(),s->present.end(),[](bool p){return p;});
                if(all||now_ns()>s->host+400'000'000) {selected=s;present=s->present;break;}
            }
            if(selected)for(auto& s:history)if(s->sequence<=selected->sequence)s->emitted=true;
        }
        if(selected)emit({{"event","preview"},{"preview",selected->preview},{"sequence",selected->sequence},
            {"scheduled_ns",selected->ptp},{"present",present},{"ptp",ptp_state}});
    }
public:
    explicit Service(const Json& configuration):config(configuration),rig(configuration),
        preview(configuration.at("preview_path"),int(rig.info.size()),1440,1080),offset(rig.clock_offset()),
        preview_fps(configuration.value("preview_fps",5.)) {
        require(preview_fps>=.2&&preview_fps<=10,"Preview rate must be 0.2–10 Hz");
        ptp_state=rig.timing_state();
        try {
            for(size_t c=0;c<rig.info.size();++c)receivers.emplace_back([this,c]{if(rig.simulated)simulate(int(c));else receive(int(c));});
            scheduler=std::thread([this]{schedule();});
            Json info=Json::array();for(auto& c:rig.info)info.push_back(c.json());
            emit({{"event","ready"},{"cameras",info},{"simulated",rig.simulated},{"ptp",ptp_state},{"protocol",1}});
        } catch(...) {
            stopping=true;history_changed.notify_all();
            if(scheduler.joinable())scheduler.join();
            for(auto& receiver:receivers)if(receiver.joinable())receiver.join();
            throw;
        }
    }
    void dispatch(Json request) {
        auto command=request.at("command").get<std::string>();
        if(command=="stop") {auto take=current();if(take&&!take->done()){if(!take->stop_requested)take->stop_kind=1;take->stop_requested=true;}}
        else if(command=="record") {
            require(!stopping,"Capture service is stopping");
            auto old=current();require(!old||old->done(),"A take is already active");
            int fps=request.at("fps"),frames=request.at("frames"),quality=request.value("quality",85);
            require(fps>=1&&fps<=60&&frames>=1&&frames<=fps*120&&quality>=50&&quality<=95,"Invalid take settings");
            emit({{"event","recording"},{"status",{{"state","arming"},{"directory",request.at("directory")},{"fps",fps},{"planned_frames",frames}}}});
            request["ptp_before"]=rig.timing_state(true,fps);request["simulated"]=rig.simulated;
            if(rig.simulated&&config.contains("test_fail_write_after"))request["test_fail_write_after"]=config["test_fail_write_after"];
            offset=rig.clock_offset();
            auto take=std::make_shared<Take>(request.at("directory").get<std::string>(),rig.info,request);
            {std::lock_guard<std::mutex> lock(take_mutex);active=take;}
        }else throw std::runtime_error("Unknown recorder command");
    }
    void poll() {
        publish_preview();auto take=current();
        if(take)take->poll();
        if(now_ns()-last_status>250'000'000) {
            Json status=take?take->snapshot():Json{{"state","idle"}};
            status["unexpected_frames"]=unexpected.load();status["incomplete_images"]=incomplete.load();
            emit({{"event","recording"},{"status",status}});last_status=now_ns();
        }
        // Camera settings stay frozen during a take. Control-channel reads stay
        // off the 60 Hz scheduling and receive loops.
        if((!take||take->done())&&now_ns()-last_clock>5'000'000'000LL) {
            try{ptp_state=rig.timing_state();offset=rig.clock_offset();}catch(const std::exception& e){fault(e.what());}
            last_clock=now_ns();
        }
        require(!stopping,"Capture scheduler stopped");
    }
    ~Service() {
        auto take=current();
        if(take&&!take->done()) {
            if(!take->stop_requested)take->stop_kind=2;
            take->stop_requested=true;
            auto until=now_ns()+8'000'000'000LL;
            while(!take->done()&&now_ns()<until){take->poll();std::this_thread::sleep_for(std::chrono::milliseconds(10));}
        }
        stopping=true;history_changed.notify_all();
        if(scheduler.joinable())scheduler.join();
        for(auto& r:receivers)if(r.joinable())r.join();
        history.clear();active.reset();take.reset();
        rig.close();
    }
};

int main(int argc,char** argv) {
    std::signal(SIGTERM,interrupt_handler);std::signal(SIGINT,interrupt_handler);std::signal(SIGPIPE,SIG_IGN);
    av_log_set_level(AV_LOG_ERROR);
    try {
        require(argc==2,"usage: multical-recorder CONFIG.json");
        std::ifstream input(argv[1]);require(bool(input),"Cannot open recorder configuration");
        Json config;input>>config;
        Service service(config);
        std::string pending;
        while(!interrupted) {
            pollfd descriptor{STDIN_FILENO,POLLIN,0};int result=::poll(&descriptor,1,20);
            if(result>0&&(descriptor.revents&(POLLIN|POLLHUP))) {
                char bytes[4096];auto n=::read(STDIN_FILENO,bytes,sizeof(bytes));
                if(n==0)break;
                if(n<0){if(errno==EINTR)continue;throw std::runtime_error("Control pipe read failed");}
                pending.append(bytes,n);require(pending.size()<65536,"Oversized control command");
                size_t end;
                while((end=pending.find('\n'))!=std::string::npos) {
                    auto line=pending.substr(0,end);pending.erase(0,end+1);
                    try {
                        auto request=Json::parse(line);
                        if(request.at("command")=="shutdown"){interrupted=1;break;}
                        service.dispatch(request);
                    }catch(const std::exception& e){emit({{"event","command_error"},{"error",e.what()}});}
                }
            }
            service.poll();
        }
        return 0;
    }catch(const std::exception& e){emit({{"event","fatal"},{"error",e.what()}});return 1;}
}
