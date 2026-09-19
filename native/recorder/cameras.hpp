#pragma once
#include "common.hpp"
#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>
#include <iomanip>
#include <sstream>

namespace recorder {
using namespace Spinnaker;
using namespace Spinnaker::GenApi;
inline Json node_get(INodeMap& nm,const std::string& name,const std::string& kind) {
    auto node=nm.GetNode(name.c_str());require(IsReadable(node),"Unreadable camera node " + name);
    if(kind=="enum")return std::string(CEnumerationPtr(node)->ToString().c_str());
    if(kind=="int")return CIntegerPtr(node)->GetValue();
    if(kind=="float")return CFloatPtr(node)->GetValue();
    if(kind=="bool")return CBooleanPtr(node)->GetValue();
    if(kind=="string")return std::string(CStringPtr(node)->GetValue().c_str());
    throw std::runtime_error("Unknown GenICam node kind");
}
inline void node_set(INodeMap& nm,const std::string& name,const std::string& kind,const Json& value) {
    auto node=nm.GetNode(name.c_str());require(IsWritable(node),"Unwritable camera node " + name);
    if(kind=="enum") {
        CEnumerationPtr p(node);auto entry=p->GetEntryByName(value.get<std::string>().c_str());
        require(IsReadable(entry),"Unsupported camera enum " + name);p->SetIntValue(entry->GetValue());
    } else if(kind=="int")CIntegerPtr(node)->SetValue(value.get<int64_t>());
    else if(kind=="float")CFloatPtr(node)->SetValue(value.get<double>());
    else if(kind=="bool")CBooleanPtr(node)->SetValue(value.get<bool>());
    else throw std::runtime_error("Unsupported writable camera node kind");
}
inline void command(INodeMap& nm,const char* name) {
    CCommandPtr p(nm.GetNode(name));require(IsWritable(p),std::string("Unavailable camera command ")+name);p->Execute();
}
struct HardwareCamera {
    CameraPtr camera;
    CameraInfo info;
    bool initialized=false,acquiring=false;
    Json restore=Json::array();
    std::string old_selector,old_trigger;
    uint64_t last_frame=0;
    explicit HardwareCamera(CameraPtr p):camera(p){}
    void change(const std::string& name,const std::string& kind,const Json& value,bool transport=false,bool write_only=false) {
        auto& nm=transport?camera->GetTLStreamNodeMap():camera->GetNodeMap();
        if(!write_only||IsReadable(nm.GetNode(name.c_str()))) {
            auto previous=node_get(nm,name,kind);
            if(previous==value)return;
            restore.push_back({{"name",name},{"kind",kind},{"value",previous},{"transport",transport}});
        }
        node_set(nm,name,kind,value);
    }
    void configure(std::optional<double> exposure,std::optional<double> gain) {
        camera->Init();initialized=true;auto& nm=camera->GetNodeMap();
        info.serial=node_get(camera->GetTLDeviceNodeMap(),"DeviceSerialNumber","string");
        auto mac=node_get(camera->GetTLDeviceNodeMap(),"GevDeviceMACAddress","int").get<uint64_t>();
        std::ostringstream out;out<<std::hex<<std::setfill('0');
        for(int b=5;b>=0;--b){if(b!=5)out<<':';out<<std::setw(2)<<((mac>>(b*8))&255);}info.mac=out.str();
        old_selector=node_get(nm,"TriggerSelector","enum");node_set(nm,"TriggerSelector","enum","FrameStart");
        old_trigger=node_get(nm,"TriggerMode","enum");node_set(nm,"TriggerMode","enum","Off");
        change("AcquisitionMode","enum","Continuous");
        change("ExposureAuto","enum","Off");change("GainAuto","enum","Off");
        change("BalanceWhiteAuto","enum","Off");change("AcquisitionFrameRateEnable","bool",false);
        change("PixelFormat","enum","BayerRG8");
        change("TriggerSource","enum","Action0");change("TriggerOverlap","enum","ReadOut");
        change("ActionUnconditionalMode","enum","Off");change("ActionDeviceKey","int",42,false,true);
        change("ActionGroupKey","int",1);change("ActionGroupMask","int",0xffffffffLL);
        // Match CaptureNet's measured fleet transport profile; do not persist a user set.
        CIntegerPtr limit(nm.GetNode("DeviceLinkThroughputLimit"));
        change("DeviceLinkThroughputLimit","int",std::min<int64_t>(125000000,limit->GetMax()));
        change("GevSCPD","int",0);
        if(exposure)change("ExposureTime","float",*exposure);
        if(gain)change("Gain","float",*gain);
        info.width=node_get(nm,"Width","int");info.height=node_get(nm,"Height","int");
        require(info.width==1440&&info.height==1080,"Recorder currently requires native 1440x1080 images");
        require(node_get(nm,"OffsetX","int")==0&&node_get(nm,"OffsetY","int")==0,
                "Recorder requires zero ROI offsets for calibrated geometry and Bayer phase");
        require(!node_get(nm,"ReverseX","bool").get<bool>()&&!node_get(nm,"ReverseY","bool").get<bool>(),
                "Recorder requires unmirrored sensor images");
        info.exposure_us=node_get(nm,"ExposureTime","float");info.gain_db=node_get(nm,"Gain","float");
        info.settings={{"pixel_format","BayerRG8"},{"offset_x",0},{"offset_y",0},{"reverse_x",false},{"reverse_y",false},
            {"action_device_key",42},{"exposure_auto","Off"},{"gain_auto","Off"},
            {"adc_bit_depth",node_get(nm,"AdcBitDepth","enum")},{"isp_enabled",node_get(nm,"IspEnable","bool")},
            {"gamma_enabled",node_get(nm,"GammaEnable","bool")},
            {"link_throughput_limit",node_get(nm,"DeviceLinkThroughputLimit","int")},
            {"packet_size",node_get(nm,"GevSCPSPacketSize","int")},{"inter_packet_delay",0}};
        change("StreamBufferCountMode","enum","Manual",true);
        change("StreamBufferCountManual","int",64,true);
        change("StreamBufferHandlingMode","enum","OldestFirst",true);
        node_set(nm,"TriggerMode","enum","On");
        camera->BeginAcquisition();acquiring=true;
    }
    Json ptp() {
        auto& nm=camera->GetNodeMap();command(nm,"GevIEEE1588DataSetLatch");
        return {{"status",node_get(nm,"GevIEEE1588StatusLatched","enum")},
                {"offset_ns",node_get(nm,"GevIEEE1588OffsetFromMasterLatched","int")}};
    }
    void close() {
        if(!initialized)return;
        auto attempt=[&](auto fn){try{fn();}catch(const std::exception& e){emit({{"event","restore_warning"},{"serial",info.serial},{"error",e.what()}});}};
        if(acquiring){attempt([&]{camera->EndAcquisition();});acquiring=false;}
        auto& nm=camera->GetNodeMap();
        attempt([&]{node_set(nm,"TriggerMode","enum","Off");});
        for(auto it=restore.rbegin();it!=restore.rend();++it)attempt([&]{
            auto& map=it->at("transport").get<bool>()?camera->GetTLStreamNodeMap():nm;
            node_set(map,it->at("name"),it->at("kind"),it->at("value"));});
        if(!old_trigger.empty())attempt([&]{node_set(nm,"TriggerMode","enum",old_trigger);});
        if(!old_selector.empty())attempt([&]{node_set(nm,"TriggerSelector","enum",old_selector);});
        attempt([&]{camera->DeInit();});initialized=false;
    }
    ~HardwareCamera(){close();}
};

class CameraRig {
    SystemPtr system;
    CameraList devices;
    InterfaceList interfaces;
    std::vector<InterfacePtr> action_interfaces;
public:
    bool simulated=false;
    std::vector<std::unique_ptr<HardwareCamera>> cameras;
    std::vector<CameraInfo> info;
    CameraRig(const Json& configuration) {
        simulated=configuration.value("simulate",false);
        int expected=configuration.value("count",25);
        require(expected>=1&&expected<=25,"Recorder supports 1–25 cameras");
        if(simulated) {
            for(int i=0;i<expected;++i){CameraInfo c;char s[32];std::snprintf(s,sizeof(s),"SIM-%02d",i+1);c.serial=s;
                c.settings={{"pixel_format","BayerRG8"},{"offset_x",0},{"offset_y",0},{"reverse_x",false},{"reverse_y",false},{"simulated",true}};info.push_back(c);}
            return;
        }
        try {
            system=System::GetInstance();devices=system->GetCameras();interfaces=system->GetInterfaces();
            std::map<std::string,unsigned> discovered;
            for(unsigned i=0;i<devices.GetSize();++i) {
                CameraPtr cam=devices.GetByIndex(i);discovered[node_get(cam->GetTLDeviceNodeMap(),"DeviceSerialNumber","string").get<std::string>()]=i;
            }
            std::vector<std::string> serials=configuration.value("serials",std::vector<std::string>{});
            if(serials.empty())for(auto& entry:discovered)serials.push_back(entry.first);
            require(int(serials.size())==expected,"Discovered camera count does not match requested rig");
            auto unique=serials;std::sort(unique.begin(),unique.end());
            require(std::adjacent_find(unique.begin(),unique.end())==unique.end(),"Duplicate camera serial");
            for(unsigned i=0;i<interfaces.GetSize();++i){auto iface=interfaces.GetByIndex(i);auto list=iface->GetCameras();if(list.GetSize())action_interfaces.push_back(iface);list.Clear();}
            for(const auto& serial:serials) {
                require(discovered.count(serial),"Missing camera " + serial);
                emit({{"event","connecting"},{"serial",serial},{"connected",info.size()},{"total",expected}});
                cameras.push_back(std::make_unique<HardwareCamera>(devices.GetByIndex(discovered.at(serial))));
                std::optional<double> exposure,gain;
                if(configuration.contains("exposure_us")&&!configuration["exposure_us"].is_null())exposure=configuration["exposure_us"];
                if(configuration.contains("gain_db")&&!configuration["gain_db"].is_null())gain=configuration["gain_db"];
                cameras.back()->configure(exposure,gain);info.push_back(cameras.back()->info);
            }
            require(!action_interfaces.empty(),"No populated GigE interfaces");
        }catch(...){close();throw;}
    }
    Json timing_state(bool recording=false,int fps=60) {
        Json result=Json::object();
        for(size_t c=0;c<info.size();++c) {
            auto state=simulated?Json{{"status","SIMULATED"},{"offset_ns",0}}:cameras[c]->ptp();
            if(recording&&!simulated) {
                require(state["status"]=="Slave","Camera " + info[c].serial + " is not PTP Slave");
                require(std::llabs(state["offset_ns"].get<int64_t>())<=20000,"PTP offset exceeds 20 microseconds on " + info[c].serial);
            }
            if(recording)require(info[c].exposure_us<1e6/fps,"Exposure exceeds recording frame interval on " + info[c].serial);
            result[info[c].serial]=state;
        }
        return result;
    }
    int64_t clock_offset() {
        if(simulated)return 1'700'000'000'000'000'000LL;
        int64_t best=INT64_MAX,offset=0;
        for(int i=0;i<5;++i){auto before=now_ns();auto& nm=cameras[0]->camera->GetNodeMap();
            command(nm,"TimestampLatch");int64_t ptp=node_get(nm,"TimestampLatchValue","int");auto after=now_ns();
            if(after-before<best){best=after-before;offset=ptp-(before+after)/2;}}
        return offset;
    }
    void action(uint64_t timestamp) {
        for(auto& iface:action_interfaces)iface->SendActionCommand(42,1,0xffffffff,timestamp,false,nullptr,nullptr);
    }
    void close() {
        cameras.clear();action_interfaces.clear();devices.Clear();interfaces.Clear();
        if(system){system->ReleaseInstance();system=nullptr;}
    }
    ~CameraRig(){close();}
};
} // namespace recorder
