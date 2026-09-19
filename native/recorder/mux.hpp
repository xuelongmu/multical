#pragma once
#include "common.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
}
namespace recorder {
inline void av_check(int code, const std::string& operation) {
    if(code>=0)return;
    char message[AV_ERROR_MAX_STRING_SIZE]; av_strerror(code,message,sizeof(message));
    throw std::runtime_error(operation + ": " + message);
}
class SegmentMuxer {
    AVFormatContext* format=nullptr;
    AVStream* stream=nullptr;
    fs::path path;
    int fps;
public:
    SegmentMuxer(const fs::path& file, int width, int height, int rate) : path(file),fps(rate) {
        require(!fs::exists(path),"Refusing to overwrite segment");
        try {
            av_check(avformat_alloc_output_context2(&format,nullptr,"matroska",path.c_str()),"Create MKV");
            stream=avformat_new_stream(format,nullptr);require(stream,"Create video stream");
            stream->time_base={1,fps};stream->avg_frame_rate={fps,1};
            auto c=stream->codecpar;c->codec_type=AVMEDIA_TYPE_VIDEO;c->codec_id=AV_CODEC_ID_MJPEG;
            c->width=width;c->height=height;c->format=AV_PIX_FMT_YUVJ420P;
            c->color_range=AVCOL_RANGE_JPEG;c->color_space=AVCOL_SPC_BT470BG;
            av_dict_set(&format->metadata,"encoder","Multical Live / nvJPEG",0);
            av_check(avio_open(&format->pb,path.c_str(),AVIO_FLAG_WRITE),"Open video segment");
            av_check(avformat_write_header(format,nullptr),"Write MKV header");
        } catch (...) { abort();throw; }
    }
    ~SegmentMuxer() { abort(); }
    void abort() { if(format){ if(format->pb)avio_closep(&format->pb);avformat_free_context(format);format=nullptr; } }
    void write(const std::vector<uint8_t>& bytes, int relative_frame) {
        AVPacket packet{};
        packet.data=const_cast<uint8_t*>(bytes.data());packet.size=int(bytes.size());
        packet.stream_index=stream->index;packet.pts=packet.dts=relative_frame;packet.duration=1;packet.flags=AV_PKT_FLAG_KEY;
        av_packet_rescale_ts(&packet,{1,fps},stream->time_base);
        av_check(av_write_frame(format,&packet),"Write JPEG packet");
        if(format->pb->error<0)av_check(format->pb->error,"Video storage error");
    }
    void finish() {
        if(!format)return;
        av_check(av_write_trailer(format),"Finalize MKV");avio_flush(format->pb);
        av_check(format->pb->error,"Flush MKV");av_check(avio_closep(&format->pb),"Close MKV");
        avformat_free_context(format);format=nullptr;
        sync_path(path);
    }
};
} // namespace recorder
