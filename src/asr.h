#pragma once

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "audio.h"

typedef void (* asr_callback)(const std::string&);
class ASR {
public:
    static ASR& instance() {
        static ASR _inst;
        return _inst;
    }
    ASR(const ASR&) = delete;
    ASR operator =(const ASR&) = delete;

    int init(const nlohmann::json& config);
    int shutdown();
    int setAudio(const Audio* audio, asr_callback func);

    std::string getOutFile() const {
        return asr_out_path; // Return the path to the ASR output file
    }

private:
    ASR() = default;
    ~ASR() = default;

    typedef struct _model_config_t {
        std::string window_type = "hamming";
        int frame_length = 25;
        int frame_shift = 10;
        int n_mels = 80;
        int lfr_m = 7;
        int lfr_n = 6;
        int asr_sample_rate = 16000;

        int encoder_size = 512;
        int fsmn_dims = 512;
    } model_config_t;
    
    model_config_t model_config;
    std::vector<float> means_list;
    std::vector<float> vars_list;
    std::vector<std::string> vocab;

    int chunk_time = 2000;
    int overlap_time = 800;

    int load_config(const std::string& path);
    int load_mvn(const std::string& path);
    int load_tokens(const std::string& path);

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    std::thread asrThread;
    bool asr_running = false;

    int extract_features(const std::vector<float>& data, 
        std::vector<std::vector<float>>& features);
    int normalize_features(std::vector<std::vector<float>>& features);
    std::string ctc_search(const float * data, 
        const std::vector<int>& speech_length, const std::vector<int64_t>& data_shape);
    std::string asr(const std::vector<float>& data);

    bool save = false;
    std::string asr_out_path = "output/asr.txt"; // Path to save ASR results
    std::ofstream out;
};
