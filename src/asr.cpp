#include <fstream>
#include <iostream>
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

#include "asr.h"
#include "audio.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"

int ASR::init(const nlohmann::json& config) {
    const std::string model_path = config.value("model_path", "models/SenseVoiceSmall");
    const std::string config_file = model_path + "/config.yaml";
    const std::string mvn_file = model_path + "/am.mvn";
    const std::string tokens_file = model_path + "/tokens.json";
    const std::string model_file = model_path + "/model_quant.onnx";

    if (load_config(config_file) != 0) {
        std::cerr << "Failed to load config from " << config_file << std::endl;
        return -1; // Return -1 on failure
    }
    if (load_mvn(mvn_file) != 0) {
        std::cerr << "Failed to load MVN from " << mvn_file << std::endl;
        return -1; // Return -1 on failure
    }
    if (load_tokens(tokens_file) != 0) {
        std::cerr << "Failed to load tokens from " << tokens_file << std::endl;
        return -1; // Return -1 on failure
    }

    chunk_time = config.value("chunk_time", 2000);
    overlap_time = config.value("overlap_time", 800);
    if (overlap_time > chunk_time) overlap_time = chunk_time;

    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "echonote-asr");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(4);
    session = std::make_unique<Ort::Session>(
        env, model_file.c_str(), so
    );

    save = config.value("save", false);
    asr_out_path = config.value("output", 
        "output/asr.txt");
    if (save) {
        out = std::ofstream(asr_out_path);
    }

    return 0; // Return 0 on success
}

int ASR::shutdown() {
    // Shutdown logic here
    asr_running = false; // Stop the ASR thread if it's running
    if (asrThread.joinable()) {
        asrThread.join(); // Wait for the ASR thread to finish
    }

    if (session) {
        session->release();
    }

    if (save) out.close();

    return 0; // Return 0 on success
}

int ASR::setAudio(const Audio* audio, asr_callback func) {
    if (!audio) {
        std::cerr << "Invalid audio pointer." << std::endl;
        return -1; // Return -1 on failure
    }

    if (asr_running) {
        std::cerr << "ASR is already running." << std::endl;
        return -1; // Return -1 if ASR is already running
    }

    asr_running = true;
    asrThread = std::thread([this, audio, func]() {
        std::vector<float> audio_data;
        const int min_chunk_length = chunk_time * model_config.asr_sample_rate / 1000;
        const int overlap_chunk_length = overlap_time * model_config.asr_sample_rate / 1000;

        while (asr_running) {
            std::vector<float> chunk = 
                const_cast<Audio *>(audio)->readAudio(chunk_time);
            if (chunk.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue; // Skip if no audio data is available
            }
            audio_data.insert(audio_data.end(), chunk.begin(), chunk.end());
            if (audio_data.size() >= min_chunk_length) {
                std::string result = asr(audio_data); // Process ASR with the accumulated audio data
                if (func) func(result);
                if (save) out << result;
                audio_data.erase(audio_data.begin(), 
                    audio_data.begin() + audio_data.size() - overlap_chunk_length);
            }
        }

        return 0; // Return 0 on success
    });

    return 0; // Return 0 on success
}

int ASR::load_config(const std::string& path) {
    try {
        YAML::Node config = YAML::LoadFile(path);
        YAML::Node frontend_conf = config["frontend_conf"];
        model_config.window_type = frontend_conf["window"].as<std::string>();
        model_config.frame_length = frontend_conf["frame_length"].as<int>();
        model_config.frame_shift = frontend_conf["frame_shift"].as<int>();
        model_config.n_mels = frontend_conf["n_mels"].as<int>();
        model_config.lfr_m = frontend_conf["lfr_m"].as<int>();
        model_config.lfr_n = frontend_conf["lfr_n"].as<int>();
        model_config.asr_sample_rate = frontend_conf["fs"].as<int>();

        YAML::Node encoder_conf = config["encoder_conf"];
        model_config.encoder_size = encoder_conf["output_size"].as<int>();
        model_config.fsmn_dims = encoder_conf["output_size"].as<int>();
    } catch (const YAML::Exception& e) {
        std::cout << "Error loading YAML file: " << e.what() << std::endl;
        return -1; // Return -1 on failure
    }

    return 0;
}

int ASR::load_mvn(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "Failed to open file: " << path << std::endl;
        return -1; // Return -1 on failure
    }

    for (std::string line; std::getline(f, line);) {
        if (line.empty()) continue; // Skip empty lines
        std::istringstream iss(line);
        std::vector<std::string> items{
            std::istream_iterator<std::string>{iss},
            std::istream_iterator<std::string>{}
        };
        if (items[0] == "<AddShift>") {
            std::getline(f, line);
            std::istringstream iss_means(line);
            std::vector<std::string> means{
                std::istream_iterator<std::string>{iss_means},
                std::istream_iterator<std::string>{}
            };
            if (means[0] == "<LearnRateCoef>") {
                for (int i=3; i < means.size() - 1; ++i) {
                    means_list.push_back(std::stof(means[i]));
                }
            }
        }

        if (items[0] == "<Rescale>") {
            std::getline(f, line);
            std::istringstream iss_vars(line);
            std::vector<std::string> vars{
                std::istream_iterator<std::string>{iss_vars},
                std::istream_iterator<std::string>{}
            };
            if (vars[0] == "<LearnRateCoef>") {
                for (int i=3; i < vars.size() - 1; ++i) {
                    vars_list.push_back(std::stof(vars[i]));
                }
            }
        }
    }

    return 0;
}

int ASR::load_tokens(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "Failed to open file: " << path << std::endl;
        return -1; // Return -1 on failure
    }
    nlohmann::json tokens_json;
    try {
        f >> tokens_json;
    } catch (const nlohmann::json::parse_error& e) {
        std::cout << "Error parsing JSON file: " << e.what() << std::endl;
        return -1; // Return -1 on failure
    }

    if (!tokens_json.is_array()) {
        std::cout << "Tokens file is not an array." << std::endl;
        return -1; // Return -1 on failure
    }

    for (const auto& token : tokens_json) {
        if (token.is_string()) {
            vocab.push_back(token.get<std::string>());
        } else {
            std::cout << "Invalid token format in JSON file." << std::endl;
            return -1; // Return -1 on failure
        }
    }
    return 0;
}

int ASR::extract_features(const std::vector<float>& data, 
    std::vector<std::vector<float>>& features) {
    std::vector<float> buf(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        buf[i] = data[i] * 32768;
    }

    knf::FbankOptions fbank_opts;
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.frame_opts.window_type = model_config.window_type;
    fbank_opts.frame_opts.frame_length_ms = model_config.frame_length;
    fbank_opts.frame_opts.frame_shift_ms = model_config.frame_shift;
    fbank_opts.frame_opts.samp_freq = model_config.asr_sample_rate;
    fbank_opts.mel_opts.num_bins = model_config.n_mels;

    knf::OnlineFbank fbank(fbank_opts);
    fbank.AcceptWaveform(model_config.asr_sample_rate, 
        buf.data(), buf.size());

    int num_frames = fbank.NumFramesReady();
    for (int i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        std::vector<float> feature(frame, frame + fbank.Dim());
        features.push_back(feature);
    }

    return 0;
}

int ASR::normalize_features(std::vector<std::vector<float>>& features) {
    std::vector<std::vector<float>> out_feats;
    int T = features.size();
    int T_lrf = ceil(1.0 * T / model_config.lfr_n);

    // Pad frames at start(copy first frame)
    for (int i = 0; i < (model_config.lfr_m - 1) / 2; i++) {
        features.insert(features.begin(), features[0]);
    }
    // Merge lfr_m frames as one,lfr_n frames per window
    T = T + (model_config.lfr_m - 1) / 2;
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (model_config.lfr_m <= T - i * model_config.lfr_n) {
            for (int j = 0; j < model_config.lfr_m; j++) {
                p.insert(p.end(), 
                    features[i * model_config.lfr_n + j].begin(), 
                    features[i * model_config.lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
            int num_padding = model_config.lfr_m - (T - i * model_config.lfr_n);
            for (int j = 0; j < (features.size() - i * model_config.lfr_n); j++) {
                p.insert(p.end(), 
                    features[i * model_config.lfr_n + j].begin(), 
                    features[i * model_config.lfr_n + j].end());
            }
            for (int j = 0; j < num_padding; j++) {
                p.insert(p.end(), 
                    features[features.size() - 1].begin(), 
                    features[features.size() - 1].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        }
    }
    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list[j]) * vars_list[j];
        }
    }
    features = out_feats;

    return 0;
}

std::string ASR::ctc_search(const float * data, 
    const std::vector<int>& speech_length, const std::vector<int64_t>& data_shape) {
    std::string unicodeChar = "▁";
    int32_t vocab_size = data_shape[2];

    std::vector<int64_t> tokens;
    const int blank_id = 0;
    std::string text="";
    int32_t prev_id = -1;
    for (int32_t t = 0; t != speech_length[0]; ++t) {
        auto y = std::distance(
            data,
            std::max_element(
                data,
                data + vocab_size));
        data += vocab_size;

        if (y != blank_id && y != prev_id) {
            tokens.push_back(y);
        }
        prev_id = y;
    }
    std::string str_lang = "";
    std::string str_emo = "";
    std::string str_event = "";
    std::string str_itn = "";
    if(tokens.size() >=3){
        str_lang  = vocab[tokens[0]];
        str_emo   = vocab[tokens[1]];
        str_event = vocab[tokens[2]];
        str_itn = vocab[tokens[3]];
    }

    for(int32_t i = 4; i < tokens.size(); ++i){
        std::string word = vocab[tokens[i]];
        size_t found = word.find(unicodeChar);
        if(found != std::string::npos){
            text += " " + word.substr(3);
        }else{
            text += word;
        }
    }
    if(str_itn == "<|withitn|>"){
        if(str_lang == "<|zh|>"){
//            text += "。";
        }else{
            text += ".";
        }
    }

    return str_lang + str_emo + str_event + " " + text;
}

std::string ASR::asr(const std::vector<float>& data) {
    //std::cout << "Processing ASR for data size: " << data.size() << std::endl;
    std::vector<std::vector<float>> features;
    if (extract_features(data, features) != 0) {
        std::cerr << "Failed to extract features." << std::endl;
        return "";
    }
    if (normalize_features(features) != 0) {
        std::cerr << "Failed to normalize features." << std::endl;
        return "";
    }
    if (features.empty()) {
        std::cerr << "No features extracted." << std::endl;
        return "";
    }

    int num_frames = features.size();
    int num_features = features[0].size();

    std::vector<float> speech_data;
    for (const auto& frame : features) {
        speech_data.insert(speech_data.end(), frame.begin(), frame.end());
    }

    const std::vector<int64_t> input_speech_shape{1, num_frames, num_features};
    Ort::Value input_speech = Ort::Value::CreateTensor<float>(memoryInfo, 
        speech_data.data(), speech_data.size(), 
        input_speech_shape.data(), input_speech_shape.size());
    const std::vector<int64_t> input_speech_length_shape{1};
    std::vector<int> speech_length{num_frames};
    Ort::Value input_speech_length = Ort::Value::CreateTensor<int>(memoryInfo, 
        speech_length.data(), speech_length.size(), 
        input_speech_length_shape.data(), input_speech_length_shape.size());
    const std::vector<int64_t> input_language_shape{1};
    std::vector<int> language{0};
    Ort::Value input_language = Ort::Value::CreateTensor<int>(memoryInfo, 
        language.data(), language.size(), 
        input_language_shape.data(), input_language_shape.size());
    const std::vector<int64_t> input_textnorm_shape{1};
    std::vector<int> textnorm{14};
    Ort::Value input_textnorm = Ort::Value::CreateTensor<int>(memoryInfo, 
        textnorm.data(), textnorm.size(), 
        input_textnorm_shape.data(), input_textnorm_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input_speech));
    inputs.emplace_back(std::move(input_speech_length));
    inputs.emplace_back(std::move(input_language));
    inputs.emplace_back(std::move(input_textnorm));
    
    const std::vector<const char *> input_names {
        "speech", "speech_lengths", "language", "textnorm"
    };
    const std::vector<const char *> output_names {
        "ctc_logits", "encoder_out_lens"
    };

    Ort::RunOptions options(nullptr);
    auto outputs = session->Run(options, input_names.data(), 
        inputs.data(), inputs.size(), 
        output_names.data(), output_names.size());
    auto ctc_logits = outputs[0].GetTensorData<float>();
    auto ctc_logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    std::string result = ctc_search(ctc_logits, 
        {num_frames}, ctc_logits_shape);
    return result;
}
