#include <chrono>
#include <filesystem>
#include <iostream>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <vector>

#include "ui.h"
#include "audio.h"
#include "asr.h"
#include "llm.h"

int save_data(const std::vector<std::string>& files) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm * now_tm = std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y%m%d%H%M");
    std::string path = "data/" + oss.str();
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    for (const auto& file: files) {
        std::string filename = std::filesystem::path(file).filename().string();
        std::string new_file = path + "/" + filename;
        std::filesystem::rename(file, new_file);
    }

    return 0;
}

int main(int argc, char* argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("config,c", 
            po::value<std::string>()->default_value("config/config.json"), 
            "set configuration file");
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    } catch (const po::error& e) {
        std::cerr << "Error parsing command line options: " << e.what() << std::endl;
        return 1;
    }
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::string configFile = vm["config"].as<std::string>();
    nlohmann::json config;
    try {
        std::ifstream f(configFile);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open configuration file: " + configFile);
        }
        f >> config;
    } catch (const std::exception& e) {
        std::cerr << "Error reading configuration file: " << e.what() << std::endl;
        return 1;
    }

    Audio& audio = Audio::instance();
    int ret = audio.init(config["audio"]);
    if (ret != 0) {
        std::cerr << "Audio initialization failed with error code: " << ret << std::endl;
        return ret;
    }
    std::cout << "Audio initialized successfully." << std::endl;

    ASR& asr = ASR::instance();
    ret = asr.init(config["asr"]);
    if (ret != 0) {
        std::cerr << "ASR initialization failed with error code: " << ret << std::endl;
        audio.shutdown();
        return ret;
    }
    std::cout << "ASR initialized successfully." << std::endl;

    LLM& llm = LLM::instance();
    ret = llm.init(config["llm"], [](const std::string& name, 
        const std::string& result) {
        //std::cout << "LLM callback: " << name << " - " << result << std::endl;
        EchoNote::UI::instance().show(name, result);
    });

    asr.setAudio(&audio, [](const std::string& result) {
        EchoNote::UI::instance().show("asr", result);
        LLM::instance().refine(result);        
    });
    std::cout << "ASR set audio successfully." << std::endl;

    EchoNote::UI::instance().show(config["ui"], &audio, &llm);

    llm.shutdown();
    std::cout << "LLM shutdown successfully." << std::endl;
    asr.shutdown();
    std::cout << "ASR shutdown successfully." << std::endl;
    audio.shutdown();
    std::cout << "Audio shutdown successfully." << std::endl;

    std::vector<std::string> files = {
        audio.getOutFile(),
        asr.getOutFile(),
        llm.getRefineOutFile(),
        llm.getSummarizeOutFile()
    };
    if (std::filesystem::file_size(audio.getOutFile())) {
        save_data(files);
        std::cout << "Data saved successfully." << std::endl;
    }

    std::cout << "main exit!" << std::endl;
    return 0;
}