#include "llm.h"
#include "openai.h"
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <thread>

#include "ui.h"

int LLM::init(const nlohmann::json& config, llm_callback func) {
    std::string schema_host_port = config.value("schema_host_port", 
        "http://localhost:8080");
    openai::start(schema_host_port);

    model = config.value("model", "Qwen3-8b");
    temperature = config.value("temperature", 0.6f);
    top_p = config.value("top_p", 0.95f);
    top_k = config.value("top_k", 20);
    presence_penalty = config.value("presence_penalty", 1.5f);

    auto load_system_prompt = [](const std::string& path) {
        std::string prompt = "";
        std::ifstream file(path);
        if (file.is_open()) {
            prompt.assign(std::istreambuf_iterator<char>(file), 
                std::istreambuf_iterator<char>());
            file.close();
        }
        return prompt;
    };

    nlohmann::json refine_config = config["refine"];
    refine_system_prompt = load_system_prompt(
        refine_config.value("system_prompt", "res/prompt/refine.txt")
    );
    refine_chunk_size = refine_config.value("chunk_size", 1024);
    refine_save = refine_config.value("save", false);
    refine_span = refine_config.value("refine_span", 120);
    refine_output_path = refine_config.value("output", "output/refine.txt");
    if (refine_save) refine_output_file.open(refine_output_path, std::ios::out | std::ios::trunc);

    nlohmann::json summarize_config = config["summarize"];
    summarize_system_prompt = load_system_prompt(
        summarize_config.value("system_prompt", "res/prompt/summarize.txt")
    );
    summarize_save = summarize_config.value("save", false);
    summarize_output_path = summarize_config.value("output", "output/summarize.txt");
    if (summarize_save) summarize_output_file.open(summarize_output_path, std::ios::out | std::ios::trunc);

    auto make_request = [this](const std::string& text, 
        const std::string& system_prompt) {
        nlohmann::json request;
        request["model"] = model;
        request["temperature"] = temperature;
        request["top_p"] = top_p;
        request["top_k"] = top_k;
        request["presence_penalty"] = presence_penalty;
        request["messages"] = {
            {{"role", "system"},
             {"content", system_prompt}},
            {{"role", "user"},
             {"content", text}}
        };
        EchoNote::UI::log(request.dump());
        return request;
    };

    auto llm_predict = [this, make_request](const std::string& text,
        const std::string& system_prompt) {
        nlohmann::json request = make_request(text, system_prompt);
        std::string content = "";
        try {
            auto response = openai::chat().create(request);
            //std::cout << "LLM response: " << response.dump() << std::endl;
            EchoNote::UI::log(response.dump());
            if (response.is_null() || !response.contains("choices")) {
                return std::string();
            }
            auto choices = response["choices"];
            if (choices.empty()) {
                return std::string();
            }
            content = choices[0]["message"]["content"].get<std::string>();
            const std::string postfix_think = "</think>\n\n";
            int pos = content.find(postfix_think);
            if (pos != std::string::npos) {
                content = content.substr(pos + postfix_think.size());
            }
        } catch (const std::exception& e) {
            std::cout << "Error chat stream: " << e.what() << std::endl;
        }
        return content;
    };
    
    llm_thread = std::thread([this, make_request, llm_predict, func]() {
        thread_running = true;
        auto start = std::chrono::steady_clock::now();
        while (thread_running) {
            status = LLM_IDLE;
            if (force_summarize && refined_text.size() > 0) {
                status = LLM_SUMMARIZE;
                summarized_text = llm_predict(refined_text, 
                    summarize_system_prompt);
                if (summarized_text.empty()) continue;
                if (func) func("summarize", summarized_text);
                force_summarize = false;
                continue;
            }
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start);
            if ((elapsed.count() < refine_span) && !force_refine) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            start = now;
            std::string text = wait_refine_messages.fetch(refine_chunk_size);
            if (force_refine) force_refine = false;
            if (text.empty()) continue;
            status = LLM_REFINE;
            std::string refined = llm_predict(text, refine_system_prompt);
            if (refined.empty()) continue;
            if (refine_output_file.is_open()) {
                refine_output_file << refined;
            }
            if (func) func("refine", refined);
            refined_text += refined;
        }
    });
    return 0;
}

int LLM::shutdown() {
    openai::stop();

    thread_running = false;
    if (llm_thread.joinable()) {
        llm_thread.join();
    }

    if (refine_output_file.is_open()) {
        refine_output_file.close();
    }
    if (summarize_output_file.is_open()) {
        summarize_output_file << summarized_text;
        summarize_output_file.close();
    }
    return 0;
}

int LLM::refine(const std::string& text) {
    if (text.size() > 0) {
        wait_refine_messages.push(text);
    } else {
        force_refine = true;
    }
    return 0;
}

int LLM::summarize() {
    force_summarize = true;
    return 0;
}
