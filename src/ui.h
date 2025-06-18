#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <nlohmann/json.hpp>

#include "audio.h"
#include "llm.h"

namespace EchoNote {
class UI {
public:
    static UI& instance() {
        static UI _inst;
        return _inst;
    }

    UI(const UI&) = delete;
    UI operator =(const UI&) = delete;

    void show(const nlohmann::json& config, Audio * audio, LLM * llm);

    // name: "asr", "refine", "summarize", "log"
    void show(const std::string& name, const std::string& text);

    static void log(const std::string& text) {
        UI::instance().show("log", text);
    }

    void clear();

private:
    UI() = default;
    ~UI() = default;

    typedef struct _queue_t {
        std::deque<std::string> q;
        std::mutex mtx;
        const int max_size = 150;

        void push(const std::string& text) {
            std::lock_guard<std::mutex> lk(mtx);
            if (q.size() >= max_size) q.pop_front();
            q.push_back(text);
        }

        std::deque<std::string> snapshot() {
            std::lock_guard<std::mutex> lk(mtx);
            return q;
        }

        void clear() {
            std::lock_guard<std::mutex> lk(mtx);
            q.clear();
        }
    } queue_t;

    queue_t asr_messages;
    queue_t refine_messages;
    std::string summarize_message;
    queue_t log_messages;
};
};
