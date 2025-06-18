#pragma once

#include <thread>
#include <fstream>
#include <nlohmann/json.hpp>

typedef void (* llm_callback)(const std::string& name, 
    const std::string& text);

class LLM {
public:
    static LLM& instance() {
        static LLM _inst;
        return _inst;
    }
    LLM(const LLM&) = delete;
    LLM operator=(const LLM&) = delete;

    int init(const nlohmann::json& config, llm_callback func);
    int shutdown();

    int refine(const std::string& text);
    int summarize();

    bool isRefine() const {
        return status == LLM_REFINE;
    };
    bool isSummarize() const {
        return status == LLM_SUMMARIZE;
    };

    std::string getRefineOutFile() const {
        return refine_output_path; // Return the path to the refine output file
    }
    std::string getSummarizeOutFile() const {
        return summarize_output_path; // Return the path to the summarize output file
    }

private:
    LLM() = default;
    ~LLM() = default;

    std::string model = "Qwen3-8b";
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 20;
    float presence_penalty = 1.5f;

    std::string refine_system_prompt = "";
    int refine_chunk_size = 1024;
    int refine_span = 120; // seconds
    bool refine_save = false;

    std::string summarize_system_prompt = "";
    bool summarize_save = false;

    bool thread_running = false;
    std::thread llm_thread;

    typedef struct _queue_t {
        std::deque<std::string> q;
        std::mutex mtx;
        int cur_size = 0;

        void push(const std::string& text) {
            std::lock_guard<std::mutex> lk(mtx);
            q.push_back(text);
            cur_size += text.size();
        }

        std::string fetch(int chunk_size) {
            std::unique_lock<std::mutex> lk(mtx);
            std::string result = "";

            while (!q.empty()) {
                result += q.front();
                cur_size -= q.front().size();
                q.pop_front();
                if (result.size() >= chunk_size) {
                    break;
                }
            }

            return result;
        }
    } queue_t;

    bool force_refine = false;
    bool force_summarize = false;

    queue_t wait_refine_messages;
    std::string refined_text;
    std::string summarized_text;

    std::string refine_output_path = "output/refine.txt";
    std::string summarize_output_path = "output/summarize.txt";
    std::ofstream refine_output_file;
    std::ofstream summarize_output_file;

    typedef enum {
        LLM_IDLE = 0,
        LLM_REFINE,
        LLM_SUMMARIZE
    } LLMStatus;
    LLMStatus status = LLM_IDLE;
};