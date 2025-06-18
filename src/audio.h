#pragma once

#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <portaudio.h>
#include <queue>
#include <thread>
#include <vector>
#include <sndfile.h>

extern "C" {
#include <libswresample/swresample.h>
}

class Audio {
    const int sampleRate = 16000; // Default sample rate

public:
    static Audio& instance() {
        static Audio _inst;
        return _inst;
    }
    Audio(const Audio&) = delete;
    Audio operator=(const Audio&) = delete;

    int init(const nlohmann::json& config);
    int shutdown();

    int start();
    int stop();

    bool isRecording() const {
        if (stream && Pa_IsStreamActive(stream) == 1) {
            return true; // Stream is active
        }
        return false; // Stream is not active
    };

    std::string getOutFile() const {
        return audio_out_path; // Return the path to the audio file
    }

    std::vector<float> readAudio(int ms);

private:
    Audio() = default;
    ~Audio() = default;

    PaStream* stream = nullptr;

    typedef struct _AudioQueue {
        std::queue<std::vector<float>> q;
        std::mutex mtx;
        std::condition_variable cv;
        int maxSize = 100000; // Maximum size of the queue

        int push(const std::vector<float>& data) {
            std::lock_guard<std::mutex> lock(mtx);
            if (q.size() >= maxSize) {
                q.pop(); // Remove the oldest element if queue is full
            }
            q.push(data);
            cv.notify_one(); // Notify one waiting thread
            return 0; // Return 0 on success
        }

        std::vector<float> pop() {
            std::unique_lock<std::mutex> lock(mtx);
            if (!cv.wait_for(lock, 
                std::chrono::milliseconds(100), 
                [this] { return !q.empty(); })) {
                return {}; // Return an empty vector if timeout occurs
            }

            auto data = q.front();
            q.pop();
            return data; // Return the popped data
        }
    } AudioQueue;
    AudioQueue audioQueue;

    static int audioCallback(const void* inputBuffer, void* outputBuffer, 
                            unsigned long framesPerBuffer, 
                            const PaStreamCallbackTimeInfo* timeInfo, 
                            PaStreamCallbackFlags statusFlags, 
                            void* userData);

    typedef struct _AudioBuffer {
        std::vector<float> data; // Audio data buffer
        size_t capacity = 0; // Capacity of the audio data buffer
        int readIndex = 0; // Read index for the buffer
        int writeIndex = 0; // Write index for the buffer

        _AudioBuffer() = default;
        _AudioBuffer(size_t size) : capacity(size) {
            data.resize(size); // Reserve space for audio data
        }

        _AudioBuffer operator=(const _AudioBuffer& other) {
            if (this != &other) {
                data = other.data;
                capacity = other.capacity;
                readIndex = other.readIndex;
                writeIndex = other.writeIndex;
            }
            return *this; // Return the current object
        }

        int write(const float* input, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                data[writeIndex] = input[i];
                writeIndex = (writeIndex + 1) % capacity; // Wrap around if necessary
                if (writeIndex == readIndex) {
                    readIndex = (readIndex + 1) % capacity; // Move read index if buffer is full
                }
            }
            return 0; // Return 0 on success
        }

        std::vector<float> read(size_t size) {
            std::vector<float> output;
            for (size_t i = 0; i < size; ++i) {
                if (readIndex == writeIndex) {
                    break; // Stop if no more data to read
                }
                output.push_back(data[readIndex]);
                readIndex = (readIndex + 1) % capacity; // Wrap around if necessary
            }
            return output; // Return the read data
        }
    } AudioBuffer;
    AudioBuffer audioBuffer; // Audio buffer for storing audio data

    SwrContext* swrContext = nullptr; // SwrContext for resampling audio

    std::thread resampleThread; // Thread for processing audio data
    bool resample_running = false; // Flag to indicate if audio processing is running

    bool saveAudio = false; // Flag to indicate if audio should be saved
    std::string audio_out_path = "output/output.mp3"; // Path to save the audio file
    SNDFILE* sndFile = nullptr; // SNDFILE handle for audio file operations

    int process(const std::vector<float>& audioData, bool resample = true);
};
