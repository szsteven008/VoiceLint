#include "audio.h"
#include <portaudio.h>
#include <sndfile.h>
#include <string>

int Audio::init(const nlohmann::json& config) {
    std::string deviceName = config.value("device", "default");
    const int inputSampleRate = config.value("sampleRate", 44100);
    const int framesPerBuffer = config.value("framesPerBuffer", 256);
    int n_samples = config.value("max_n_samples", 30000);
    n_samples = n_samples * sampleRate / 1000;
    audioBuffer = AudioBuffer(n_samples); // Initialize audio buffer with max samples

    saveAudio = config.value("save", false);
    if (saveAudio) {
        audio_out_path = config.value("output", "output/output.mp3");
        SF_INFO sfinfo;
        sfinfo.format = SF_FORMAT_MPEG | SF_FORMAT_MPEG_LAYER_III; // Set format for MP3 output
        sfinfo.samplerate = sampleRate; // Set sample rate
        sfinfo.channels = 1; // Mono output
        sfinfo.frames = 0; // Initialize frames to 0
        sndFile = sf_open(audio_out_path.c_str(), SFM_WRITE, &sfinfo);
        if (!sndFile) {
            return sf_error(sndFile); // Return error code if file opening fails
        }
    }

    // Initialization logic here
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        return err; // Return error code if initialization fails
    }

    PaDeviceIndex inputDevice = Pa_GetDefaultInputDevice();
    for (int i = 0; i < Pa_GetDeviceCount(); ++i) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo && deviceName == deviceInfo->name) {
            inputDevice = i; // Set the input device if found
            break;
        }
    }
    if (inputDevice == paNoDevice) {
        return paInvalidDevice; // Return error if no valid device is found
    }

    // Set up the audio stream parameters
    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1; // Mono input
    inputParameters.sampleFormat = paFloat32; // 32-bit floating point
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    err = Pa_OpenStream(&stream, 
        &inputParameters, 
        nullptr, 
        inputSampleRate, 
        framesPerBuffer, 
        paClipOff, 
        audioCallback, 
        this);
    if (err != paNoError) {
        shutdown();
        return err; // Return error code if stream opening fails
    }

    swrContext = swr_alloc(); // Allocate SwrContext for resampling
    if (!swrContext) {
        shutdown();
        return paInsufficientMemory; // Return error if SwrContext allocation fails
    }

    // Set options for SwrContext
    const AVChannelLayout inputChannelLayout = AV_CHANNEL_LAYOUT_MONO;
    const AVChannelLayout outputChannelLayout = AV_CHANNEL_LAYOUT_MONO;
    swr_alloc_set_opts2(&swrContext, 
        &outputChannelLayout, // Output channel layout (will be set later)
        AV_SAMPLE_FMT_FLTP, // Output sample format
        sampleRate, // Output sample rate
        &inputChannelLayout, // Input channel layout (will be set later)
        AV_SAMPLE_FMT_FLT, // Input sample format
        inputSampleRate, // Input sample rate
        0, // Log offset
        nullptr); // Log context
    if (swr_init(swrContext) < 0) {
        shutdown();
        return paUnanticipatedHostError; // Return error if SwrContext initialization fails
    }

    resampleThread = std::thread([this, inputSampleRate]() {
        resample_running = true; // Set resample running flag to true
        while (resample_running) {
            auto data = audioQueue.pop(); // Pop data from the audio queue
            if (data.empty()) {
                continue; // Skip if no data is available
            }
            process(data, (inputSampleRate != sampleRate)); // Call resample function with the popped data
        }
    });

    return 0; // Return 0 on success
}

int Audio::shutdown() {
    resample_running = false; // Stop the resample thread
    if (resampleThread.joinable()) {
        resampleThread.join(); // Wait for the resample thread to finish
    }

    if (sndFile) {
        sf_close(sndFile); // Close the sound file
        sndFile = nullptr;
    }
    
    if (swrContext) {
        swr_free(&swrContext); // Free the SwrContext
        swrContext = nullptr;
    }

    if (stream) {
        stop();
        Pa_CloseStream(stream);
        stream = nullptr;
    }
    Pa_Terminate(); // Terminate PortAudio
    return 0; // Return 0 on success
}

int Audio::start() {
    if (!stream) return -1;
    if (Pa_IsStreamActive(stream)) return 0;

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        return -1; // Return error code if starting the stream fails
    }

    return 0;
}

int Audio::stop() {
    if (!stream) return -1;
    if (Pa_IsStreamStopped(stream)) return 0;

    Pa_StopStream(stream);
    while (Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(100); // Wait for the stream to stop
    }

    return 0;
}

std::vector<float> Audio::readAudio(int ms) {
    if (ms <= 0) {
        return {}; // Return empty vector if ms is not positive
    }
    int n_samples = ms * sampleRate / 1000; // Calculate number of samples to read
    return audioBuffer.read(n_samples); // Read audio data from the buffer
}

int Audio::audioCallback(const void* inputBuffer, void* outputBuffer, 
                        unsigned long framesPerBuffer, 
                        const PaStreamCallbackTimeInfo* timeInfo, 
                        PaStreamCallbackFlags statusFlags, 
                        void* userData) {
    (void)outputBuffer; // Unused output buffer
    (void)timeInfo; // Unused time info
    (void)statusFlags; // Unused status flags

    Audio* audio = static_cast<Audio*>(userData);
    if (!audio || !inputBuffer) {
        return paContinue; // Continue processing audio if no valid audio object or input buffer
    }
    const float* in = static_cast<const float*>(inputBuffer);
    std::vector<float> data(in, in + framesPerBuffer); // Copy input data to vector
    audio->audioQueue.push(data); // Push data to the audio queue
    return paContinue; // Continue processing audio
}

int Audio::process(const std::vector<float>& audioData, bool resample /* = true */) {
    if (audioData.empty()) {
        return 0; // Return 0 if no audio data is provided
    }

    std::vector<float> resampledData;
    if (resample && swrContext) {
        // Resample the audio data if resampling is enabled
        int outSamples = swr_get_out_samples(swrContext, audioData.size());
        resampledData.resize(outSamples); // Resize to accommodate output samples (mono)
        // Prepare input and output buffer arrays as required by swr_convert
        const float* inData[1] = { audioData.data() };
        float* outData[1] = { resampledData.data() };
        int ret = swr_convert(swrContext,
                              reinterpret_cast<uint8_t**>(outData), outSamples,
                              reinterpret_cast<const uint8_t**>(inData), audioData.size());
        if (ret < 0) {
            return ret; // Return error code if resampling fails
        }
        // Resize the resampled data to the actual number of samples written
        resampledData.resize(ret); // Resize to the number of samples actually written
    } else {
        // If resampling is not enabled, use the original audio data
        resampledData = audioData; // Copy the original audio data
    }
    // For simplicity, we will just write the audio data to the buffer
    audioBuffer.write(resampledData.data(), resampledData.size()); // Write only the number of samples actually written
    if (sndFile) {
        // Write the resampled audio data to the sound file
        sf_count_t framesWritten = sf_writef_float(sndFile, resampledData.data(), resampledData.size()); // Mono output
        if (framesWritten < 0) {
            return sf_error(sndFile); // Return error code if writing fails
        }
    }

    return 0; // Return 0 on success
}
