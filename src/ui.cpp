#include "ui.h"
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#define GL_SILENCE_DEPRECATION

#include <OpenGL/gl.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static auto error_callback = [](
    int error_code, const char* description 
) {
    std::cout << std::format("GLFW error[{}]: {}\n", error_code, description);
};

typedef struct _user_data_t {
    EchoNote::UI * ui;
    Audio * audio;
    LLM * llm;
    bool& show_log;
    GLuint& waiting;
    GLuint& processing;
    std::vector<std::string>& history;
    int& current_history_index;
} user_data_t;

static auto key_callback = [](GLFWwindow* window, int key, 
    int scancode, int action, int mods) {
    user_data_t * user_data = reinterpret_cast<user_data_t *>(
        glfwGetWindowUserPointer(window)
    );
    if (!user_data || !user_data->audio) return;
    if (key == GLFW_KEY_C && action == GLFW_PRESS 
        && (mods & GLFW_MOD_CONTROL) != GLFW_MOD_CONTROL) {
        user_data->audio->start();
        user_data->ui->clear();
        user_data->current_history_index = -1;
    }
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        user_data->audio->stop();
    }
    if (key == GLFW_KEY_L && action == GLFW_PRESS) {
        user_data->show_log = !user_data->show_log;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        user_data->llm->refine("");
    }
    if (key == GLFW_KEY_S && action == GLFW_PRESS) {
        user_data->llm->summarize();
    }
};

static auto setup_fonts = [](const nlohmann::json& fonts) {
    auto io = ImGui::GetIO();
    ImFontConfig fc;
    fc.MergeMode = false;
    fc.OversampleH = fc.OversampleV = 3;
    fc.PixelSnapH = false;
    io.FontDefault = io.Fonts->AddFontFromFileTTF(
        fonts[0]["filename"].get<std::string>().c_str(), 
        fonts[0]["size"].get<float>(), 
        &fc
    );
    for (int i=1; i<fonts.size(); ++i) {
        const ImWchar * ranges = nullptr;
        if (fonts[i]["language"].get<std::string>() == "chinese") {
            ranges = io.Fonts->GetGlyphRangesChineseFull();
        } else if (fonts[i]["language"].get<std::string>() == "greek") {
            ranges = io.Fonts->GetGlyphRangesGreek();
        } else if (fonts[i]["language"].get<std::string>() == "korean") {
            ranges = io.Fonts->GetGlyphRangesKorean();
        } else if (fonts[i]["language"].get<std::string>() == "japanese") {
            ranges = io.Fonts->GetGlyphRangesJapanese();
        } else if (fonts[i]["language"].get<std::string>() == "cyrillic") {
            ranges = io.Fonts->GetGlyphRangesCyrillic();
        } else if (fonts[i]["language"].get<std::string>() == "thai") {
            ranges = io.Fonts->GetGlyphRangesThai();
        } else if (fonts[i]["language"].get<std::string>() == "vietnamese") {
            ranges = io.Fonts->GetGlyphRangesVietnamese();
        } else {
            continue;
        }
        fc.MergeMode = true;
        io.Fonts->AddFontFromFileTTF(
            fonts[i]["filename"].get<std::string>().c_str(), 
            fonts[i]["size"].get<float>(), 
            &fc,
            ranges
        );
    }
    io.Fonts->Build();
};

typedef void (*create_child_components)(const user_data_t&);
static auto create_component = [](const user_data_t& user_data, 
    const ImVec2& pos, const ImVec2& size, const std::string& name, 
    create_child_components child_components, bool no_bring_to_front = true) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGuiWindowFlags flags = ImGuiWindowFlags_None;
    flags |= ImGuiWindowFlags_NoTitleBar;
    flags |= ImGuiWindowFlags_NoMove;
    flags |= ImGuiWindowFlags_NoResize;
    flags |= ImGuiWindowFlags_NoCollapse;
    if (no_bring_to_front) flags |= ImGuiWindowFlags_NoBringToFrontOnFocus; 

    ImGui::Begin(name.c_str(), nullptr, flags);
    child_components(user_data);
    ImGui::End();
};

static std::vector<std::string> load_history() {
    std::vector<std::string> history;
    for (auto & entry: std::filesystem::directory_iterator("data")) {
        if (entry.is_directory()) {
            history.push_back(entry.path().filename().string());
        }
    }
    if (!history.empty()) {
        std::sort(history.begin(), history.end(), 
            std::greater<std::string>());
    }
    return history;
}

static void load_history_to_ui(EchoNote::UI * ui, const std::string& history) {
    const std::string path = "data/" + history;
    ui->clear();
    {
        std::ifstream file(path + "/asr.txt");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                ui->show("asr", line);
            }
            file.close();
        }
    }
    {
        std::ifstream file(path + "/refine.txt");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                ui->show("refine", line);
            }
            file.close();
        }
    }
    {
        std::ifstream file(path + "/summarize.txt");
        if (file.is_open()) {
            std::string line = std::string(
                std::istreambuf_iterator<char>(file), 
                std::istreambuf_iterator<char>());
            ui->show("summarize", line);
            file.close();
        }
    }
}

static GLuint load_texture(const std::string& filename) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, channels;
    unsigned char * data = stbi_load(filename.c_str(), &width, &height, 
        &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 
        0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);

    return texture_id;
}

void EchoNote::UI::show(const nlohmann::json& config, Audio * audio, LLM * llm) {
    std::string name = config.value("name", "EchoNote");
    int width = config.value("width", 1280);
    int height = config.value("height", 720);
    const float status_bar_height = 30.f;
    auto fonts = config.value("fonts", R"(
        [
            {
                "filename": "res/fonts/MonaspaceRadonVarVF[wght,wdth,slnt].ttf",
                "size": 16.0
            },
            {
                "filename": "res/fonts/LXGWWenKai-Regular.ttf",
                "size": 16.0,
                "language": "chinese"
            }
        ]
        )"_json);
    
    if (!glfwInit()) {
        std::cout << "GLFW initialization failed." << std::endl;
        return;
    }

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow * window = glfwCreateWindow(
        width, height, name.c_str(), nullptr, nullptr
    );
    if (!window) {
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    auto status_waiting = load_texture("res/images/emoji_1067.png");
    auto status_processing = load_texture("res/images/emoji_1068.png");
    if (status_waiting == 0 || status_processing == 0) {
        std::cout << "Failed to load status textures." << std::endl;
        return;
    }

    auto history = load_history();
    //for (int i=0; i<100; ++i) history.push_back(std::to_string(i) + "_202506171800");
    int current_history_index = -1;

    bool show_log = false;
    user_data_t user_data = {
        this, audio, llm, show_log, 
        status_waiting, status_processing,
        history, current_history_index
    };
    glfwSetWindowUserPointer(window, &user_data);
    glfwSetKeyCallback(window, key_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    setup_fonts(fonts);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            //history
            create_component(user_data, {0, 0}, 
                {width * 0.15f, height - status_bar_height}, 
                "History", 
                [](const user_data_t& user_data) {
                    ImGui::SeparatorText("History");
                    auto size = ImGui::GetContentRegionAvail();
                    ImGui::BeginChild("history", size);
                    for (int i=0; i<user_data.history.size(); ++i) {
                        bool is_selected = (i == user_data.current_history_index);
                        ImGui::Selectable(user_data.history[i].c_str(), 
                            is_selected);
                        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                            if (user_data.audio->isRecording()) {
                                user_data.ui->log("Cannot load history while recording.");
                                continue;
                            }
                            load_history_to_ui(user_data.ui, user_data.history[i]);
                            user_data.current_history_index = i;
                        }
                    }
                    ImGui::EndChild();
                });
            //asr
            create_component(user_data, {width * 0.15f, 0}, 
                {width * 0.525f, height - status_bar_height}, 
                "Automatic Speech Recognition", 
                [](const user_data_t& user_data) {
                    ImGui::SeparatorText("Automatic Speech Recognition");
                    ImGui::BeginChild("asr messages", ImVec2(0, 0), 
                        ImGuiChildFlags_None, 
                        ImGuiWindowFlags_AlwaysVerticalScrollbar);
                    auto q = user_data.ui->asr_messages.snapshot();
                    for (auto& text: q) {
                        ImGui::TextWrapped("%s", text.c_str());
                    }

                    float scroll_y = ImGui::GetScrollY();
                    float scroll_max_y = ImGui::GetScrollMaxY();
                    float delta_y = scroll_max_y - scroll_y;
                    bool auto_scroll = (delta_y > 1.0f) ? false : true;
                    if (auto_scroll) ImGui::SetScrollHereY(1.0f);

                    ImGui::EndChild();
                });
        }
        {
            //refine
            create_component(user_data, {width * 0.675f, 0}, 
                {width * 0.325f, (height - status_bar_height) / 2.0f}, 
                "Refine Message", 
                [](const user_data_t& user_data) {
                    ImGui::SeparatorText("Refine Message");
                    ImGui::BeginChild("refine messages", ImVec2(0, 0), 
                        ImGuiChildFlags_None, 
                        ImGuiWindowFlags_AlwaysVerticalScrollbar);
                    auto q = user_data.ui->refine_messages.snapshot();
                    for (auto& text: q) {
                        ImGui::TextWrapped("%s", text.c_str());
                    }

                    float scroll_y = ImGui::GetScrollY();
                    float scroll_max_y = ImGui::GetScrollMaxY();
                    float delta_y = scroll_max_y - scroll_y;
                    bool auto_scroll = (delta_y > 1.0f) ? false : true;
                    if (auto_scroll) ImGui::SetScrollHereY(1.0f);

                    ImGui::EndChild();
                });
        }
        {
            //summarize
            create_component(user_data, {width * 0.675f, (height - 30.f) / 2.0f}, 
                {width * 0.325f, (height - status_bar_height) / 2.0f}, 
                "Summary", 
                [](const user_data_t& user_data) {
                    ImGui::SeparatorText("Summary");
                    ImGui::BeginChild("summary message", ImVec2(0, 0), 
                        ImGuiChildFlags_None, 
                        ImGuiWindowFlags_AlwaysVerticalScrollbar);
                    ImGui::TextWrapped("%s", user_data.ui->summarize_message.c_str());
                    ImGui::EndChild();
                });
        }
        {
            //help and status
            create_component(user_data, {0.f, height - 30.f}, 
                {width * 0.80f, status_bar_height}, 
                "Help", 
                [](const user_data_t& user_data) {
                    ImGui::Text(
                        "Press 'C' to start Recording. "
                        "Press 'P' to stop Recording. "
                        "Press 'R' to refine. "
                        "Press 'S' to summarize. "
                        "Press 'L' to show|hide log window. "
                    );
                });
            ImGui::SameLine();
            create_component(user_data, {width * 0.80f, height - 30.f}, 
                {width * 0.20f, status_bar_height}, 
                "Status", 
                [](const user_data_t& user_data) {
                    {
                        ImTextureID audio_icon = user_data.audio->isRecording() ? 
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.processing)) :
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.waiting));
                        ImGui::Image(audio_icon, ImVec2(13, 13));
                        ImGui::SameLine();
                        ImGui::Text("Audio");
                    }
                    ImGui::SameLine();
                    {
                        ImTextureID audio_icon = user_data.llm->isRefine() ? 
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.processing)) :
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.waiting));
                        ImGui::Image(audio_icon, ImVec2(13, 13));
                        ImGui::SameLine();
                        ImGui::Text("Refine");
                    }
                    ImGui::SameLine();
                    {
                        ImTextureID audio_icon = user_data.llm->isSummarize() ? 
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.processing)) :
                            static_cast<ImTextureID>(static_cast<intptr_t>(user_data.waiting));
                        ImGui::Image(audio_icon, ImVec2(13, 13));
                        ImGui::SameLine();
                        ImGui::Text("Summarize");
                    }
                });
        }

        if (show_log) {
            create_component(user_data, {width * 0.25f, height * 0.25f}, 
                {width * 0.5f, height * 0.5f}, 
                "log", 
                [](const user_data_t& user_data) {
                    ImGui::SetNextWindowFocus();
                    ImGui::SeparatorText("Log Message");
                    ImGui::BeginChild("Log messages", ImVec2(0, 0), 
                        ImGuiChildFlags_None, 
                        ImGuiWindowFlags_AlwaysVerticalScrollbar);
                    auto q = user_data.ui->log_messages.snapshot();
                    for (auto& text: q) {
                        ImGui::TextWrapped("%s", text.c_str());
                    }

                    float scroll_y = ImGui::GetScrollY();
                    float scroll_max_y = ImGui::GetScrollMaxY();
                    float delta_y = scroll_max_y - scroll_y;
                    bool auto_scroll = (delta_y > 1.0f) ? false : true;
                    if (auto_scroll) ImGui::SetScrollHereY(1.0f);

                    ImGui::EndChild();
                }, false);
        }

        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

// name: "asr", "refine", "summarize", "log"
void EchoNote::UI::show(const std::string& name, const std::string& text) {
    if (name == "asr") asr_messages.push(text);
    if (name == "refine") refine_messages.push(text);
    if (name == "summarize") summarize_message = text;
    if (name == "log") log_messages.push(text);
}

void EchoNote::UI::clear() {
    asr_messages.clear();
    refine_messages.clear();
    summarize_message.clear(); 
}
