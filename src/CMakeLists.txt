set(IMGUI_SOURCE ../third_party/imgui)
set(IMGUI_FILES 
    ${IMGUI_SOURCE}/imgui.cpp 
    ${IMGUI_SOURCE}/imgui_draw.cpp
    ${IMGUI_SOURCE}/imgui_tables.cpp
    ${IMGUI_SOURCE}/imgui_widgets.cpp
    ${IMGUI_SOURCE}/backends/imgui_impl_glfw.cpp
    ${IMGUI_SOURCE}/backends/imgui_impl_opengl3.cpp
)
find_library(OpenGL_LIBS OpenGL)
set(IMGUI_LIBS glfw ${OpenGL_LIBS})

set(FILES main.cpp ui.cpp audio.cpp asr.cpp llm.cpp)

add_executable(voicelint ${FILES} ${IMGUI_FILES})
target_link_libraries(voicelint 
    PRIVATE
        boost_program_options
        portaudio
        swresample
        sndfile
        yaml-cpp
        kaldi-native-fbank-core
        onnxruntime
        crypto 
        ssl
        ${IMGUI_LIBS}
)