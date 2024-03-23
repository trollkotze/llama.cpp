include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp.in")
set(TEMP_FILE "${OUTPUT_DIR}/common/build-info.cpp.in")
set(OUTPUT_FILE "${OUTPUT_DIR}/common/build-info.cpp")

# Only write the build info if it changed
if(EXISTS ${OUTPUT_FILE})
    file(READ ${OUTPUT_FILE} CONTENTS)
    string(REGEX MATCH "LLAMA_COMMIT = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_COMPILER = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMPILER ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_TARGET = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_TARGET ${CMAKE_MATCH_1})
    if (
        NOT OLD_COMMIT   STREQUAL BUILD_COMMIT   OR
        NOT OLD_COMPILER STREQUAL BUILD_COMPILER OR
        NOT OLD_TARGET   STREQUAL BUILD_TARGET
    )
        message(STATUS ${TEMPLATE_FILE} ${TEMP_FILE} ${OUTPUT_FILE})
        configure_file(${TEMPLATE_FILE} ${TEMP_FILE} COPYONLY)
        configure_file(${TEMP_FILE} ${OUTPUT_FILE})
    endif()
else()
    message(STATUS ${TEMPLATE_FILE} ${TEMP_FILE} ${OUTPUT_FILE})
    configure_file(${TEMPLATE_FILE} ${TEMP_FILE} COPYONLY)
    configure_file(${TEMP_FILE} ${OUTPUT_FILE})
endif()
