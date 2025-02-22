
set(proj VTKv8)

# Set dependency list
set(${proj}_DEPENDENCIES "zlib")

set(CIP_USE_PYTHONQT OFF)

#if (CIP_USE_PYTHONQT)
#  list(APPEND ${proj}_DEPENDENCIES python)
#endif()

if (USE_BOOST)
  list(APPEND ${proj}_DEPENDENCIES Boost)
endif()

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  unset(VTK_DIR CACHE)
  unset(VTK_SOURCE_DIR CACHE)
  find_package(VTK REQUIRED NO_MODULE)
endif()

# Sanity checks
if(DEFINED VTK_DIR AND NOT EXISTS ${VTK_DIR})
  message(FATAL_ERROR "VTK_DIR variable is defined but corresponds to non-existing directory")
endif()

if(DEFINED VTK_SOURCE_DIR AND NOT EXISTS ${VTK_SOURCE_DIR})
  message(FATAL_ERROR "VTK_SOURCE_DIR variable is defined but corresponds to non-existing directory")
endif()


if((NOT DEFINED VTK_DIR OR NOT DEFINED VTK_SOURCE_DIR) AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  set(CIP_VTK_RENDERING_BACKEND "OpenGL2" CACHE STRING "Choose the rendering backend.")
  set_property(CACHE CIP_VTK_RENDERING_BACKEND PROPERTY STRINGS "OpenGL" "OpenGL2")
  mark_as_superbuild(CIP_VTK_RENDERING_BACKEND)
  set(CIP_VTK_RENDERING_USE_${CIP_VTK_RENDERING_BACKEND}_BACKEND 1)


  set(EXTERNAL_PROJECT_OPTIONAL_ARGS)

  set(VTK_WRAP_TCL OFF)
  set(VTK_WRAP_PYTHON OFF)

#  if(CIP_USE_PYTHONQT)
#    set(VTK_WRAP_PYTHON ON)
#  endif()
#
#  if(CIP_USE_PYTHONQT)
#    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
#      -DPYTHON_EXECUTABLE:PATH=${PYTHON_EXECUTABLE}
#      -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
#      -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
#      )
#  endif()

  if(${PRIMARY_PROJECT_NAME}_USE_QT)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
            #-DDESIRED_QT_VERSION:STRING=4 # Unused
            -DVTK_USE_GUISUPPORT:BOOL=ON
            -DVTK_USE_QVTK_QTOPENGL:BOOL=ON
            -DVTK_USE_QT:BOOL=ON
            -DModule_vtkTestingRendering:BOOL=ON
            -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
            )
    if("${CIP_VTK_RENDERING_BACKEND}" STREQUAL "OpenGL2")
      list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
              -DModule_vtkGUISupportQtOpenGL:BOOL=ON
              #-DModule_vtkRenderingGL2PSOpenGL2:BOOL=ON

              )
    endif()
    if(APPLE)
      list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
              -DVTK_USE_CARBON:BOOL=OFF
              -DVTK_USE_COCOA:BOOL=ON # Default to Cocoa, VTK/CMakeLists.txt will enable Carbon and disable cocoa if needed
              -DVTK_USE_X:BOOL=OFF
              -DVTK_REQUIRED_OBJCXX_FLAGS:STRING=
              #-DVTK_USE_RPATH:BOOL=ON # Unused
              )
    endif()
    if(UNIX AND NOT APPLE)
      find_package(FontConfig QUIET)
      if(FONTCONFIG_FOUND)
        list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
                -DModule_vtkRenderingFreeTypeFontConfig:BOOL=ON
                )
      endif()
    endif()
  endif()

  # Disable Tk when Python wrapping is enabled
#  if(CIP_USE_PYTHONQT)
#    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS -DVTK_USE_TK:BOOL=OFF)
#  endif()

  if(VTK_WRAP_TCL)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
      -DTCL_INCLUDE_PATH:PATH=${TCL_INCLUDE_PATH}
      -DTK_INCLUDE_PATH:PATH=${TK_INCLUDE_PATH}
      -DTCL_LIBRARY:FILEPATH=${TCL_LIBRARY}
      -DTK_LIBRARY:FILEPATH=${TK_LIBRARY}
      -DTCL_TCLSH:FILEPATH=${TCL_TCLSH}
      )
  endif()

  # Enable VTK_ENABLE_KITS only if CMake >= 3.0 is used
  set(VTK_ENABLE_KITS 0)
  if(CMAKE_MAJOR_VERSION EQUAL 3)
    set(VTK_ENABLE_KITS 1)
  endif()

  set(${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY "github.com/Slicer/VTK.git" CACHE STRING "Repository from which to get VTK" FORCE)
  set(${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG "31dc6a08b8268133eb8bad83b7a65d70535673fa" CACHE STRING "VTK git tag to use" FORCE)   # slicer-2019-06-21


  mark_as_advanced(${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY ${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG)

  if(NOT DEFINED git_protocol)
    set(git_protocol "git")
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}
    BINARY_DIR ${proj}-build
    GIT_REPOSITORY "${git_protocol}://${${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY}"
    GIT_TAG ${${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG}
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      -DCMAKE_BUILD_TYPE:STRING=${EXTERNAL_PROJECT_BUILD_TYPE}
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DVTK_USE_PARALLEL:BOOL=ON
      -DVTK_DEBUG_LEAKS:BOOL=${VTK_DEBUG_LEAKS}
      -DVTK_LEGACY_REMOVE:BOOL=ON
      -DVTK_WRAP_TCL:BOOL=${VTK_WRAP_TCL}
      #-DVTK_USE_RPATH:BOOL=ON # Unused
      -DVTK_WRAP_PYTHON:BOOL=${VTK_WRAP_PYTHON}
      -DVTK_INSTALL_RUNTIME_DIR:PATH=${CIP_INSTALL_BIN_DIR}
      -DVTK_INSTALL_LIB_DIR:PATH=${CIP_INSTALL_LIB_DIR}
      -DVTK_INSTALL_ARCHIVE_DIR:PATH=${CIP_INSTALL_LIB_DIR}
      #-DVTK_Group_Qt:BOOL=ON   # Error when ON and QT4 is not installed
      -DVTK_USE_SYSTEM_ZLIB:BOOL=OFF
      -DZLIB_ROOT:PATH=${ZLIB_ROOT}
      -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
      -DZLIB_LIBRARY:FILEPATH=${ZLIB_LIBRARY}
      -DVTK_ENABLE_KITS:BOOL=${VTK_ENABLE_KITS}
      -DVTK_RENDERING_BACKEND:STRING=${CIP_VTK_RENDERING_BACKEND}
      -DModule_vtkTestingRendering:BOOL=ON
      -DModule_vtkInfovisBoostGraphAlgorithms:BOOL=${USE_BOOST}   # Build this module only if Boost is active
      -DBOOST_ROOT:PATH=${BOOST_ROOT}
      -DBoost_NO_BOOST_CMAKE:BOOL=ON    # Important in order not to search for System Boost in Unix!
#      -DQT_QMAKE_EXECUTABLE:PATH=${CIP_PYTHON_INSTALL_DIR}/bin/qmake
#      -DQT_RCC_EXECUTABLE:PATH=${CIP_PYTHON_INSTALL_DIR}/bin/rcc
      ${EXTERNAL_PROJECT_OPTIONAL_ARGS}
    INSTALL_COMMAND ""
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(VTK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(VTK_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})

  set(PNG_INCLUDE_DIR ${VTK_SOURCE_DIR}/Utilities/vtkpng)

  set(PNG_LIBRARY_DIR ${VTK_DIR}/bin)
  if(CMAKE_CONFIGURATION_TYPES)
    set(PNG_LIBRARY_DIR ${PNG_LIBRARY_DIR}/${CMAKE_CFG_INTDIR})
  endif()
  if(WIN32)
    set(PNG_LIBRARY ${PNG_LIBRARY_DIR}/vtkpng.lib)
  elseif(APPLE)
    set(PNG_LIBRARY ${PNG_LIBRARY_DIR}/libvtkpng.dylib)
  else()
    set(PNG_LIBRARY ${PNG_LIBRARY_DIR}/libvtkpng.so)
  endif()

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(VTK_SOURCE_DIR:PATH)

set (VTK_LIBXML_INCLUDE_DIR ${VTK_SOURCE_DIR}/ThirdParty/libxml2/vtklibxml2)
mark_as_superbuild(VTK_LIBXML_INCLUDE_DIR:PATH)


mark_as_superbuild(
  VARS VTK_DIR:PATH
  LABELS "FIND_PACKAGE"
  )

ExternalProject_Message(${proj} "PNG_INCLUDE_DIR:${PNG_INCLUDE_DIR}")
ExternalProject_Message(${proj} "PNG_LIBRARY:${PNG_LIBRARY}")
ExternalProject_Message(${proj} "LIBXML_INCLUDE_DIR:${VTK_LIBXML_INCLUDE_DIR}")