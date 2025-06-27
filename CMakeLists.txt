cmake_minimum_required(VERSION 3.15...3.30)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  ${SKBUILD_PROJECT_NAME}_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/${SKBUILD_PROJECT_NAME}_ext.cpp
  src/fixed_dimensional_encoding.cpp
)
target_include_directories(${SKBUILD_PROJECT_NAME}_ext
  PRIVATE
    ${PROJECT_SOURCE_DIR}
)

# Install directive for scikit-build-core
install(TARGETS ${SKBUILD_PROJECT_NAME}_ext LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

nanobind_add_stub(
  ${SKBUILD_PROJECT_NAME}_ext_stub
  MODULE ${SKBUILD_PROJECT_NAME}_ext
  OUTPUT ${SKBUILD_PROJECT_NAME}_ext.pyi  # sphinx is happy with this one
  PYTHON_PATH $<TARGET_FILE_DIR:${SKBUILD_PROJECT_NAME}_ext>
  DEPENDS ${SKBUILD_PROJECT_NAME}_ext
  MARKER_FILE py.typed
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/py.typed
  ${CMAKE_CURRENT_BINARY_DIR}/${SKBUILD_PROJECT_NAME}_ext.pyi
  DESTINATION ${SKBUILD_PROJECT_NAME})
