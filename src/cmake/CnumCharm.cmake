# Locates the charmtyles checkout and Charm++, shared by src/ and the Python build.
# Sets CHARMTYLES_HOME, CNUM_DEPS_DIR, CHARM_HOME, CHARMC, CHARM_ARCH, CHARM_CUDA,
# CHARM_RECONVERSE, CHARM_SMP.
include_guard(GLOBAL)

get_filename_component(_cnum_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT CHARMTYLES_HOME AND DEFINED ENV{CHARMTYLES_HOME})
    set(CHARMTYLES_HOME "$ENV{CHARMTYLES_HOME}")
endif()
if(NOT CHARMTYLES_HOME)
    get_filename_component(CHARMTYLES_HOME "${_cnum_root}/../.." ABSOLUTE)
endif()
set(CHARMTYLES_HOME "${CHARMTYLES_HOME}" CACHE PATH "charmtyles checkout (contains include/charmtyles)")
if(NOT EXISTS "${CHARMTYLES_HOME}/include/charmtyles/core/charmtyles.ci")
    message(FATAL_ERROR "charmtyles not found at ${CHARMTYLES_HOME}.\n"
        "charmnumeric builds against a charmtyles checkout: clone it at "
        "<charmtyles>/example/charmnumeric, or pass -DCHARMTYLES_HOME=<charmtyles> "
        "(or set the CHARMTYLES_HOME environment variable).")
endif()

# Dependencies fetched by scripts/install.sh land here and are picked up automatically.
if(NOT CNUM_DEPS_DIR)
    if(DEFINED ENV{CNUM_DEPS_DIR})
        set(CNUM_DEPS_DIR "$ENV{CNUM_DEPS_DIR}")
    else()
        set(CNUM_DEPS_DIR "${_cnum_root}/.deps")
    endif()
endif()
set(CNUM_DEPS_DIR "${CNUM_DEPS_DIR}" CACHE PATH "Prefix of dependencies installed by scripts/install.sh")

function(_cnum_charm_is_cuda dir out)
    set(_cuda OFF)
    if(EXISTS "${dir}/include/conv-mach-opt.h")
        file(STRINGS "${dir}/include/conv-mach-opt.h" _m REGEX "conv-mach-cuda\\.h")
        if(_m)
            set(_cuda ON)
        endif()
    endif()
    set(${out} ${_cuda} PARENT_SCOPE)
endfunction()

# CHARM_HOME is fixed at the first configure of a build directory, so a later
# reconfigure from a shell with another CHARM_HOME cannot silently switch it.
set(CHARM_HOME "" CACHE PATH "Charm++ build directory (contains bin/charmc)")
if(NOT CHARM_HOME)
    set(_charm "")
    if(DEFINED ENV{CHARM_HOME})
        set(_charm "$ENV{CHARM_HOME}")
    else()
        find_program(_cnum_charmc charmc NO_CACHE)
        if(_cnum_charmc)
            get_filename_component(_c "${_cnum_charmc}" REALPATH)
            get_filename_component(_charm "${_c}/../.." ABSOLUTE)
        else()
            file(GLOB _cands "${CNUM_DEPS_DIR}/src/charm-*/*/bin/charmc")
            string(TOUPPER "${GPU_BACKEND}" _want)
            foreach(_c IN LISTS _cands)
                get_filename_component(_d "${_c}/../.." ABSOLUTE)
                _cnum_charm_is_cuda("${_d}" _is_cuda)
                if(_want STREQUAL "NVIDIA" AND _is_cuda)
                    set(_charm "${_d}")
                elseif(NOT _want STREQUAL "NVIDIA" AND NOT _is_cuda AND NOT _charm)
                    set(_charm "${_d}")
                endif()
            endforeach()
        endif()
    endif()
    if(NOT _charm)
        message(FATAL_ERROR "Charm++ not found.\n"
            "Run scripts/install.sh (builds Charm++ into ${CNUM_DEPS_DIR}), or point "
            "-DCHARM_HOME / the CHARM_HOME environment variable at a Charm++ build "
            "directory such as charm/netlrts-linux-x86_64 (charm/netlrts-linux-x86_64-cuda "
            "for NVIDIA GPUs).")
    endif()
    set(CHARM_HOME "${_charm}" CACHE PATH "Charm++ build directory (contains bin/charmc)" FORCE)
elseif(DEFINED ENV{CHARM_HOME})
    get_filename_component(_a "$ENV{CHARM_HOME}" REALPATH)
    get_filename_component(_b "${CHARM_HOME}" REALPATH)
    if(NOT _a STREQUAL _b)
        message(WARNING "Ignoring CHARM_HOME=$ENV{CHARM_HOME} from the environment: this "
            "build directory uses ${CHARM_HOME}. Pass -DCHARM_HOME=<dir> to switch.")
    endif()
endif()

set(CHARMC "${CHARM_HOME}/bin/charmc")
if(NOT EXISTS "${CHARMC}")
    message(FATAL_ERROR "CHARM_HOME=${CHARM_HOME} has no bin/charmc; it must be a Charm++ "
        "build directory such as charm/netlrts-linux-x86_64.")
endif()

_cnum_charm_is_cuda("${CHARM_HOME}" CHARM_CUDA)
set(CHARM_RECONVERSE OFF)
if(EXISTS "${CHARM_HOME}/lib/libreconverse.so" OR EXISTS "${CHARM_HOME}/lib/libreconverse.a")
    set(CHARM_RECONVERSE ON)
endif()
set(CHARM_SMP ${CHARM_RECONVERSE})
file(STRINGS "${CHARM_HOME}/include/conv-mach-opt.h" _smp REGEX "conv-mach-smp\\.h")
if(_smp)
    set(CHARM_SMP ON)
endif()
# CHARM_ARCH: the platform Charm++ records, plus the build options that matter here.
file(STRINGS "${CHARM_HOME}/include/conv-autoconfig.h" _plat REGEX "^#define CHARM_PLATFORM ")
string(REGEX REPLACE "^#define CHARM_PLATFORM +" "" CHARM_ARCH "${_plat}")
if(NOT CHARM_ARCH)
    get_filename_component(_real "${CHARM_HOME}" REALPATH)
    get_filename_component(CHARM_ARCH "${_real}" NAME)
else()
    if(CHARM_SMP AND NOT CHARM_RECONVERSE)
        string(APPEND CHARM_ARCH "-smp")
    endif()
    if(CHARM_CUDA)
        string(APPEND CHARM_ARCH "-cuda")
    endif()
endif()
message(STATUS "Charm++: ${CHARM_HOME} (${CHARM_ARCH}; CUDA=${CHARM_CUDA}, Reconverse=${CHARM_RECONVERSE}, SMP=${CHARM_SMP})")
