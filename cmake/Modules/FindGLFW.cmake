# Try to find the GLFW library and its include path.
# (C) 2014 Wong Yong Jie
#
# These variables are defined:
# GLFW_FOUND
# GLFW_INCLUDE_DIRS
# GLFW_LIBRARIES

# If we have pkg-config, just use it.
find_package (PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_search_module (GLFW glfw3)
endif (PKG_CONFIG_FOUND)

# Sometimes pkg-config may not find it, but manual search does.
# Or the user does not have pkg-config.
if (NOT PKG_CONFIG_FOUND OR NOT GLFW_FOUND)
    find_path (GLFW_INCLUDE_DIR GLFW/glfw3.h
        /usr/include
        /usr/local/include)

    find_library (GLFW_LIBRARY NAMES glfw3 PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib)

    include (FindPackageHandleStandardArgs)
    find_package_handle_standard_args (GLFW GLFW_LIBRARY GLFW_INCLUDE_DIR)

    if (GLFW_FOUND)
        set (GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
        set (GLFW_LIBRARIES ${GLFW_LIBRARY})
    endif (GLFW_FOUND)

    mark_as_advanced (GLFW_INCLUDE_DIR GLFW_LIBRARY)

endif (NOT PKG_CONFIG_FOUND OR NOT GLFW_FOUND)

# vim: set ts=4 sw=4 et:
