# Try to find the GLM library and its include path.
# (C) 2014 Wong Yong Jie
#
# These variables are defined:
# GLM_FOUND
# GLM_INCLUDE_DIRS

find_path (GLM_INCLUDE_DIR glm/glm.hpp
    /usr/include
    /usr/local/include)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (GLM FOUND_VAR GLM_FOUND
    REQUIRED_VARS GLM_INCLUDE_DIR)

if (GLM_FOUND)
    set (GLM_INCLUDE_DIRS ${GLM_INCLUDE_DIR})
endif (GLM_FOUND)

mark_as_advanced (GLM_INCLUDE_DIR)

# vim: set ts=4 sw=4 et:
