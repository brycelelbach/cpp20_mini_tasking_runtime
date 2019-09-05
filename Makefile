###############################################################################
# Copyright (c) 2018-9 NVIDIA Corporation
# Copyright (c) 2015-6 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

# Class of build (`debug`, `optimized_debug`, `release`).
BUILD_TYPE      ?= debug
# The build directory to create and use.
BUILD_DIRECTORY ?= $(ROOT)/build__$(BUILD_TYPE)

# ISO C++ (`.cpp`) -> Object File (`.o`) -> Executable

# ISO C++ compiler (path to executable).
ISO_CXX                     ?= /home/wash/install/cpp_toolchains/clang_llvm_concepts/bin/clang++
# ISO C++ dialect.
ISO_CXX_DIALECT             ?= c++2a
# Flags passed to the ISO C++ compiler when compiling ISO C++ code.
ISO_CXX_FLAGS               ?= -Xclang -fconcepts-ts -D__TABLE_INLINE -D__BARRIER_INLINE_FAVORITE_HASH -D__BARRIER_NO_BUTTERFLY -D__NO_SEM_FRONT -pthread -Wall
# Directories to search for ISO C++ headers in.
ISO_CXX_INCLUDE_DIRECTORIES ?= /home/wash/development/nvidia/cjdb_concepts_ranges/include \
                               /home/wash/development/nvidia/cpp20_synchronization_library/include

###############################################################################

# Clear previous build log, if there is one.
IGNORE := $(shell rm -f build.log)

# Diagnostics macro for use outside of rules. Prints its single argument to
# both stdout and the build log. Note that it has to escape `$`.
define PRINT_CONFIG =
  $(info $(shell echo $(subst $$,\$$,$(1)) | tee -a build.log))
endef

# Diagnostics macro for use within rules. Prints its single argument to both
# stdout and the build log.
define PRINT_RULE =
	@echo $(1) | tee -a build.log
	@$(1) 2>&1 | tee -a build.log
endef

IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// Settings")
IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// BUILD_TYPE                  : $(value BUILD_TYPE)")
IGNORE := $(call PRINT_CONFIG,"// BUILD_DIRECTORY             : $(value BUILD_DIRECTORY)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX                     : $(value ISO_CXX)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_DIALECT             : $(value ISO_CXX_DIALECT)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_FLAGS               : $(value ISO_CXX_FLAGS)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_INCLUDE_DIRECTORIES : $(value ISO_CXX_INCLUDE_DIRECTORIES)")
IGNORE := $(call PRINT_CONFIG) # Print blank newline.

###############################################################################

# Get the directory where the Makefile and code live.
ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Strip terminating `/`s from directory paths.
# This has to be done late because these inplace updates must be immediate.
ROOT                        := $(ROOT:/=)
BUILD_DIRECTORY             := $(BUILD_DIRECTORY:/=)
ISO_CXX_INCLUDE_DIRECTORIES := $(foreach flag,$(ISO_CXX_INCLUDE_DIRECTORIES),$(flag:/=))

###############################################################################

# Strip leading and trailing whitespace from flags.
ISO_CXX_FLAGS   := $(strip $(ISO_CXX_FLAGS))

ifeq      ($(BUILD_TYPE),release)
  ISO_CXX_FLAGS += -O3 -DNDEBUG
else ifeq ($(BUILD_TYPE),optimized_debug)
  ISO_CXX_FLAGS += -O3 -g -DNDEBUG
else ifeq ($(BUILD_TYPE),debug)
  ISO_CXX_FLAGS += -O0 -g -DSAFE
endif

ISO_CXX_FLAGS   += -std=$(ISO_CXX_DIALECT)

# Add include directories to `ISO_CXX_FLAGS`.
ISO_CXX_FLAGS   += $(foreach flag,$(ISO_CXX_INCLUDE_DIRECTORIES),-I $(flag))

# Strip leading and trailing whitespace from flags.
ISO_CXX_FLAGS   := $(strip $(ISO_CXX_FLAGS))

###############################################################################

ISO_CXX_SOURCES = $(wildcard $(ROOT)/*.cpp)
ISO_CXX_TARGETS = $(ISO_CXX_SOURCES:.cpp=)

###############################################################################

IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// Computed Variables")
IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// ROOT                        : $(ROOT)")
IGNORE := $(call PRINT_CONFIG,"// BUILD_DIRECTORY             : $(BUILD_DIRECTORY)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_FLAGS               : $(ISO_CXX_FLAGS)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_INCLUDE_DIRECTORIES : $(ISO_CXX_INCLUDE_DIRECTORIES)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_SOURCES             : $(ISO_CXX_SOURCES)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_TARGETS             : $(ISO_CXX_TARGETS)")
IGNORE := $(call PRINT_CONFIG) # Print blank newline.

###############################################################################

all: $(ISO_CXX_TARGETS)

print_environment:
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// ISO C++ Compiler Version' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@$(ISO_CXX) --version 2>&1 | tee -a build.log
	@echo | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Environment' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@env 2>&1 | tee -a build.log
	@echo | tee -a build.log

clean:
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Cleaning Build Directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,rm -f $(BUILD_DIRECTORY)/*)
	$(call PRINT_RULE,[ -d "$(BUILD_DIRECTORY)" ] && rmdir $(BUILD_DIRECTORY))
	@echo | tee -a build.log

.PHONY: all print_environment clean

$(BUILD_DIRECTORY): print_environment
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Making Build Directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,mkdir -p $@)
	@echo | tee -a build.log

% : %.cpp $(BUILD_DIRECTORY)
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Building Executable $(*F) in Directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,$(ISO_CXX) $(ISO_CXX_FLAGS) $< -o $(BUILD_DIRECTORY)/$(*F))
	@echo | tee -a build.log

