# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tapendra/ellipse_detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tapendra/ellipse_detection

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/tapendra/ellipse_detection/CMakeFiles /home/tapendra/ellipse_detection/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/tapendra/ellipse_detection/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named FindEllipse

# Build rule for target.
FindEllipse: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FindEllipse
.PHONY : FindEllipse

# fast build rule for target.
FindEllipse/fast:
	$(MAKE) -f CMakeFiles/FindEllipse.dir/build.make CMakeFiles/FindEllipse.dir/build
.PHONY : FindEllipse/fast

#=============================================================================
# Target rules for targets named featurereco_lib

# Build rule for target.
featurereco_lib: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 featurereco_lib
.PHONY : featurereco_lib

# fast build rule for target.
featurereco_lib/fast:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/build
.PHONY : featurereco_lib/fast

Configuration.o: Configuration.cpp.o

.PHONY : Configuration.o

# target to build an object file
Configuration.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/Configuration.cpp.o
.PHONY : Configuration.cpp.o

Configuration.i: Configuration.cpp.i

.PHONY : Configuration.i

# target to preprocess a source file
Configuration.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/Configuration.cpp.i
.PHONY : Configuration.cpp.i

Configuration.s: Configuration.cpp.s

.PHONY : Configuration.s

# target to generate assembly for a file
Configuration.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/Configuration.cpp.s
.PHONY : Configuration.cpp.s

MedianTextReader.o: MedianTextReader.cpp.o

.PHONY : MedianTextReader.o

# target to build an object file
MedianTextReader.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/MedianTextReader.cpp.o
.PHONY : MedianTextReader.cpp.o

MedianTextReader.i: MedianTextReader.cpp.i

.PHONY : MedianTextReader.i

# target to preprocess a source file
MedianTextReader.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/MedianTextReader.cpp.i
.PHONY : MedianTextReader.cpp.i

MedianTextReader.s: MedianTextReader.cpp.s

.PHONY : MedianTextReader.s

# target to generate assembly for a file
MedianTextReader.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/MedianTextReader.cpp.s
.PHONY : MedianTextReader.cpp.s

PMTIdentified.o: PMTIdentified.cpp.o

.PHONY : PMTIdentified.o

# target to build an object file
PMTIdentified.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/PMTIdentified.cpp.o
.PHONY : PMTIdentified.cpp.o

PMTIdentified.i: PMTIdentified.cpp.i

.PHONY : PMTIdentified.i

# target to preprocess a source file
PMTIdentified.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/PMTIdentified.cpp.i
.PHONY : PMTIdentified.cpp.i

PMTIdentified.s: PMTIdentified.cpp.s

.PHONY : PMTIdentified.s

# target to generate assembly for a file
PMTIdentified.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/PMTIdentified.cpp.s
.PHONY : PMTIdentified.cpp.s

distance_to_ellipse.o: distance_to_ellipse.cpp.o

.PHONY : distance_to_ellipse.o

# target to build an object file
distance_to_ellipse.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/distance_to_ellipse.cpp.o
.PHONY : distance_to_ellipse.cpp.o

distance_to_ellipse.i: distance_to_ellipse.cpp.i

.PHONY : distance_to_ellipse.i

# target to preprocess a source file
distance_to_ellipse.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/distance_to_ellipse.cpp.i
.PHONY : distance_to_ellipse.cpp.i

distance_to_ellipse.s: distance_to_ellipse.cpp.s

.PHONY : distance_to_ellipse.s

# target to generate assembly for a file
distance_to_ellipse.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/distance_to_ellipse.cpp.s
.PHONY : distance_to_ellipse.cpp.s

ellipse.o: ellipse.cpp.o

.PHONY : ellipse.o

# target to build an object file
ellipse.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse.cpp.o
.PHONY : ellipse.cpp.o

ellipse.i: ellipse.cpp.i

.PHONY : ellipse.i

# target to preprocess a source file
ellipse.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse.cpp.i
.PHONY : ellipse.cpp.i

ellipse.s: ellipse.cpp.s

.PHONY : ellipse.s

# target to generate assembly for a file
ellipse.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse.cpp.s
.PHONY : ellipse.cpp.s

ellipse_intersection.o: ellipse_intersection.cpp.o

.PHONY : ellipse_intersection.o

# target to build an object file
ellipse_intersection.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse_intersection.cpp.o
.PHONY : ellipse_intersection.cpp.o

ellipse_intersection.i: ellipse_intersection.cpp.i

.PHONY : ellipse_intersection.i

# target to preprocess a source file
ellipse_intersection.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse_intersection.cpp.i
.PHONY : ellipse_intersection.cpp.i

ellipse_intersection.s: ellipse_intersection.cpp.s

.PHONY : ellipse_intersection.s

# target to generate assembly for a file
ellipse_intersection.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/ellipse_intersection.cpp.s
.PHONY : ellipse_intersection.cpp.s

featureFunctions.o: featureFunctions.cpp.o

.PHONY : featureFunctions.o

# target to build an object file
featureFunctions.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/featureFunctions.cpp.o
.PHONY : featureFunctions.cpp.o

featureFunctions.i: featureFunctions.cpp.i

.PHONY : featureFunctions.i

# target to preprocess a source file
featureFunctions.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/featureFunctions.cpp.i
.PHONY : featureFunctions.cpp.i

featureFunctions.s: featureFunctions.cpp.s

.PHONY : featureFunctions.s

# target to generate assembly for a file
featureFunctions.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/featureFunctions.cpp.s
.PHONY : featureFunctions.cpp.s

find_ellipses.o: find_ellipses.cpp.o

.PHONY : find_ellipses.o

# target to build an object file
find_ellipses.cpp.o:
	$(MAKE) -f CMakeFiles/FindEllipse.dir/build.make CMakeFiles/FindEllipse.dir/find_ellipses.cpp.o
.PHONY : find_ellipses.cpp.o

find_ellipses.i: find_ellipses.cpp.i

.PHONY : find_ellipses.i

# target to preprocess a source file
find_ellipses.cpp.i:
	$(MAKE) -f CMakeFiles/FindEllipse.dir/build.make CMakeFiles/FindEllipse.dir/find_ellipses.cpp.i
.PHONY : find_ellipses.cpp.i

find_ellipses.s: find_ellipses.cpp.s

.PHONY : find_ellipses.s

# target to generate assembly for a file
find_ellipses.cpp.s:
	$(MAKE) -f CMakeFiles/FindEllipse.dir/build.make CMakeFiles/FindEllipse.dir/find_ellipses.cpp.s
.PHONY : find_ellipses.cpp.s

hough_ellipse.o: hough_ellipse.cpp.o

.PHONY : hough_ellipse.o

# target to build an object file
hough_ellipse.cpp.o:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/hough_ellipse.cpp.o
.PHONY : hough_ellipse.cpp.o

hough_ellipse.i: hough_ellipse.cpp.i

.PHONY : hough_ellipse.i

# target to preprocess a source file
hough_ellipse.cpp.i:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/hough_ellipse.cpp.i
.PHONY : hough_ellipse.cpp.i

hough_ellipse.s: hough_ellipse.cpp.s

.PHONY : hough_ellipse.s

# target to generate assembly for a file
hough_ellipse.cpp.s:
	$(MAKE) -f CMakeFiles/featurereco_lib.dir/build.make CMakeFiles/featurereco_lib.dir/hough_ellipse.cpp.s
.PHONY : hough_ellipse.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... FindEllipse"
	@echo "... featurereco_lib"
	@echo "... Configuration.o"
	@echo "... Configuration.i"
	@echo "... Configuration.s"
	@echo "... MedianTextReader.o"
	@echo "... MedianTextReader.i"
	@echo "... MedianTextReader.s"
	@echo "... PMTIdentified.o"
	@echo "... PMTIdentified.i"
	@echo "... PMTIdentified.s"
	@echo "... distance_to_ellipse.o"
	@echo "... distance_to_ellipse.i"
	@echo "... distance_to_ellipse.s"
	@echo "... ellipse.o"
	@echo "... ellipse.i"
	@echo "... ellipse.s"
	@echo "... ellipse_intersection.o"
	@echo "... ellipse_intersection.i"
	@echo "... ellipse_intersection.s"
	@echo "... featureFunctions.o"
	@echo "... featureFunctions.i"
	@echo "... featureFunctions.s"
	@echo "... find_ellipses.o"
	@echo "... find_ellipses.i"
	@echo "... find_ellipses.s"
	@echo "... hough_ellipse.o"
	@echo "... hough_ellipse.i"
	@echo "... hough_ellipse.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

