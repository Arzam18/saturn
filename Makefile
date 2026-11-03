# this is stupid, but whatever

ifndef EVALFILE
	EVALFILE=default.nnue
endif
ifndef EXE
	EXE=saturn
endif

all:
	mkdir -p build
	cp ${EVALFILE} build/
	cd build; EVALFILE=$(EVALFILE) && cmake ../ && cmake --build . --config Release --parallel; cp saturn ../${EXE}

clean:
	rm -r ./build
	rm ./${EXE}


.PHONY: android-arm64
ANDROID_NDK_HOME ?= $(ANDROID_HOME)/ndk/27.2.12479018

android-arm64:
	mkdir -p build-android
	cp ${EVALFILE} build-android/
	cd build-android && EVALFILE=${EVALFILE} cmake .. \
		-DCMAKE_TOOLCHAIN_FILE="$(ANDROID_NDK_HOME)/build/cmake/android.toolchain.cmake" \
		-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build build-android --config Release --parallel
	mkdir -p release
	cp build-android/saturn release/saturn-android-arm64
