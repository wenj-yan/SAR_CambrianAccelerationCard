g++ -std=c++11 -O2 -Werror -I/usr/local/include/opencv4 `pkg-config --cflags opencv4` \
    $PROJ_ROOT_PATH/src/*.cc $CPP_COMMON_PATH/*.cc \
    -I $NEUWARE_HOME/include \
    -I $PROJ_ROOT_PATH/include \
    -I $CPP_COMMON_PATH \
    -L $NEUWARE_HOME/lib64 \
    -o $PROJ_ROOT_PATH/infer \
    -lmagicmind_runtime -lcnrt -lcndrv -lgflags `pkg-config --libs opencv4`
