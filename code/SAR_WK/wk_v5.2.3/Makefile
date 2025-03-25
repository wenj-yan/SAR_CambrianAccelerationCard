# 定义输出目录
OBJ_DIR = obj

# Target rules
all: build

build: wk

# 修改OBJ路径
OBJ = $(OBJ_DIR)/wk.o $(OBJ_DIR)/tool.o $(OBJ_DIR)/public.o \
      $(OBJ_DIR)/read.o $(OBJ_DIR)/complex.o $(OBJ_DIR)/stolt.o $(OBJ_DIR)/fft.o $(OBJ_DIR)/compression.o \
	  $(OBJ_DIR)/preprocess.o


INCLUDES := -I${NEUWARE_HOME}/include
LIBRARIES := -L${NEUWARE_HOME}/lib64
CXXFLAGS := -Og -std=c++11 -fPIC -lstdc++ -Wall
LDFLAGS := -lcnnl -lcnrt -lcndrv

# 确保输出目录存在
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

wk: $(OBJ_DIR) $(OBJ)
	${CXX} -o $@ $(OBJ) $(LIBRARIES) $(LDFLAGS)

# 修改编译规则
$(OBJ_DIR)/%.o: %.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/tool.o: tool/tool.cc tool/tool.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/public.o: public/public.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/read.o: readfile/read.cc readfile/read.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/complex.o: complex/complex.cc complex/complex.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/stolt.o: task/stolt.cc  task/task.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/fft.o: task/fft.cc  task/task.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/compression.o: task/compression.cc  task/task.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

$(OBJ_DIR)/preprocess.o: task/preprocess.cc  task/task.h
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -rf $(OBJ_DIR)
	rm -rf wk

clobber: clean