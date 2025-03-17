CC=$(HIP_PATH)/bin/hipcc


CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB   --offload-arch=gfx928 -g
INCLUDES  += -I$(HIP_PATH)/include -I./include
LDFLAGS =

#获取当前目录下的cpp文件集，放在变量CUR_SOURCE中
CUR_SOURCE=${wildcard ./src/*.cpp}

#将对应的c文件名转为o文件后放在下面的CUR_OBJS变量中
CUR_OBJS=${patsubst %.cpp, %.o, $(CUR_SOURCE)}

EXECUTABLE=conv2dfp16demo


all:$(EXECUTABLE)

$(EXECUTABLE): $(CUR_OBJS)
	$(CC) $(CUR_OBJS) $(LDFLAGS) -o $(EXECUTABLE)

	
%.o:%.cpp
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
	
clean:
	rm -f $(EXECUTABLE)
	rm -f ./src/*.o
