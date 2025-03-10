# 开始运行本仓库前,先检查数据集路径是否存在
# 若不存在则根据您的实际路径修改
export COCO_DATASETS_PATH=/root/myproject/magicmind/data   #数据集路径
if [ -z ${COCO_DATASETS_PATH} ] || [ ! -d ${COCO_DATASETS_PATH} ];then
    echo "Error: COCO_DATASETS_PATH is not found, please set it and export it to env!"
fi

export NEUWARE_HOME=/usr/local/neuware #neuware软件栈路径
export MM_RUN_PATH=${NEUWARE_HOME}/bin #magicmind运行路径
#本sample工作路径
export PROJ_ROOT_PATH=$(pwd)
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models #模型路径

# CPP公共接口路径
export CPP_COMMON_PATH=$PROJ_ROOT_PATH/common
#相关依赖
export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
