// test_grid_sample.cpp
#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include <complex>
#include <iostream> 
#include <complex.h>
#include <cnrt.h>
#include <cnnl.h>

using namespace std;

#define CHECK_CNRT(call) \
    do { \
        cnrtRet_t ret = call; \
        if (ret != CNRT_RET_SUCCESS) { \
            std::cerr << "CNRT error: " << ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

#define CHECK_WORKSPACE(workspace, workspace_size) \
    if (workspace_size > 0 && workspace == nullptr) { \
        std::cerr << "Failed to allocate workspace memory" << std::endl; \
        return -1; \
    }

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));
    CNRT_CHECK(cnrtQueueCreate(&queue));
    CNNL_CHECK(cnnlCreate(&handle));
    CNNL_CHECK(cnnlSetQueue(handle, queue));
}

int main() {
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;
    initDevice(dev, queue, handle);

    // 3. 设置输入参数
    const int batch_size = 1;
    const int channels = 1;
    const int height = 8;
    const int width = 8;
    const int out_height = 8;
    const int out_width = 8;

    // 4. 创建描述符
    cnnlGridSampleDescriptor_t grid_sample_desc;
    cnnlTensorDescriptor_t input_desc, grid_desc, output_desc;
    
    CNNL_CHECK(cnnlCreateGridSampleDescriptor(&grid_sample_desc));
    CNNL_CHECK(cnnlSetGridSampleDescriptor(
        grid_sample_desc,
        CNNL_INTERP_BILINEAR,
        CNNL_GRIDSAMPLE_PADDING_ZEROS,
        true  // align_corners
    ));

    // 5. 设置tensor描述符 - 修改为NHWC布局
    int input_dims[] = {batch_size, height, width, channels};  // NHWC
    int grid_dims[] = {batch_size, out_height, out_width, 2}; // NHWC 
    int output_dims[] = {batch_size, out_height, out_width, channels}; // NHWC

    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&grid_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));

    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, input_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(grid_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, grid_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dims));

    // 获取workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetGridSampleForwardWorkspaceSize(
        handle,
        input_desc,
        grid_desc, 
        output_desc,
        &workspace_size));

    // 分配workspace内存
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CNRT(cnrtMalloc(&workspace, workspace_size));
    }
    CHECK_WORKSPACE(workspace, workspace_size);

    // 6. 准备输入数据 - 使用更简单的测试数据
    std::vector<float> host_input(batch_size * height * width * channels, 0.0f);
    std::vector<float> host_grid(batch_size * out_height * out_width * 2, 0.0f);
    std::vector<float> host_output(batch_size * out_height * out_width * channels, 0.0f);

    // 填充测试数据 - 使用简单的递增模式
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            // NHWC布局: index = h * (width * channels) + w * channels + c
            int idx = h * width * channels + w * channels;
            host_input[idx] = static_cast<float>(h + w);  // 简单的模式便于验证
        }
    }

    // 打印输入数据以验证
    std::cout << "Input data:" << std::endl;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int idx = h * width * channels + w * channels;
            std::cout << host_input[idx] << "\t";
        }
        std::cout << std::endl;
    }

    // 生成网格数据 - 90度旋转变换
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            int idx = h * out_width * 2 + w * 2;
            // 90度旋转：(x,y) -> (-y,x)
            float normalized_h = 2.0f * h / (out_height - 1) - 1.0f;
            float normalized_w = 2.0f * w / (out_width - 1) - 1.0f;
            host_grid[idx] = -normalized_h;     // x = -y
            host_grid[idx + 1] = normalized_w;  // y = x
        }
    }
    // 修改第二行第一个点的坐标为(0.85,-1)
    if (out_height > 1 && out_width > 0) {
        int idx = 1 * out_width * 2 + 0 * 2;  // 第二行(h=1)第一个点(w=0)
        host_grid[idx] = 0.85f;     // x = 0.85
        host_grid[idx + 1] = -1.0f; // y = -1
    }

    // 打印网格数据以验证
    std::cout << "\nGrid data (first 3x3 points):" << std::endl;
    for (int h = 0; h < std::min(3, out_height); h++) {
        for (int w = 0; w < std::min(3, out_width); w++) {
            int idx = h * out_width * 2 + w * 2;
            std::cout << "(" << host_grid[idx] << "," << host_grid[idx + 1] << ")\t";
        }
        std::cout << std::endl;
    }

    // 7. 分配设备内存
    void *dev_input, *dev_grid, *dev_output;
    size_t input_size = host_input.size() * sizeof(float);
    size_t grid_size = host_grid.size() * sizeof(float);
    size_t output_size = host_output.size() * sizeof(float);

    CHECK_CNRT(cnrtMalloc(&dev_input, input_size));
    CHECK_CNRT(cnrtMalloc(&dev_grid, grid_size));
    CHECK_CNRT(cnrtMalloc(&dev_output, output_size));

    // 8. 拷贝数据到设备
    CHECK_CNRT(cnrtMemcpy(dev_input, host_input.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CHECK_CNRT(cnrtMemcpy(dev_grid, host_grid.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 9. 执行GridSampleForward
    CNNL_CHECK(cnnlGridSampleForward(
        handle,
        grid_sample_desc,
        input_desc,
        dev_input,
        grid_desc,
        dev_grid,
        output_desc,
        dev_output,
        workspace,
        workspace_size
    ));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 10. 拷贝结果回主机
    CHECK_CNRT(cnrtMemcpy(host_output.data(), dev_output, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    // 11. 打印结果时也需要考虑NHWC布局
    std::cout << "\nOutput results:" << std::endl;
    for (int h = 0; h < std::min(3, out_height); h++) {
        for (int w = 0; w < std::min(3, out_width); w++) {
            int idx = h * out_width * channels + w * channels;
            std::cout << host_output[idx] << "\t";
        }
        std::cout << std::endl;
    }

    // 12. 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(grid_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
    CNNL_CHECK(cnnlDestroyGridSampleDescriptor(grid_sample_desc));
    CNNL_CHECK(cnnlDestroy(handle));

    CHECK_CNRT(cnrtFree(dev_input));
    CHECK_CNRT(cnrtFree(dev_grid));
    CHECK_CNRT(cnrtFree(dev_output));
    CHECK_CNRT(cnrtQueueDestroy(queue));

    // 在清理资源时释放workspace
    if (workspace != nullptr) {
        CHECK_CNRT(cnrtFree(workspace));
    }

    return 0;
}