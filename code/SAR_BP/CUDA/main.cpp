#include "BPA_CUDA.h"

int main() {
    try {
        SAR_Processor processor;
        
        // 处理SAR图像
        processor.processImage();
        
        // 保存结果
        processor.saveResult("sar_image.bin");
        
        std::cout << "SAR processing completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}