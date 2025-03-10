#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>

#include "pre_process.h"
#include "post_process.h"
#include "../common/utils.hpp"
#include "../common/model_runner.h"

using namespace magicmind;
//./infer --magicmind_model=./model/model --image_dir=./data --output_dir=./output --save_img=true
DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");  //"./../../../datasets/coco/test";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "coco_file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch_size, 1, "The batch size");

/**
 * @brief get detection box from model's output
 *        see mm_network.h IDetectionOutputNode for details
 * @param[out] results
 * @param[in] detection_num
 * @param[in] data_ptr
 */
void Yolov5GetBox(std::vector<std::vector<float>> &results, int detection_num, float *data_ptr) {
  for (int i = 0; i < detection_num; ++i) {
    std::vector<float> result;
    float class_idx = *(data_ptr + 7 * i + 1);
    float score = *(data_ptr + 7 * i + 2);
    float xmin = *(data_ptr + 7 * i + 3);
    float ymin = *(data_ptr + 7 * i + 4);
    float xmax = *(data_ptr + 7 * i + 5);
    float ymax = *(data_ptr + 7 * i + 6);

    result.push_back(class_idx);
    result.push_back(score);
    result.push_back(xmin);
    result.push_back(ymin);
    result.push_back(xmax);
    result.push_back(ymax);
    results.push_back(result);
  }
}

// YOLOv8 的框解码函数
void Yolov8DecodeBox(float* output, int num_classes, int num_boxes, std::vector<std::vector<float>>& results, float conf_threshold = 0.5, float nms_threshold = 0.5) {
    for (int i = 0; i < num_boxes; ++i) {// row * cols + i
        float x = output[0+i];  // 中心点 x
        float y = output[num_boxes+i];  // 中心点 y
        float w = output[2*num_boxes+i];  // 宽度
        float h = output[3*num_boxes+i];  // 高度
        // 转换为左上角和右下角坐标
        float xmin = x - w / 2;
        float ymin = y - h / 2;
        float xmax = x + w / 2;
        float ymax = y + h / 2;

        // 获取类别概率起始位置
        float* class_scores = output + (4 * num_boxes) + i;
        
        // 找出最大概率的类别
        float max_score = 0.0f;
        int max_class_id = -1;
        for(int c = 0; c < num_classes; ++c) {
            float score = class_scores[c * num_boxes];
            if(score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // 如果最大概率大于阈值，保存这个检测框
        if (max_score > conf_threshold) {
            std::vector<float> detection = {
                static_cast<float>(max_class_id), // 类别ID
                max_score,                        // 置信度
                xmin,                            // 左上角x
                ymin,                            // 左上角y
                xmax,                            // 右下角x
                ymax                             // 右下角y
            };
            results.push_back(detection);
    }
}
}

void process_yolov8_output(float* host_output_ptr, int num_classes, int num_boxes, cv::Mat& img, std::map<int, std::string> name_map, const std::string name, const std::string output_dir, bool save_img, float dst_h, float dst_w) {
    std::vector<std::vector<float>> results;
    Yolov8DecodeBox(host_output_ptr, num_classes, num_boxes, results);
    post_process(img, results, name_map, name, output_dir, save_img, dst_h, dst_w);
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // create an instance of ModelRunner
  auto yolov8_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);   //初始化mlu设备，以及模型路径
  //推理初始化，包括engine、input、context等等
  if (!yolov8_runner->Init(FLAGS_batch_size)) {  
    SLOG(ERROR) << "Init yolov8 runnner failed.";
    return false;
  }

  // load image加载图像
  std::cout << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths =
      LoadImages(FLAGS_image_dir, FLAGS_batch_size, FLAGS_image_num, FLAGS_file_list);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  size_t rem_image_num = image_num % FLAGS_batch_size;
  SLOG(INFO) << "Total images : " << image_num;
  // load label
  std::cout << "read to load name" << std::endl;
  std::map<int, std::string> name_map = load_name(FLAGS_label_path);

  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;

  // allocate host memory for batch preprpcessed data
  auto batch_data = yolov8_runner->GetHostInputData();

  // one batch input data addr offset
  int batch_image_offset = yolov8_runner->GetInputSizes()[0] / FLAGS_batch_size;
  std::cout << "batch_image_offset :"<< batch_image_offset << std::endl;

  auto input_dim = yolov8_runner->GetInputDims()[0];
  
  int h = input_dim[2];
  int w = input_dim[3];

  SLOG(INFO) << "Start run...";
  for (int i = 0; i < image_num; i++) {
    std::string image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
    std::cout << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/" << image_num
              << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    std::cout << "image_path:" << image_paths[i] << std::endl;
    if (img.empty()) {
    std::cerr << "Error: Failed to read image at path: " << image_paths[i] << std::endl;
}
    cv::Mat img_pro = process_img(img, h, w,true,true,true,CV_32F);
    //cv::imwrite("pre_process.jpg", img_pro);
    batch_image_name.push_back(image_name);
    batch_image.push_back(img);

    // batching preprocessed data
    memcpy((u_char *)(batch_data[0]) + batch_counter * batch_image_offset, img_pro.data,
           batch_image_offset);
    std::cout << "finish memcpy" << std::endl;
    batch_counter += 1;
    // image_num may not be divisible by FLAGS_batch.
    // real_batch_size records number of images in every loop, real_batch_size may change in the
    // last loop.
    size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
    if (batch_counter % real_batch_size == 0) {
      // copy in
      yolov8_runner->H2D();
      std::cout << "finish H2D" << std::endl;
      // compute
      yolov8_runner->Run(FLAGS_batch_size);
      std::cout << "finish RUN" << std::endl;
      // copy out
      yolov8_runner->D2H();
      std::cout << "finish D2H" << std::endl;
      // get model's output addr in host
      auto host_output_ptr = yolov8_runner->GetHostOutputData();
      auto output = (float *)host_output_ptr[0];
      int num_classes = 80;  // COCO dataset
      int num_boxes = 8400;
      std::cout << "ready to process output" << std::endl;
      process_yolov8_output(output, num_classes, num_boxes, img, name_map, std::to_string(i), "./output", true, 640, 640);

      batch_counter = 0;
      batch_image.clear();
      batch_image_name.clear();
    }
  }
  // destroy resource
  yolov8_runner->Destroy();
  return 0;
}