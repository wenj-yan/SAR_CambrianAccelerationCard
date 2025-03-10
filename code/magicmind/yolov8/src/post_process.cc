#include "post_process.h"
#include "utils.hpp"

std::map<int, std::string> load_name(std::string name_map_file) {
  if (!check_file_exist(name_map_file)) {
    std::cout << "coco_name file: " + name_map_file + " does not exist.\n";
    exit(0);
  }
  std::cout << "finish check" << std::endl;
  std::map<int, std::string> coco_name_map;
  std::ifstream in(name_map_file);
  if (!in) {
    std::cout << "failed to load coco_name file: " + name_map_file + ".\n";
    exit(0);
  }
  std::cout << "finish check2" ;
  std::string line;
  int index = 0;
  while (getline(in, line)) {
    coco_name_map[index] = line;
    index += 1;
  }
  std::cout << "finish load_name" ;
  return coco_name_map;
}
// 计算两个边界框的IOU
float calculate_iou(const std::vector<float>& box1, const std::vector<float>& box2) {
  float x1 = std::max(box1[2], box2[2]);
  float y1 = std::max(box1[3], box2[3]);
  float x2 = std::min(box1[4], box2[4]);
  float y2 = std::min(box1[5], box2[5]);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float box1_area = (box1[4] - box1[2]) * (box1[5] - box1[3]);
  float box2_area = (box2[4] - box2[2]) * (box2[5] - box2[3]);
  
  return intersection / (box1_area + box2_area - intersection);
}

// NMS函数
std::vector<std::vector<float>> nms(std::vector<std::vector<float>>& boxes, float iou_threshold) {
  std::vector<std::vector<float>> selected_boxes;
  if (boxes.empty()) return selected_boxes;

  // 按置信度降序排序
  std::sort(boxes.begin(), boxes.end(), 
      [](const std::vector<float>& a, const std::vector<float>& b) {
          return a[1] > b[1];
      });

  std::vector<bool> is_suppressed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); ++i) {
      if (is_suppressed[i]) continue;

      selected_boxes.push_back(boxes[i]);

      // 抑制其他重叠框
      for (size_t j = i + 1; j < boxes.size(); ++j) {
          if (is_suppressed[j]) continue;
          
          // 只对同类别的框进行NMS
          if (boxes[i][0] == boxes[j][0]) {
              if (calculate_iou(boxes[i], boxes[j]) > iou_threshold) {
                  is_suppressed[j] = true;
              }
          }
      }
  }
  
  return selected_boxes;
}
// YOLOv8 后处理函数
bool post_process(cv::Mat &img,
                  std::vector<std::vector<float>> results,
                  std::map<int, std::string> name_map,
                  const std::string name,
                  const std::string output_dir,
                  bool save_img,
                  float dst_h,
                  float dst_w) {
    results = nms(results, 0.45f);
    std::string filename = output_dir + "/" + name + ".txt";
    std::ofstream file_map(filename);
    int src_h = img.rows;
    int src_w = img.cols;
    float ratio = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));
    float scale_w = ratio * src_w;
    float scale_h = ratio * src_h;
    int detect_num = results.size();
    for (int i = 0; i < detect_num; ++i) {
        int detect_class = static_cast<int>(results[i][0]);  // 类别索引
        float score = results[i][1];  // 置信度
        float xmin = results[i][2];   // xmin
        float ymin = results[i][3];   // ymin
        float xmax = results[i][4];   // xmax
        float ymax = results[i][5];   // ymax

        // 坐标转换
        xmin = std::max(float(0.0), std::min(xmin, dst_w));
        xmax = std::max(float(0.0), std::min(xmax, dst_w));
        ymin = std::max(float(0.0), std::min(ymin, dst_h));
        ymax = std::max(float(0.0), std::min(ymax, dst_h));
        xmin = (xmin - (dst_w - scale_w) / 2) / ratio;
        ymin = (ymin - (dst_h - scale_h) / 2) / ratio;
        xmax = (xmax - (dst_w - scale_w) / 2) / ratio;
        ymax = (ymax - (dst_h - scale_h) / 2) / ratio;
        xmin = std::max(0.0f, float(xmin));
        xmax = std::max(0.0f, float(xmax));
        ymin = std::max(0.0f, float(ymin));
        ymax = std::max(0.0f, float(ymax));

        // 写入文件
        file_map << name_map[detect_class] << "," << score << "," << xmin << "," << ymin << "," << xmax
                 << "," << ymax << "\n";

        // 绘制框和类别
        if (save_img) {
            cv::rectangle(img, cv::Rect(cv::Point(int(xmin), int(ymin)), cv::Point(int(xmax), int(ymax))),
                          cv::Scalar(0, 255, 0));
            auto fontface = cv::FONT_HERSHEY_TRIPLEX;
            double fontscale = 0.5;
            int thickness = 1;
            int baseline = 0;
            std::string text = name_map[detect_class] + ": " + std::to_string(score);
            cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
            cv::putText(img, text, cv::Point(int(xmin), int(ymin) + text_size.height), fontface,
                        fontscale, cv::Scalar(255, 255, 255), thickness);
        }
    }
    if (save_img) {
        imwrite(output_dir + "/" + name + ".jpg", img);
    }
    file_map.close();
    return true;
}