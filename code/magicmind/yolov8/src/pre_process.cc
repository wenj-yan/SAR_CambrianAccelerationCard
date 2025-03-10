#include "pre_process.h"

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir,
                                    int batch_size,
                                    int image_num,
                                    const std::string file_list) {
  char abs_path[PATH_MAX];
  //转换为绝对路径
  if (realpath(image_dir.c_str(), abs_path) == NULL) {
    std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
    exit(1);
  }
  std::string glob_path = std::string(abs_path);
  std::ifstream in(file_list);
  std::string image_name;
  std::vector<std::string> image_paths;
  int count = 0;
  std::string image_path;
  while (getline(in, image_name)) {
    image_path = glob_path + "/" + image_name;
    image_paths.push_back(image_path);
    count += 1;
    if (count >= image_num) {
      break;
    }
  }
  return image_paths;
}

cv::Mat process_img(cv::Mat src_img,
                    int dst_h,
                    int dst_w,
                    bool transpose,
                    bool normlize,
                    bool swapBR,
                    int depth) {
  //获取原始图像的高度和宽度
  int src_h = src_img.rows;
  int src_w = src_img.cols;
  std::cout << "src_size:" << src_h <<  src_w << "dst_size: "<<dst_h <<dst_w <<std::endl;
  //计算缩放比例，保持长宽比
  float ratio = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));
  int unpad_h = std::floor(src_h * ratio);
  int unpad_w = std::floor(src_w * ratio);
  //如果进行缩放
  if (ratio != 1) {
    //选择插值方法
    int interpolation;
    if (ratio < 1) { //缩小图像
      interpolation = cv::INTER_AREA;   //区域插值
    } else {   //放大图像
      interpolation = cv::INTER_LINEAR;  //线性插值
    }
    //// 执行缩放操作
    cv::resize(src_img, src_img, cv::Size(unpad_w, unpad_h), interpolation);
  }
  //// 计算上下左右的填充量
  int pad_t = std::floor((dst_h - unpad_h) / 2);
  int pad_b = dst_h - unpad_h - pad_t;
  int pad_l = std::floor((dst_w - unpad_w) / 2);
  int pad_r = dst_w - unpad_w - pad_l;
  //// 执行填充操作，使用固定值(114,114,114)填充
  cv::copyMakeBorder(src_img, src_img, pad_t, pad_b, pad_l, pad_r, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  //归一化
  if (normlize) {
    src_img.convertTo(src_img, CV_32F);
    cv::Scalar std(0.00392, 0.00392, 0.00392);
    cv::multiply(src_img, std, src_img);
  }
  //RGB的BR转换
  if (swapBR) {
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
  }
  //转换图像深度
  if (src_img.depth() != depth) {
    src_img.convertTo(src_img, depth);
  }
  //创建输出blob进行维度转换
  cv::Mat blob;
  if (transpose) {
    int c = src_img.channels();
    int h = src_img.rows;
    int w = src_img.cols;
    int sz[] = {1, c, h, w};
    blob.create(4, sz, depth);
    cv::Mat ch[3];
    for (int j = 0; j < c; j++) {
      ch[j] = cv::Mat(src_img.rows, src_img.cols, depth, blob.ptr(0, j));
    }
    cv::split(src_img, ch);
  } else {
    blob = src_img;
  }
  std::cout << "finish pre_process" << std::endl;
  return blob;
}