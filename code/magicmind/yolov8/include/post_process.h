#ifndef _SAMPLE_POST_PROCESS_H
#define _SAMPLE_POST_PROCESS_H

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

std::map<int, std::string> load_name(std::string name_map_file);

bool post_process(cv::Mat &img,
                  std::vector<std::vector<float>> results,
                  std::map<int, std::string> name_map,
                  const std::string name,
                  const std::string output_dir,
                  bool save_img,
                  float dst_h,
                  float dst_w);

#endif