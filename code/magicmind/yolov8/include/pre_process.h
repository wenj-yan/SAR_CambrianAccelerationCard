#ifndef _SAMPLE_PRE_PROCESS_H
#define _SAMPLE_PRE_PROCESS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

std::vector<std::string> LoadImages(const std::string image_dir,
                                    const int batch_size,
                                    int image_num,
                                    const std::string file_list);
cv::Mat process_img(cv::Mat src_img,
                    int h,
                    int w,
                    bool transpose = false,
                    bool normlize  = false,
                    bool swapBR    = true,
                    int depth      = CV_8U);

#endif  //_SAMPLE_PRE_PROCESS_HPP