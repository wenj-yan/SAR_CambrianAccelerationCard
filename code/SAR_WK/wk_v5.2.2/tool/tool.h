/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef SAMPLES_OPTENSOR_SAMPLE_TOOL_H_
#define SAMPLES_OPTENSOR_SAMPLE_TOOL_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <random>
#include "../public/public.h"

#define CHECK(condition, ...) \
  if (!(condition)) {         \
    LOG("ERROR");             \
  }

#define THRESHOLD_MSE (1e-5)

typedef enum {
  WARNING = 1,
  ERROR   = 2,
} DepCheckLevel;

struct HostTimer {
  struct timespec t0 = {0, 0};
  struct timespec t1 = {0, 0};
  double tv_nsec     = 0.0;
  double tv_sec      = 0.0;
  double tv_usec     = 0.0;
  void start() { clock_gettime(CLOCK_MONOTONIC, &t0); }
  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    tv_nsec = (double)t1.tv_nsec - (double)t0.tv_nsec;
    tv_sec  = (double)t1.tv_sec - (double)t0.tv_sec;
    tv_usec = tv_nsec / 1000 + tv_sec * 1000 * 1000;
  }
};

void optype(char *argv, cnnlOpTensorDesc_t &OpTensorDesc);

void datatype(char *argv, cnnlDataType_t &DataType);

int numAfterFirst(int num);

void paramCheck(int argc, int PARAM_NUM);

template <typename T>
static bool MSE(T *data_cpu, T *data_mlu, size_t size, bool isNull) {
  CHECK(data_cpu != NULL);
  CHECK(data_mlu != NULL);

  int num = size / sizeof(T);

  if (isNull) {
    return true;
  }

  for (int i = 0; i < num; ++i) {
    double mean = ((double)data_cpu[i] + (double)data_mlu[i]) / 2.0;
    double mse =
        pow(fabs((double)data_cpu[i] - mean), 2) + pow(fabs((double)data_mlu[i] - mean), 2) / 2.0;
    if (mse > (double)THRESHOLD_MSE) {
      std::stringstream threshold_mes;
      threshold_mes << "cpu[" << i << "] = " << (double)data_cpu[i] << ", mlu[" << i
                    << "] = " << (double)data_mlu[i];
      threshold_mes << "mse = " << mse << " exceeding THRESHOLD_MSE = " << THRESHOLD_MSE;
      LOG(threshold_mes.str());
      return false;
    }
  }

  return true;
}

template <typename T>
static bool ABSO(T *data_cpu, T *data_mlu, size_t size, bool isNull) {
  if (isNull) {
    return true;
  }
  bool temp = true;
  int num   = size / sizeof(T);
  for (int i = 0; i < num; ++i) {
    double diff = fabs(data_cpu[i] - data_mlu[i]);
    if (diff > 0.03) {
      temp = false;
      std::cout << "i:" << i << "    cpu:" << data_cpu[i] << "    "
                << "mlu:" << data_mlu[i] << "    " << diff << std::endl;
    }
  }
  return temp;
}

template <typename T>
static void computeExpandCpu(int *shapeA, int *shapeB, T *input, T *output) {
  uint64_t sizeA = 1;
  uint64_t sizeB = 1;

  for (int i = 0; i < CNNL_DIM_MAX; i++) {
    sizeA = sizeA * shapeA[i];
    sizeB = sizeB * shapeB[i];
  }
  T *tmp = (T *)malloc(sizeB * sizeof(T));
  memcpy(tmp, input, sizeA * sizeof(T));

  int leftSizeA  = 1;
  int rightSizeA = 1;
  int rightSizeB = 1;
  int E          = 1;
  int ExpandA    = 1;

  int size = CNNL_DIM_MAX;
#if 0
for (int i = size - 1; i >= 0; i--) {
    rightSizeA = rightSizeA * shapeA[i];
    rightSizeB = rightSizeB * shapeB[i];
    leftSizeA = sizeA / rightSizeA;
    leftSizeB = sizeB / rightSizeB;
    if (shapeA[i] != shapeB[i]) {
             E = shapeB[i];
          /* ExpandA = ExpandA * shapeA[i]; */
      /* shapeA[i] = shapeB[i]; */
      for (int j = 0; j < leftSizeA; j++) {
                for (int k = 0; k < E; k++) {
                  memcpy(output + j * rightSizeB + k * (rightSizeB / E),
                         tmp + j * (rightSizeB / E),
                         rightSizeB / E * sizeof(T));
                }
      }
      memcpy(tmp, output, sizeB * sizeof(T));
    }
  }
#endif

  for (int i = size - 1; i >= 0; i--) {
    rightSizeA = rightSizeA * shapeA[i];
    rightSizeB = rightSizeB * shapeB[i];
    leftSizeA  = sizeA / rightSizeA;
    if (shapeA[i] != shapeB[i]) {
      E         = shapeB[i];
      ExpandA   = ExpandA * shapeA[i];
      shapeA[i] = shapeB[i];
      for (int j = 0; j < leftSizeA; j++) {
#if 1
        int numAfter = numAfterFirst(E);
        memcpy(output + j * rightSizeB, tmp + j * (rightSizeB / E), rightSizeB / E * sizeof(T));
        for (int k = 1; k <= numAfter; k++) {
          memcpy(output + j * rightSizeB + (1 << (k - 1)) * (rightSizeB / E),
                 output + j * rightSizeB, (1 << (k - 1)) * (rightSizeB / E) * sizeof(T));
        }
        int done = 1 << numAfter;
        int rem  = E - (1 << numAfter);
        memcpy(output + j * rightSizeB + done * (rightSizeB / E), output + j * rightSizeB,
               rem * (rightSizeB / E) * sizeof(T));
#endif
#if 0
                for (int k = 0; k < E; k++) {
                    memcpy(output + j * rightSizeB + k * (rightSizeB / E),
                           tmp + j * (rightSizeB / E),
                           rightSizeB / E * sizeof(T));
                }
#endif
      }
      memcpy(tmp, output, sizeB * sizeof(T));
    }
  }
  memcpy(output, tmp, sizeB * sizeof(T));
  free(tmp);
}

#endif  // SAMPLES_OPTENSOR_SAMPLE_TOOL_H_
