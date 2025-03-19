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
#include "tool.h"
#include <random>
#include <iomanip>
#include <limits>
#include <cfloat>
#include <chrono>
#include <string>

const double EPSILON = 1e-9;

void optype(char *argv, cnnlOpTensorDesc_t &OpTensorDesc) {
  if (std::string(argv) == "CNNL_OP_TENSOR_ADD")
    OpTensorDesc = CNNL_OP_TENSOR_ADD;
  else if (std::string(argv) == "CNNL_OP_TENSOR_SUB")
    OpTensorDesc = CNNL_OP_TENSOR_SUB;
  else if (std::string(argv) == "CNNL_OP_TENSOR_MUL")
    OpTensorDesc = CNNL_OP_TENSOR_MUL;
}

void datatype(char *argv, cnnlDataType_t &DataType) {
  if (std::string(argv) == "CNNL_DTYPE_HALF")
    DataType = CNNL_DTYPE_HALF;
  else if (std::string(argv) == "CNNL_DTYPE_FLOAT")
    DataType = CNNL_DTYPE_FLOAT;
  else if (std::string(argv) == "CNNL_DTYPE_INT16")
    DataType = CNNL_DTYPE_INT16;
  else if (std::string(argv) == "CNNL_DTYPE_INT11")
    DataType = CNNL_DTYPE_INT31;
  else if (std::string(argv) == "CNNL_DTYPE_INT32")
    DataType = CNNL_DTYPE_INT32;
  else if (std::string(argv) == "CNNL_DTYPE_BOOL")
    DataType = CNNL_DTYPE_BOOL;
  else if (std::string(argv) == "CNNL_DTYPE_INVALID")
    DataType = CNNL_DTYPE_INVALID;
}

void paramCheck(int argc, int PARAM_NUM) {
  if (argc != PARAM_NUM) {
    std::string error = "command format error, please reference run_opTensor_sample.sh";
    throw std::runtime_error(error);
  }
}

int numAfterFirst(int num) {
  int tmp = 0;
  while (num) {
    num = num >> 1;
    tmp++;
  }
  return tmp - 1;
}
