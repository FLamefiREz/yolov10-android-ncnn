#include <jni.h>
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"
#include "cpu.h"
#include "benchmark.h"
#include "vector"
#include "unordered_map"

struct Object
{
    int label;
    float prob;
    cv::Rect_<float> rect;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

float fast_exp(float x);

float sigmoid(float x);

float softmax(const float* src,float* dst,int length);

float intersection_area(const Object& a, const Object& b);

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

void qsort_descent_inplace(std::vector<Object>& faceobjects);

void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked);

void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);

void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, std::vector<Object>& objects,float conf);