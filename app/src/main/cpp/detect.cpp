#include <jni.h>
#include <string>
#include "net.h"
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "opencv2/opencv.hpp"
#include "cpu.h"
#include "benchmark.h"
#include "vector"
#include "unordered_map"
#include "detect.hpp"
using namespace std;

float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

float softmax(
        const float* src,
        float* dst,
        int length
)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static float clamp(
        float val,
        float min = 0.f,
        float max = 1280.f
)
{
    return val > min ? (val < max ? val : max) : min;
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > 0.45)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void generate_proposals(int stride,
                        const ncnn::Mat& feat_blob,
                        const float prob_threshold,
                        std::vector<Object>& objects)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_w - 4 * reg_max;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {

            const float* matat = feat_blob.channel(i).row(j);

            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++)
            {
                float score = matat[c];
                if (score > class_score)
                {
                    class_index = c;
                    class_score = score;
                }
            }
            if (class_score >= prob_threshold)
            {

                float x0 = j + 0.5f - softmax(matat + num_class, dst, 16);
                float y0 = i + 0.5f - softmax(matat + num_class + 16, dst, 16);
                float x1 = j + 0.5f + softmax(matat + num_class + 2 * 16, dst, 16);
                float y1 = i + 0.5f + softmax(matat + num_class + 3 * 16, dst, 16);

                x0 *= stride;
                y0 *= stride;
                x1 *= stride;
                y1 *= stride;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = class_score;
                objects.push_back(obj);

            }
        }
    }
}

extern "C" {
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net model;
const int target_size =640;

static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;


JNIEXPORT jboolean JNICALL
Java_com_example_yolov10NcnnAndroid_DetectNcnn_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
//    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
//    opt.use_int8_inference= true;
    opt.use_vulkan_compute = false;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    model.opt = opt;

    // init param
    {
        int ret = model.load_param(mgr, "yolov10n.ncnn.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "Detect_Ncnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = model.load_model(mgr, "yolov10n.ncnn.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "Detect_Ncnn", "load_bin failed");
            return JNI_FALSE;
        }
    }
    jclass localObjCls = env->FindClass("com/example/yolov10NcnnAndroid/DetectNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/example/yolov10NcnnAndroid/DetectNcnn;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");
    return JNI_TRUE;
}


JNIEXPORT jobjectArray JNICALL
Java_com_example_yolov10NcnnAndroid_DetectNcnn_Detect(JNIEnv *env, jobject thiz, jobject bitmap){
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    const float prob_threshold = 0.5f;
    float scale = 1.f;
    if (width > height)
    {
        scale = (float)target_size / width;
        width = target_size;
        height = height * scale;
    }
    else
    {
        scale = (float)target_size / height;
        height = target_size;
        width = width * scale;
    }

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR2RGB,width,height);

    // pad to target_size rectangle
    int wpad = target_size - width;
    int hpad = target_size - height;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    float norm_vals[] ={ 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

//    ex.set_vulkan_compute("false");

    ex.input("in0", in_pad);

    std::vector<Object> proposals;


    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<Object> objects8;
        generate_proposals(8, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("out1", out);

        std::vector<Object> objects16;
        generate_proposals(16, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("out2", out);

        std::vector<Object> objects32;
        generate_proposals(32, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // objects = proposals;
    int count = proposals.size();
    std::vector<Object> objects;
    objects.resize(count);
    for (auto& pro : proposals)
    {
        float x0 = pro.rect.x;
        float y0 = pro.rect.y;
        float x1 = pro.rect.x + pro.rect.width;
        float y1 = pro.rect.y + pro.rect.height;
        float& score = pro.prob;
        int& label = pro.label;

        x0 = (x0 - (wpad / 2)) / scale;
        y0 = (y0 - (hpad / 2)) / scale;
        x1 = (x1 - (wpad / 2)) / scale;
        y1 = (y1 - (hpad / 2)) / scale;

        x0 = clamp(x0, 0.f, info.width);
        y0 = clamp(y0, 0.f, info.height);
        x1 = clamp(x1, 0.f, info.width);
        y1 = clamp(y1, 0.f, info.height);

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        objects.push_back(obj);
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.prob > b.prob;
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }
    return jObjArray;
}
};
