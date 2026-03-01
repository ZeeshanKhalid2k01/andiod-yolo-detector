// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov8.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    // FPS is now tracked via g_avg_fps for Java overlay — no native drawing
    return 0;
}

// Global FPS value readable from JNI
static float g_avg_fps = 0.f;

// Global box color (stored as BGR for OpenCV) — settable from Java via setBoxColor
int g_box_r = 100;
int g_box_g = 255;
int g_box_b = 0;  // default green (BGR: 0, 255, 100)

// Global label visibility flag — settable from Java via setShowLabels
bool g_show_labels = true;

static YOLOv8* g_yolov8 = 0;
static bool g_model_ready = false;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // yolov8
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolov8 && g_model_ready)
        {
            std::vector<Object> objects;
            g_yolov8->detect(rgb, objects);

            g_yolov8->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    // Compute FPS and store in global for Java overlay (no native drawing)
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 != 0.f)
        {
            float fps = 1000.f / (t1 - t0);
            for (int i = 9; i >= 1; i--)
                fps_history[i] = fps_history[i - 1];
            fps_history[0] = fps;

            if (fps_history[9] != 0.f)
            {
                float sum = 0.f;
                for (int i = 0; i < 10; i++)
                    sum += fps_history[i];
                g_avg_fps = sum / 10.f;
            }
        }
        t0 = t1;
    }
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        g_model_ready = false;
        delete g_yolov8;
        g_yolov8 = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    // taskid 0-5 = yolov8 tasks, 6 = yolov11 face
    // modelid 0-8 for yolov8 (3 sizes x 3 res), 0-11 for face (4 sizes x 3 res)
    int max_modelid = (taskid == 6) ? 11 : 8;
    if (taskid < 0 || taskid > 6 || modelid < 0 || modelid > max_modelid || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* tasknames[7] =
    {
        "",
        "_oiv7",
        "_seg",
        "_pose",
        "_cls",
        "_obb",
        "_face"
    };

    const char* modeltypes[9] =
    {
        "n",
        "s",
        "m",
        "n",
        "s",
        "m",
        "n",
        "s",
        "m"
    };

    std::string parampath;
    std::string modelpath;

    if (taskid == 6)
    {
        // YOLOv11 face: modelid = resolution * 4 + modelSize
        // modelSize: 0=n, 1=s, 2=m, 3=l  resolution: 0=320, 1=480, 2=640
        const char* face_sizes[4] = { "n", "s", "m", "l" };
        const char* face_res[3] = { "320", "480", "640" };
        int faceModelSize = (int)modelid % 4;
        int faceResolution = (int)modelid / 4;
        if (faceModelSize > 3) faceModelSize = 0;
        if (faceResolution > 2) faceResolution = 0;
        parampath = std::string("yolov11") + face_sizes[faceModelSize] + "_face_" + face_res[faceResolution] + ".ncnn.param";
        modelpath = std::string("yolov11") + face_sizes[faceModelSize] + "_face_" + face_res[faceResolution] + ".ncnn.bin";
    }
    else
    {
        parampath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
        modelpath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";
    }
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        // Mark model not ready while reloading
        g_model_ready = false;

        {
            // Always delete and rebuild g_yolov8.
            // The GPU instance is always destroyed and recreated below, so any
            // existing model holds stale Vulkan handles (VkPipeline, VkFence, etc.)
            // for the old VkDevice.  Reusing it causes vkWaitForFences to hang on
            // the destroyed device, which blocks on_image_render forever → blank screen.
            delete g_yolov8;
            g_yolov8 = 0;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (taskid == 0) g_yolov8 = new YOLOv8_det_coco;
            if (taskid == 1) g_yolov8 = new YOLOv8_det_oiv7;
            if (taskid == 2) g_yolov8 = new YOLOv8_seg;
            if (taskid == 3) g_yolov8 = new YOLOv8_pose;
            if (taskid == 4) g_yolov8 = new YOLOv8_cls;
            if (taskid == 5) g_yolov8 = new YOLOv8_obb;
            if (taskid == 6) g_yolov8 = new YOLOv11_face;

            g_yolov8->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            int target_size = 320;
            if (taskid == 6)
            {
                // face: modelid = resolution * 4 + modelSize
                int faceRes = (int)modelid / 4;
                if (faceRes == 1) target_size = 480;
                if (faceRes == 2) target_size = 640;
            }
            else
            {
                if ((int)modelid >= 3)
                    target_size = 480;
                if ((int)modelid >= 6)
                    target_size = 640;
            }
            g_yolov8->set_det_target_size(target_size);

            // Model is now fully loaded and ready
            g_model_ready = true;
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native float getFps();
JNIEXPORT jfloat JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getFps(JNIEnv* env, jobject thiz)
{
    return g_avg_fps;
}

// public native void setBoxColor(int r, int g, int b);
JNIEXPORT void JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setBoxColor(JNIEnv* env, jobject thiz, jint r, jint g, jint b)
{
    // Java passes RGB — swap R and B for OpenCV BGR storage
    g_box_r = (int)b;  // store B in r slot
    g_box_g = (int)g;  // G stays same
    g_box_b = (int)r;  // store R in b slot
}

// public native void setShowLabels(boolean show);
JNIEXPORT void JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setShowLabels(JNIEnv* env, jobject thiz, jboolean show)
{
    g_show_labels = (bool)show;
}

}
