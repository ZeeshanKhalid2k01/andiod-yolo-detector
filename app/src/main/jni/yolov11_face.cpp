// YOLOv11 Face Detection - single class (face) detector
// Based on YOLOv8_det detection logic, simplified for face-only output.

#include "yolov8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>

// Box color from yolov8ncnn.cpp (stored as BGR)
extern int g_box_r;
extern int g_box_g;
extern int g_box_b;

// Label visibility from yolov8ncnn.cpp
extern bool g_show_labels;

// Only 1 class: face
static const char* face_class = "face";

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    {
        if (left < j) qsort_descent_inplace(objects, left, j);
        if (i < right) qsort_descent_inplace(objects, i, right);
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // should be 1 for face

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);
        pred_row_offset += num_grid;
    }
}

int YOLOv11_face::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;
    const float prob_threshold = 0.55f;
    const float nms_threshold = 0.35f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox: fit inside target_size x target_size square
    float scale = (float)target_size / std::max(img_w, img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, new_w, new_h);

    int wpad = target_size - new_w;
    int hpad = target_size - new_h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
        "in_pad: w=%d h=%d, img: w=%d h=%d, scale=%.4f, wpad=%d hpad=%d",
        in_pad.w, in_pad.h, img_w, img_h, scale, wpad, hpad);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    // Input
    ex.input("in0", in_pad);

    // Output
    ncnn::Mat out;
    int ret = ex.extract("out0", out);
    if (ret != 0)
        return -1;
    if (out.empty())
        return 0;

    __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
        "out: c=%d h=%d w=%d dims=%d", out.c, out.h, out.w, out.dims);

    std::vector<Object> proposals;

    if (out.h == 65 || (out.c == 1 && out.h > 10 && out.w > 1000))
    {
        // FORMAT A: DFL format — [1, 65, N] or out.h==65
        // (64 reg_max + 1 class, anchors in w dimension)
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "Using FORMAT A (DFL decode)");
        generate_proposals(out, strides, in_pad, prob_threshold, proposals);
    }
    else if (out.h == 5 || out.h == 6)
    {
        // FORMAT B: Simple decode [5/6, N]
        // Row 0=cx, 1=cy, 2=w, 3=h, 4=conf (already sigmoid-applied, in [0,1])
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "Using FORMAT B [h=%d, w=%d]", out.h, out.w);
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "FORMAT B in_pad: w=%d h=%d", in_pad.w, in_pad.h);

        // ── FORMAT B diagnostics ─────────────────────────────────────────────

        // Anchor 0 geometry: what do cx/cy/w/h actually look like?
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "B anchor0 geom: cx=%.6f cy=%.6f w=%.6f h=%.6f",
            out.row(0)[0], out.row(1)[0], out.row(2)[0], out.row(3)[0]);

        // First 10 raw conf values — are they logits or already-sigmoid probabilities?
        // Logits: expect large negatives (-8..-3) for background, large positives for face.
        // Probabilities: expect values near 0.0 for background, near 1.0 for face.
        if (out.w >= 10)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "B conf[0..4]:  %.6f %.6f %.6f %.6f %.6f",
                out.row(4)[0], out.row(4)[1], out.row(4)[2],
                out.row(4)[3], out.row(4)[4]);
            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "B conf[5..9]:  %.6f %.6f %.6f %.6f %.6f",
                out.row(4)[5], out.row(4)[6], out.row(4)[7],
                out.row(4)[8], out.row(4)[9]);
        }

        // Middle anchors (1000-1009) — different spatial region of the grid
        if (out.w >= 1010)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "B conf[1000..1004]: %.6f %.6f %.6f %.6f %.6f",
                out.row(4)[1000], out.row(4)[1001], out.row(4)[1002],
                out.row(4)[1003], out.row(4)[1004]);
            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "B conf[1005..1009]: %.6f %.6f %.6f %.6f %.6f",
                out.row(4)[1005], out.row(4)[1006], out.row(4)[1007],
                out.row(4)[1008], out.row(4)[1009]);
        }

        // Max conf value and its anchor index across all anchors
        float max_conf = -FLT_MAX;
        int   max_conf_idx = -1;
        for (int i = 0; i < out.w; i++)
        {
            float v = out.row(4)[i];
            if (v > max_conf) { max_conf = v; max_conf_idx = i; }
        }
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "B max_conf=%.6f at anchor_idx=%d  total_anchors=%d",
            max_conf, max_conf_idx, out.w);

        // Count anchors passing threshold (values used directly, no extra sigmoid)
        int pass_count = 0;
        for (int i = 0; i < out.w; i++)
        {
            if (out.row(4)[i] >= prob_threshold) pass_count++;
        }
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "B anchors passing threshold %.4f: %d / %d",
            prob_threshold, pass_count, out.w);

        // ────────────────────────────────────────────────────────────────────

        int num_anchors = out.w;
        for (int i = 0; i < num_anchors; i++)
        {
            // Model already applies sigmoid internally; row(4) is a probability in [0,1].
            // Do NOT apply sigmoid again — that was causing ~850 false positives.
            float conf = out.row(4)[i];
            // Anchors 1600-2099 are padding/garbage filled with exactly 0.5 (sigmoid(0)).
            // Skip them before the threshold check to avoid false proposals.
            if (fabsf(conf - 0.5f) < 1e-6f)
                continue;
            if (conf < prob_threshold)
                continue;

            float cx = out.row(0)[i];
            float cy = out.row(1)[i];
            float bw = out.row(2)[i];
            float bh = out.row(3)[i];

            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "B raw anchor[%d]: cx=%.4f cy=%.4f bw=%.4f bh=%.4f conf=%.4f",
                i, cx, cy, bw, bh, conf);

            // Reject anchors outside valid padded-input bounds
            if (cx < 0 || cx >= in_pad.w || cy < 0 || cy >= in_pad.h)
                continue;
            // Reject boxes wider/taller than 60% of padded input (filters coarse stride-16)
            if (bw > in_pad.w * 0.85f || bh > in_pad.h * 0.85f)
                continue;

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = 0;
            obj.prob = conf;
            proposals.push_back(obj);
        }

        // Debug: log first 3 proposals after rect conversion
        for (int dbg = 0; dbg < std::min(3, (int)proposals.size()); dbg++) {
            __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
                "prop[%d]: x=%.1f y=%.1f w=%.1f h=%.1f prob=%.3f",
                dbg,
                proposals[dbg].rect.x, proposals[dbg].rect.y,
                proposals[dbg].rect.width, proposals[dbg].rect.height,
                proposals[dbg].prob);
        }
    }
    else if (out.w == 5 || out.w == 6)
    {
        // FORMAT C: Transposed [N, 5/6]
        // Each row is (cx, cy, w, h, conf)
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "Using FORMAT C [h=%d, w=%d]", out.h, out.w);
        int num_anchors = out.h;
        for (int i = 0; i < num_anchors; i++)
        {
            const float* row = out.row(i);
            // Already sigmoid-applied by the model; use directly.
            float conf = row[4];
            if (conf < prob_threshold)
                continue;

            float cx = row[0];
            float cy = row[1];
            float bw = row[2];
            float bh = row[3];

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = 0;
            obj.prob = conf;
            proposals.push_back(obj);
        }
    }
    else if (out.dims == 3 && (out.c == 5 || out.c == 6))
    {
        // FORMAT D: 3D layout [c=5/6, h, w] — same values as B but in channels
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "Using FORMAT D [c=%d, h=%d, w=%d]", out.c, out.h, out.w);
        int num_anchors = out.h * out.w;
        for (int i = 0; i < num_anchors; i++)
        {
            // Already sigmoid-applied by the model; use directly.
            float conf = ((const float*)out.channel(4))[i];
            if (conf < prob_threshold)
                continue;

            float cx = ((const float*)out.channel(0))[i];
            float cy = ((const float*)out.channel(1))[i];
            float bw = ((const float*)out.channel(2))[i];
            float bh = ((const float*)out.channel(3))[i];

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = 0;
            obj.prob = conf;
            proposals.push_back(obj);
        }
    }
    else
    {
        // Unknown format — log and return safely (NO CRASH)
        __android_log_print(ANDROID_LOG_WARN, "YOLOv11face",
            "Unknown output format: c=%d h=%d w=%d dims=%d",
            out.c, out.h, out.w, out.dims);
        return 0;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
        "proposals before NMS: %d", (int)proposals.size());

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
        "NMS: proposals=%d  picked=%d  nms_threshold=%.2f",
        (int)proposals.size(), count, nms_threshold);
    // Debug: log first 3 picked detections
    for (int dbg = 0; dbg < std::min(3, count); dbg++) {
        const Object& o = proposals[picked[dbg]];
        __android_log_print(ANDROID_LOG_DEBUG, "YOLOv11face",
            "picked[%d]: x=%.1f y=%.1f w=%.1f h=%.1f prob=%.3f",
            dbg, o.rect.x, o.rect.y, o.rect.width, o.rect.height, o.prob);
    }

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // Convert padded-input coords → original image coords.
        // ncnn::copy_make_border added wpad/2 on the left and hpad/2 on the top,
        // so we subtract those offsets then divide by the letterbox scale.
        // x1-x0 = rect.width/scale, y1-y0 = rect.height/scale (padding cancels).
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width  - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)(img_w)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOv11_face::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    // Use user-selected box color (BGR order for OpenCV)
    cv::Scalar box_color(g_box_b, g_box_g, g_box_r);

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        // Draw bounding box
        cv::rectangle(rgb, obj.rect, box_color, 2);

        if (g_show_labels)
        {
            char label[64];
            sprintf(label, "face %.0f%%", obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int lx = obj.rect.x;
            int ly = obj.rect.y - label_size.height - baseLine;
            if (ly < 0) ly = 0;
            if (lx + label_size.width > rgb.cols)
                lx = rgb.cols - label_size.width;

            // Dark background for label
            cv::rectangle(rgb,
                cv::Rect(lx, ly, label_size.width, label_size.height + baseLine),
                cv::Scalar(0, 0, 0), -1);

            // White text
            cv::putText(rgb, label, cv::Point(lx, ly + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }

    return 0;
}
