#include "models/yolov5/rknn_yolov5.hpp"

#include <cstdint>
#include <cstdio>

namespace rknn_deploy {
namespace {
constexpr int kAnchor0[6] = {10, 13, 16, 30, 33, 23};
constexpr int kAnchor1[6] = {30, 61, 62, 45, 59, 119};
constexpr int kAnchor2[6] = {116, 90, 156, 198, 373, 326};
}

int rkYolov5s::initModel() {
    if (io_num_.n_output < 3) {
        std::fprintf(stderr, "[YOLOv5] expected at least 3 output tensors, got %u\n", io_num_.n_output);
        return -1;
    }
    for (int i = 0; i < 3; ++i) {
        const auto& a = output_attrs_[i];
        if (a.n_dims != 4 || a.fmt != RKNN_TENSOR_NCHW || a.type != RKNN_TENSOR_INT8) {
            std::fprintf(stderr, "[YOLOv5] output %d must be INT8 NCHW\n", i);
            return -1;
        }
        int channels = static_cast<int>(a.dims[1]);
        if (channels % 3 != 0 || channels / 3 <= 5) return -1;
        int classes = channels / 3 - 5;
        if (i == 0) num_classes_ = classes;
        if (classes != num_classes_) return -1;
    }
    std::printf("[YOLOv5] detected %d classes\n", num_classes_);
    return 0;
}

int rkYolov5s::decodeBranch(int output_index, int stride, const int anchors[6], const PreprocessMeta& meta,
                            float conf_threshold, std::vector<Detection>& detections) {
    const auto& a = output_attrs_[output_index];
    const auto* input = static_cast<const int8_t*>(outputPtr(output_index));
    if (!input) return -1;
    const int grid_h = static_cast<int>(a.dims[2]);
    const int grid_w = static_cast<int>(a.dims[3]);
    const int ws = tensorWidthStride(a, grid_w);
    const int hs = tensorHeightStride(a, grid_h);
    const int plane = ws * hs;
    const int prop = 5 + num_classes_;
    const int8_t thr = quantize_f32_to_affine(conf_threshold, a.zp, a.scale);

    for (int anchor = 0; anchor < 3; ++anchor) {
        const int channel_base = prop * anchor;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int pos = y * ws + x;
                const int8_t obj_q = input[(channel_base + 4) * plane + pos];
                if (obj_q < thr) continue;
                const float obj = dequantize_affine_to_f32(obj_q, a.zp, a.scale);

                int best_class = -1;
                int8_t best_q = INT8_MIN;
                for (int c = 0; c < num_classes_; ++c) {
                    int8_t q = input[(channel_base + 5 + c) * plane + pos];
                    if (q > best_q) { best_q = q; best_class = c; }
                }
                if (best_class < 0 || best_q < thr) continue;
                const float cls = dequantize_affine_to_f32(best_q, a.zp, a.scale);
                const float score = obj * cls;
                if (score < conf_threshold) continue;

                const float tx = dequantize_affine_to_f32(input[(channel_base + 0) * plane + pos], a.zp, a.scale);
                const float ty = dequantize_affine_to_f32(input[(channel_base + 1) * plane + pos], a.zp, a.scale);
                const float tw = dequantize_affine_to_f32(input[(channel_base + 2) * plane + pos], a.zp, a.scale);
                const float th = dequantize_affine_to_f32(input[(channel_base + 3) * plane + pos], a.zp, a.scale);

                float cx = (tx * 2.0f - 0.5f + x) * stride;
                float cy = (ty * 2.0f - 0.5f + y) * stride;
                float bw = (tw * 2.0f) * (tw * 2.0f) * anchors[anchor * 2];
                float bh = (th * 2.0f) * (th * 2.0f) * anchors[anchor * 2 + 1];

                Detection det;
                det.score = score;
                det.label_id = best_class;
                det.bbox.left = map_x(cx - bw * 0.5f, meta);
                det.bbox.top = map_y(cy - bh * 0.5f, meta);
                det.bbox.right = map_x(cx + bw * 0.5f, meta);
                det.bbox.bottom = map_y(cy + bh * 0.5f, meta);
                if (best_class < static_cast<int>(labels_.size())) det.label_name = labels_[best_class];
                if (det.bbox.Width() > 0 && det.bbox.Height() > 0) detections.push_back(std::move(det));
            }
        }
    }
    return 0;
}

int rkYolov5s::postprocess(const PreprocessMeta& meta, float conf_threshold, float nms_threshold,
                           std::vector<Detection>& detections) {
    if (decodeBranch(0, 8, kAnchor0, meta, conf_threshold, detections) != 0) return -1;
    if (decodeBranch(1, 16, kAnchor1, meta, conf_threshold, detections) != 0) return -1;
    if (decodeBranch(2, 32, kAnchor2, meta, conf_threshold, detections) != 0) return -1;
    nms_detections(detections, nms_threshold);
    return 0;
}

std::vector<Detection> rkYolov5s::detect(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    std::vector<Detection> detections;
    PreprocessMeta meta;
    if (preprocess(image, meta) != 0 || runBound() != 0) return detections;
    postprocess(meta, conf_threshold, nms_threshold, detections);
    return detections;
}

std::vector<Detection> rkYolov5s::detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                               float nms_threshold) {
    std::vector<Detection> detections;
    PreprocessMeta meta;
    if (preprocess(image, meta) != 0 || runBound() != 0) return detections;
    postprocess(meta, conf_threshold, nms_threshold, detections);
    return detections;
}

}  // namespace rknn_deploy
