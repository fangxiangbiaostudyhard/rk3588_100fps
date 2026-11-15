#ifndef RKNN_COMMON_HPP
#define RKNN_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "api/rknn_types.hpp"

namespace rknn_deploy {

inline unsigned char* load_file_data(FILE* fp, size_t offset, size_t size) {
    if (!fp || fseek(fp, offset, SEEK_SET) != 0) return nullptr;
    auto* data = static_cast<unsigned char*>(malloc(size));
    if (!data) return nullptr;
    if (fread(data, 1, size, fp) != size) {
        free(data);
        return nullptr;
    }
    return data;
}

inline unsigned char* load_model_file(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::fprintf(stderr, "[RKNN] open %s failed\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size <= 0 || size > INT32_MAX) {
        fclose(fp);
        return nullptr;
    }
    auto* data = load_file_data(fp, 0, static_cast<size_t>(size));
    fclose(fp);
    if (model_size) *model_size = static_cast<int>(size);
    return data;
}

inline int clamp_int(float val, int min_val, int max_val) {
    return val > min_val ? (val < max_val ? static_cast<int>(val) : max_val) : min_val;
}
inline float clamp_float(float val, float min_val, float max_val) {
    return std::max(min_val, std::min(max_val, val));
}
inline int8_t quantize_f32_to_affine(float f32, int32_t zp, float scale) {
    if (scale == 0.0f) return static_cast<int8_t>(zp);
    return static_cast<int8_t>(clamp_float((f32 / scale) + zp, -128.0f, 127.0f));
}
inline float dequantize_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

inline void compute_dfl(const float* exp_values, int dfl_len, float* box) {
    for (int b = 0; b < 4; ++b) {
        float exp_sum = 0.0f, acc_sum = 0.0f;
        for (int i = 0; i < dfl_len; ++i) {
            float val = exp_values[i + b * dfl_len];
            exp_sum += val;
            acc_sum += val * i;
        }
        box[b] = exp_sum > 0 ? acc_sum / exp_sum : 0.0f;
    }
}

inline void build_exp_lut(int32_t zp, float scale, float* lut) {
    for (int i = -128; i <= 127; ++i) {
        lut[static_cast<uint8_t>(i)] = std::exp(dequantize_affine_to_f32(static_cast<int8_t>(i), zp, scale));
    }
}

inline float map_x(float x, const PreprocessMeta& meta) {
    return clamp_float(meta.ToSourceX(x), 0.0f, static_cast<float>(meta.src_width));
}
inline float map_y(float y, const PreprocessMeta& meta) {
    return clamp_float(meta.ToSourceY(y), 0.0f, static_cast<float>(meta.src_height));
}

/**
 * INT8 YOLOv8/11 branch decoder. Supports padded width/height strides in zero-copy output tensors.
 */
inline int process_i8_branch(const int8_t* box_tensor, int32_t box_zp, float box_scale,
                             int box_w_stride, int box_h_stride,
                             const int8_t* score_tensor, int32_t score_zp, float score_scale,
                             int score_w_stride, int score_h_stride,
                             const int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                             int sum_w_stride, int sum_h_stride,
                             int grid_h, int grid_w, int stride, int dfl_len, int num_classes,
                             std::vector<Detection>& detections, float conf_threshold,
                             const PreprocessMeta& meta, const std::vector<std::string>& labels) {
    if (!box_tensor || !score_tensor || grid_h <= 0 || grid_w <= 0 || dfl_len <= 0 || num_classes <= 0) return 0;
    const int box_plane = box_w_stride * box_h_stride;
    const int score_plane = score_w_stride * score_h_stride;
    const int sum_plane = sum_w_stride * sum_h_stride;
    (void)sum_plane;

    const int8_t score_thr = quantize_f32_to_affine(conf_threshold, score_zp, score_scale);
    const int8_t sum_thr = quantize_f32_to_affine(conf_threshold, score_sum_zp, score_sum_scale);
    float exp_lut[256];
    build_exp_lut(box_zp, box_scale, exp_lut);
    std::vector<float> exp_vals(static_cast<size_t>(dfl_len) * 4);
    int valid_count = 0;

    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            const int box_offset0 = y * box_w_stride + x;
            const int score_offset0 = y * score_w_stride + x;
            const int sum_offset0 = y * sum_w_stride + x;
            if (score_sum_tensor && score_sum_tensor[sum_offset0] < sum_thr) continue;

            int max_class = -1;
            int8_t max_score = INT8_MIN;
            for (int c = 0; c < num_classes; ++c) {
                int8_t s = score_tensor[c * score_plane + score_offset0];
                if (s >= score_thr && (max_class < 0 || s > max_score)) {
                    max_score = s;
                    max_class = c;
                }
            }
            if (max_class < 0) continue;

            for (int k = 0; k < dfl_len * 4; ++k) {
                int8_t v = box_tensor[k * box_plane + box_offset0];
                exp_vals[k] = exp_lut[static_cast<uint8_t>(v)];
            }
            float box[4];
            compute_dfl(exp_vals.data(), dfl_len, box);

            float x1 = (-box[0] + x + 0.5f) * stride;
            float y1 = (-box[1] + y + 0.5f) * stride;
            float x2 = ( box[2] + x + 0.5f) * stride;
            float y2 = ( box[3] + y + 0.5f) * stride;

            Detection det;
            det.score = dequantize_affine_to_f32(max_score, score_zp, score_scale);
            det.label_id = max_class;
            det.bbox.left = map_x(x1, meta);
            det.bbox.top = map_y(y1, meta);
            det.bbox.right = map_x(x2, meta);
            det.bbox.bottom = map_y(y2, meta);
            if (max_class < static_cast<int>(labels.size())) det.label_name = labels[max_class];
            if (det.bbox.Width() > 0 && det.bbox.Height() > 0) {
                detections.push_back(std::move(det));
                ++valid_count;
            }
        }
    }
    return valid_count;
}

inline void process_single_output_fp32(const float* output_data, int dim1, int dim2, int num_classes,
                                       const PreprocessMeta& meta, float conf_threshold,
                                       const std::vector<std::string>& labels,
                                       std::vector<Detection>& detections) {
    if (!output_data || dim1 <= 0 || dim2 <= 0 || num_classes <= 0) return;
    const bool transposed = dim1 < dim2; // [C,N] vs [N,C]
    const int num_predictions = transposed ? dim2 : dim1;
    const int feat_dim = transposed ? dim1 : dim2;
    if (feat_dim < 4 + num_classes) return;

    for (int i = 0; i < num_predictions; ++i) {
        float cx, cy, w, h, max_score = -1.0f;
        int max_class = -1;
        if (transposed) {
            cx = output_data[i];
            cy = output_data[num_predictions + i];
            w  = output_data[2 * num_predictions + i];
            h  = output_data[3 * num_predictions + i];
            for (int c = 0; c < num_classes; ++c) {
                float s = output_data[(4 + c) * num_predictions + i];
                if (s > max_score) { max_score = s; max_class = c; }
            }
        } else {
            const float* pred = output_data + static_cast<size_t>(i) * feat_dim;
            cx = pred[0]; cy = pred[1]; w = pred[2]; h = pred[3];
            for (int c = 0; c < num_classes; ++c) {
                float s = pred[4 + c];
                if (s > max_score) { max_score = s; max_class = c; }
            }
        }
        if (max_score < conf_threshold || max_class < 0) continue;

        Detection det;
        det.score = max_score;
        det.label_id = max_class;
        det.bbox.left = map_x(cx - w * 0.5f, meta);
        det.bbox.top = map_y(cy - h * 0.5f, meta);
        det.bbox.right = map_x(cx + w * 0.5f, meta);
        det.bbox.bottom = map_y(cy + h * 0.5f, meta);
        if (max_class < static_cast<int>(labels.size())) det.label_name = labels[max_class];
        if (det.bbox.Width() > 0 && det.bbox.Height() > 0) detections.push_back(std::move(det));
    }
}


inline void process_single_output_i8(const int8_t* output_data, int dim1, int dim2, int num_classes,
                                     int32_t zp, float scale, const PreprocessMeta& meta,
                                     float conf_threshold, const std::vector<std::string>& labels,
                                     std::vector<Detection>& detections) {
    if (!output_data || dim1 <= 0 || dim2 <= 0 || num_classes <= 0) return;
    const bool transposed = dim1 < dim2;
    const int num_predictions = transposed ? dim2 : dim1;
    const int feat_dim = transposed ? dim1 : dim2;
    if (feat_dim < 4 + num_classes) return;
    for (int i = 0; i < num_predictions; ++i) {
        auto at = [&](int f) -> float {
            const int idx = transposed ? f * num_predictions + i : i * feat_dim + f;
            return dequantize_affine_to_f32(output_data[idx], zp, scale);
        };
        float cx = at(0), cy = at(1), w = at(2), h = at(3);
        float max_score = -1.0f;
        int max_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float sc = at(4 + c);
            if (sc > max_score) { max_score = sc; max_class = c; }
        }
        if (max_score < conf_threshold || max_class < 0) continue;
        Detection det;
        det.score = max_score;
        det.label_id = max_class;
        det.bbox.left = map_x(cx - w * 0.5f, meta);
        det.bbox.top = map_y(cy - h * 0.5f, meta);
        det.bbox.right = map_x(cx + w * 0.5f, meta);
        det.bbox.bottom = map_y(cy + h * 0.5f, meta);
        if (max_class < static_cast<int>(labels.size())) det.label_name = labels[max_class];
        if (det.bbox.Width() > 0 && det.bbox.Height() > 0) detections.push_back(std::move(det));
    }
}

inline void nms_detections(std::vector<Detection>& detections, float threshold) {
    if (detections.empty()) return;
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<uint8_t> suppressed(detections.size(), 0);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].label_id != detections[j].label_id) continue;
            if (CalculateIoU(detections[i].bbox, detections[j].bbox) > threshold) suppressed[j] = 1;
        }
    }
    size_t write = 0;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i]) detections[write++] = std::move(detections[i]);
    }
    detections.resize(write);
}

}  // namespace rknn_deploy
#endif
