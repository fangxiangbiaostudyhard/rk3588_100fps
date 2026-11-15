#include "models/yolov11/rknn_yolov11.hpp"

#include <cstdio>

namespace rknn_deploy {

int rkYolov11::initModel() {
    if (io_num_.n_output == 1) {
        const auto& a = output_attrs_[0];
        if (a.n_dims < 3) return -1;
        int dim1 = static_cast<int>(a.dims[a.n_dims - 2]);
        int dim2 = static_cast<int>(a.dims[a.n_dims - 1]);
        int feat_dim = std::min(dim1, dim2);
        num_classes_ = feat_dim - 4;
        multi_output_ = false;
    } else if (io_num_.n_output == 6 || io_num_.n_output == 9) {
        multi_output_ = true;
        output_per_branch_ = static_cast<int>(io_num_.n_output / 3);
        const auto& score = output_attrs_[1];
        if (score.n_dims != 4 || score.fmt != RKNN_TENSOR_NCHW) {
            std::fprintf(stderr, "[YOLOv11] multi-output zero-copy currently requires NCHW score tensors\n");
            return -1;
        }
        num_classes_ = static_cast<int>(score.dims[1]);
        for (int b = 0; b < 3; ++b) {
            const auto& box = output_attrs_[b * output_per_branch_];
            const auto& cls = output_attrs_[b * output_per_branch_ + 1];
            if (box.n_dims != 4 || cls.n_dims != 4 || box.fmt != RKNN_TENSOR_NCHW || cls.fmt != RKNN_TENSOR_NCHW)
                return -1;
            if (static_cast<int>(cls.dims[1]) != num_classes_) return -1;
            if (box.dims[1] % 4 != 0) return -1;
        }
    } else {
        std::fprintf(stderr, "[YOLOv11] unsupported output count %u (expected 1, 6 or 9)\n", io_num_.n_output);
        return -1;
    }
    if (num_classes_ <= 0) return -1;
    std::printf("[YOLOv11] detected %d classes, outputs=%u\n", num_classes_, io_num_.n_output);
    return 0;
}

int rkYolov11::postprocess(const PreprocessMeta& meta, float conf_threshold, float nms_threshold,
                           std::vector<Detection>& detections) {
    if (!multi_output_) {
        const auto& a = output_attrs_[0];
        int dim1 = static_cast<int>(a.dims[a.n_dims - 2]);
        int dim2 = static_cast<int>(a.dims[a.n_dims - 1]);
        if (a.type == RKNN_TENSOR_FLOAT32) {
            process_single_output_fp32(static_cast<const float*>(outputPtr(0)), dim1, dim2, num_classes_, meta,
                                       conf_threshold, labels_, detections);
        } else if (a.type == RKNN_TENSOR_INT8) {
            process_single_output_i8(static_cast<const int8_t*>(outputPtr(0)), dim1, dim2, num_classes_, a.zp, a.scale,
                                     meta, conf_threshold, labels_, detections);
        } else {
            std::fprintf(stderr, "[YOLOv11] unsupported single-output type %d\n", a.type);
            return -1;
        }
    } else {
        for (int b = 0; b < 3; ++b) {
            int box_idx = b * output_per_branch_;
            int score_idx = box_idx + 1;
            int sum_idx = output_per_branch_ == 3 ? box_idx + 2 : -1;
            const auto& ba = output_attrs_[box_idx];
            const auto& sa = output_attrs_[score_idx];
            if (ba.type != RKNN_TENSOR_INT8 || sa.type != RKNN_TENSOR_INT8) return -1;
            int grid_h = static_cast<int>(ba.dims[2]);
            int grid_w = static_cast<int>(ba.dims[3]);
            int stride = height_ / grid_h;
            int dfl_len = static_cast<int>(ba.dims[1]) / 4;

            const int8_t* sum_ptr = nullptr;
            int32_t sum_zp = 0;
            float sum_scale = 1.0f;
            int sum_ws = grid_w, sum_hs = grid_h;
            if (sum_idx >= 0) {
                const auto& ua = output_attrs_[sum_idx];
                sum_ptr = static_cast<const int8_t*>(outputPtr(sum_idx));
                sum_zp = ua.zp;
                sum_scale = ua.scale;
                sum_ws = tensorWidthStride(ua, grid_w);
                sum_hs = tensorHeightStride(ua, grid_h);
            }

            process_i8_branch(static_cast<const int8_t*>(outputPtr(box_idx)), ba.zp, ba.scale,
                              tensorWidthStride(ba, grid_w), tensorHeightStride(ba, grid_h),
                              static_cast<const int8_t*>(outputPtr(score_idx)), sa.zp, sa.scale,
                              tensorWidthStride(sa, grid_w), tensorHeightStride(sa, grid_h),
                              sum_ptr, sum_zp, sum_scale, sum_ws, sum_hs,
                              grid_h, grid_w, stride, dfl_len, num_classes_, detections, conf_threshold,
                              meta, labels_);
        }
    }
    nms_detections(detections, nms_threshold);
    return 0;
}

std::vector<Detection> rkYolov11::detect(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    std::vector<Detection> detections;
    PreprocessMeta meta;
    if (preprocess(image, meta) != 0) return detections;
    if (runBound() != 0) return detections;
    postprocess(meta, conf_threshold, nms_threshold, detections);
    return detections;
}

std::vector<Detection> rkYolov11::detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                               float nms_threshold) {
    std::vector<Detection> detections;
    PreprocessMeta meta;
    if (preprocess(image, meta) != 0) return detections;
    if (runBound() != 0) return detections;
    postprocess(meta, conf_threshold, nms_threshold, detections);
    return detections;
}

}  // namespace rknn_deploy
