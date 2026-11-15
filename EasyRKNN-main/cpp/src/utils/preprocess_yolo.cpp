#include "utils/preprocess_yolo.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "opencv2/imgproc.hpp"

namespace rknn_deploy {
namespace {

int to_rga_format(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::BGR888: return RK_FORMAT_BGR_888;
        case PixelFormat::RGB888: return RK_FORMAT_RGB_888;
        case PixelFormat::NV12: return RK_FORMAT_YCbCr_420_SP;
        case PixelFormat::NV21: return RK_FORMAT_YCrCb_420_SP;
        default: return RK_FORMAT_UNKNOWN;
    }
}

PreprocessMeta make_meta(int sw, int sh, int dw, int dh, ResizeMode mode) {
    PreprocessMeta meta;
    meta.src_width = sw;
    meta.src_height = sh;
    meta.dst_width = dw;
    meta.dst_height = dh;
    meta.mode = mode;
    if (mode == ResizeMode::Letterbox) {
        float scale = std::min(static_cast<float>(dw) / sw, static_cast<float>(dh) / sh);
        int rw = std::max(1, static_cast<int>(std::round(sw * scale)));
        int rh = std::max(1, static_cast<int>(std::round(sh * scale)));
        meta.scale_x = static_cast<float>(rw) / sw;
        meta.scale_y = static_cast<float>(rh) / sh;
        meta.pad_left = (dw - rw) / 2;
        meta.pad_top = (dh - rh) / 2;
    } else {
        meta.scale_x = static_cast<float>(dw) / sw;
        meta.scale_y = static_cast<float>(dh) / sh;
    }
    return meta;
}

int rga_process(rga_buffer_t src, rga_buffer_t dst, const PreprocessMeta& meta) {
    im_rect srect{0, 0, meta.src_width, meta.src_height};
    im_rect drect{};
    if (meta.mode == ResizeMode::Letterbox) {
        drect.x = meta.pad_left;
        drect.y = meta.pad_top;
        drect.width = static_cast<int>(std::round(meta.src_width * meta.scale_x));
        drect.height = static_cast<int>(std::round(meta.src_height * meta.scale_y));

        im_rect full{0, 0, meta.dst_width, meta.dst_height};
        // 0x727272 = RGB(114,114,114), Ultralytics default pad value.
        IM_STATUS fill_ret = imfill(dst, full, 0x727272, 1);
        if (fill_ret != IM_STATUS_SUCCESS) {
            fprintf(stderr, "[RGA] imfill failed: %s\n", imStrError(fill_ret));
            return -1;
        }
    } else {
        drect = {0, 0, meta.dst_width, meta.dst_height};
    }

    rga_buffer_t pat{};
    im_rect prect{};
    im_opt_t opt{};
    IM_STATUS ret = improcess(src, dst, pat, srect, drect, prect, -1, nullptr, &opt, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[RGA] improcess failed: %s\n", imStrError(ret));
        return -1;
    }
    return 0;
}

int fallback_cv(const cv::Mat& image, rknn_tensor_mem* input_mem, int dw, int dh, int wstride,
                ResizeMode mode, PreprocessMeta& meta) {
    if (!input_mem || !input_mem->virt_addr) return -1;
    const size_t row_bytes = static_cast<size_t>(wstride) * 3;
    cv::Mat dst(dh, wstride, CV_8UC3, input_mem->virt_addr, row_bytes);
    cv::Mat active = dst(cv::Rect(0, 0, dw, dh));

    if (mode == ResizeMode::Stretch) {
        cv::Mat resized_bgr;
        cv::resize(image, resized_bgr, cv::Size(dw, dh));
        cv::cvtColor(resized_bgr, active, cv::COLOR_BGR2RGB);
        return 0;
    }

    active.setTo(cv::Scalar(114, 114, 114));
    int rw = static_cast<int>(std::round(image.cols * meta.scale_x));
    int rh = static_cast<int>(std::round(image.rows * meta.scale_y));
    cv::Mat resized_bgr;
    cv::resize(image, resized_bgr, cv::Size(rw, rh));
    cv::Mat roi = active(cv::Rect(meta.pad_left, meta.pad_top, rw, rh));
    cv::cvtColor(resized_bgr, roi, cv::COLOR_BGR2RGB);
    return 0;
}

}  // namespace

int resize_rga(rga_buffer_t& src, rga_buffer_t& dst, const cv::Mat& image, cv::Mat& resized_image,
               const cv::Size& target_size) {
    if (image.type() != CV_8UC3 || resized_image.type() != CV_8UC3) return -1;
    int src_wstride = static_cast<int>(image.step / image.elemSize());
    int dst_wstride = static_cast<int>(resized_image.step / resized_image.elemSize());
    src = wrapbuffer_virtualaddr(image.data, image.cols, image.rows, RK_FORMAT_RGB_888, src_wstride, image.rows);
    dst = wrapbuffer_virtualaddr(resized_image.data, target_size.width, target_size.height, RK_FORMAT_RGB_888, dst_wstride, target_size.height);
    IM_STATUS ret = imresize(src, dst);
    return ret == IM_STATUS_SUCCESS ? 0 : -1;
}

int preprocess_cvmat_to_rknn(const cv::Mat& image, rknn_tensor_mem* input_mem,
                             int dst_width, int dst_height, int dst_w_stride, int dst_h_stride,
                             ResizeMode mode, PreprocessMeta& meta) {
    if (image.empty() || image.type() != CV_8UC3 || !input_mem) return -1;
    meta = make_meta(image.cols, image.rows, dst_width, dst_height, mode);

    int src_wstride = static_cast<int>(image.step / image.elemSize());
    rga_buffer_t src = wrapbuffer_virtualaddr(image.data, image.cols, image.rows, RK_FORMAT_BGR_888, src_wstride, image.rows);
    rga_buffer_t dst{};
    if (input_mem->fd >= 0) {
        dst = wrapbuffer_fd(input_mem->fd, dst_width, dst_height, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else if (input_mem->virt_addr) {
        dst = wrapbuffer_virtualaddr(input_mem->virt_addr, dst_width, dst_height, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        return -1;
    }

    if (rga_process(src, dst, meta) == 0) return 0;
    fprintf(stderr, "[RGA] falling back to OpenCV preprocessing\n");
    return fallback_cv(image, input_mem, dst_width, dst_height, dst_w_stride, mode, meta);
}

int preprocess_dmabuf_to_rknn(const DmaBufImage& image, rknn_tensor_mem* input_mem,
                              int dst_width, int dst_height, int dst_w_stride, int dst_h_stride,
                              ResizeMode mode, PreprocessMeta& meta) {
    if (!image.Valid() || !input_mem) return -1;
    int src_fmt = to_rga_format(image.format);
    if (src_fmt == RK_FORMAT_UNKNOWN) return -1;

    meta = make_meta(image.width, image.height, dst_width, dst_height, mode);
    int sws = image.w_stride > 0 ? image.w_stride : image.width;
    int shs = image.h_stride > 0 ? image.h_stride : image.height;
    rga_buffer_t src = wrapbuffer_fd(image.fd, image.width, image.height, src_fmt, sws, shs);
    rga_buffer_t dst{};
    if (input_mem->fd >= 0) {
        dst = wrapbuffer_fd(input_mem->fd, dst_width, dst_height, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else if (input_mem->virt_addr) {
        dst = wrapbuffer_virtualaddr(input_mem->virt_addr, dst_width, dst_height, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        return -1;
    }
    return rga_process(src, dst, meta);
}

}  // namespace rknn_deploy
