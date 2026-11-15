#ifndef RKNN_DEPLOY_PREPROCESS_YOLO_HPP
#define RKNN_DEPLOY_PREPROCESS_YOLO_HPP

#include "api/rknn_types.hpp"
#include "im2d_version.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "rga.h"
#include "rknn_api.h"

namespace rknn_deploy {

int resize_rga(rga_buffer_t& src, rga_buffer_t& dst, const cv::Mat& image, cv::Mat& resized_image,
               const cv::Size& target_size);

/**
 * Copy-minimized OpenCV path:
 * BGR cv::Mat -> RGA -> pre-bound RKNN input DMA buffer (RGB888).
 * No cvtColor/resized cv::Mat is allocated on the fast path.
 */
int preprocess_cvmat_to_rknn(const cv::Mat& image, rknn_tensor_mem* input_mem,
                             int dst_width, int dst_height, int dst_w_stride, int dst_h_stride,
                             ResizeMode mode, PreprocessMeta& meta);

/**
 * True DMA-BUF zero-copy source path:
 * external DMA-BUF -> RGA -> pre-bound RKNN input DMA buffer.
 */
int preprocess_dmabuf_to_rknn(const DmaBufImage& image, rknn_tensor_mem* input_mem,
                              int dst_width, int dst_height, int dst_w_stride, int dst_h_stride,
                              ResizeMode mode, PreprocessMeta& meta);

}  // namespace rknn_deploy
#endif
