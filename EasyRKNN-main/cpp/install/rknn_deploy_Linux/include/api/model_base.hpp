/**
 * @file model_base.hpp
 * @brief RKNN model base with pre-bound zero-copy IO tensors.
 * @version 3.0.0
 */
#ifndef MODEL_BASE_HPP
#define MODEL_BASE_HPP

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "api/rknn_types.hpp"
#include "core/core_num.hpp"
#include "opencv2/core/core.hpp"
#include "rknn_api.h"
#include "utils/preprocess_yolo.hpp"
#include "utils/rknn_common.hpp"

namespace rknn_deploy {

class RknnModelBase {
   public:
    RknnModelBase(const std::string& model_path, const char* model_name = "Model")
        : model_path_(model_path), model_name_(model_name) {
        std::memset(&io_num_, 0, sizeof(io_num_));
    }

    virtual ~RknnModelBase() {
        releaseIoMemory();
        if (input_attrs_) std::free(input_attrs_);
        if (output_attrs_) std::free(output_attrs_);
        if (ctx_) rknn_destroy(ctx_);
        if (model_data_) std::free(model_data_);
    }

    RknnModelBase(const RknnModelBase&) = delete;
    RknnModelBase& operator=(const RknnModelBase&) = delete;

    /**
     * @param ctx_in parent context used by rknn_dup_context
     * @param share_weight false for first context; true for duplicated contexts
     * @param preferred_core 0..2 on RK3588. Negative => round-robin fallback.
     */
    int init(rknn_context* ctx_in, bool share_weight, int preferred_core = -1) {
        std::printf("[%s] Initializing: %s%s\n", model_name_, model_path_.c_str(),
                    share_weight ? " (dup context)" : "");

        int ret = 0;
        if (share_weight) {
            if (!ctx_in) return -1;
            ret = rknn_dup_context(ctx_in, &ctx_);
        } else {
            int model_data_size = 0;
            model_data_ = load_model_file(model_path_.c_str(), &model_data_size);
            if (!model_data_) {
                std::fprintf(stderr, "[%s] failed to load model\n", model_name_);
                return -1;
            }
            ret = rknn_init(&ctx_, model_data_, model_data_size, 0, nullptr);
            // rknn_init has consumed the model blob; release host-side duplicate immediately.
            std::free(model_data_);
            model_data_ = nullptr;
        }
        if (ret < 0) {
            std::fprintf(stderr, "[%s] context init failed: %d\n", model_name_, ret);
            return ret;
        }

        int core = preferred_core >= 0 ? (preferred_core % 3) : get_core_num();
        rknn_core_mask mask = RKNN_NPU_CORE_0;
        if (core == 1) mask = RKNN_NPU_CORE_1;
        if (core == 2) mask = RKNN_NPU_CORE_2;
        ret = rknn_set_core_mask(ctx_, mask);
        if (ret < 0) {
            std::fprintf(stderr, "[%s] rknn_set_core_mask failed: %d\n", model_name_, ret);
            return ret;
        }
        core_id_ = core;

        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret < 0 || io_num_.n_input < 1 || io_num_.n_output < 1) return -1;

        input_attrs_ = static_cast<rknn_tensor_attr*>(std::calloc(io_num_.n_input, sizeof(rknn_tensor_attr)));
        output_attrs_ = static_cast<rknn_tensor_attr*>(std::calloc(io_num_.n_output, sizeof(rknn_tensor_attr)));
        if (!input_attrs_ || !output_attrs_) return -1;

        for (uint32_t i = 0; i < io_num_.n_input; ++i) {
            input_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
            if (ret < 0) return ret;
        }
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            output_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
            if (ret < 0) return ret;
        }

        const auto& in = input_attrs_[0];
        if (in.n_dims != 4) {
            std::fprintf(stderr, "[%s] only 4D image inputs are supported\n", model_name_);
            return -1;
        }
        if (in.fmt == RKNN_TENSOR_NCHW) {
            channel_ = static_cast<int>(in.dims[1]);
            height_ = static_cast<int>(in.dims[2]);
            width_ = static_cast<int>(in.dims[3]);
        } else {
            height_ = static_cast<int>(in.dims[1]);
            width_ = static_cast<int>(in.dims[2]);
            channel_ = static_cast<int>(in.dims[3]);
        }
        if (channel_ != 3) {
            std::fprintf(stderr, "[%s] only 3-channel image models are supported, got %d\n", model_name_, channel_);
            return -1;
        }

        ret = initModel();
        if (ret != 0) return ret;

        if (zero_copy_io_) {
            ret = initIoMemory();
            if (ret != 0) {
                std::fprintf(stderr, "[%s] zero-copy IO setup failed: %d\n", model_name_, ret);
                return ret;
            }
        }

        std::printf("[%s] ctx core=%d, input=%dx%dx%d, outputs=%u, zero-copy=%s, native-input=%s\n",
                    model_name_, core_id_, width_, height_, channel_, io_num_.n_output,
                    zero_copy_io_ ? "ON" : "OFF", native_input_passthrough_ ? "YES" : "NO");
        return 0;
    }

    rknn_context* get_pctx() { return &ctx_; }

    virtual std::vector<Detection> detect(const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
    virtual std::vector<Detection> detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                                float nms_threshold) = 0;

    void setNmsThreshold(float threshold) { nms_threshold_ = threshold; }
    float getNmsThreshold() const { return nms_threshold_; }
    void setConfThreshold(float threshold) { conf_threshold_ = threshold; }
    float getConfThreshold() const { return conf_threshold_; }
    void setResizeMode(ResizeMode mode) { resize_mode_ = mode; }
    ResizeMode getResizeMode() const { return resize_mode_; }
    void setZeroCopyEnabled(bool enabled) {
        if (!enabled) std::fprintf(stderr, "[%s] legacy copied IO is removed in v3; zero-copy remains enabled\n", model_name_);
        zero_copy_io_ = true;
    }
    bool isZeroCopyEnabled() const { return zero_copy_io_; }

    int getInputWidth() const { return width_; }
    int getInputHeight() const { return height_; }
    int getInputChannels() const { return channel_; }
    virtual int getNumClasses() const { return 0; }

    void setLabels(const std::vector<std::string>& labels) { labels_ = labels; }
    const std::vector<std::string>& getLabels() const { return labels_; }

   protected:
    virtual int initModel() { return 0; }

    int preprocess(const cv::Mat& image, PreprocessMeta& meta) {
        if (!input_mem_) return -1;
        return preprocess_cvmat_to_rknn(image, input_mem_, width_, height_, input_w_stride_, input_h_stride_,
                                        resize_mode_, meta);
    }

    int preprocess(const DmaBufImage& image, PreprocessMeta& meta) {
        if (!input_mem_) return -1;
        return preprocess_dmabuf_to_rknn(image, input_mem_, width_, height_, input_w_stride_, input_h_stride_,
                                         resize_mode_, meta);
    }

    int runBound() { return rknn_run(ctx_, nullptr); }

    void* outputPtr(size_t index) const {
        return index < output_mems_.size() && output_mems_[index] ? output_mems_[index]->virt_addr : nullptr;
    }

    size_t outputCount() const { return output_mems_.size(); }

    int tensorWidthStride(const rknn_tensor_attr& attr, int logical_width) const {
        return attr.w_stride > 0 ? static_cast<int>(attr.w_stride) : logical_width;
    }
    int tensorHeightStride(const rknn_tensor_attr& attr, int logical_height) const {
        return attr.h_stride > 0 ? static_cast<int>(attr.h_stride) : logical_height;
    }

   private:
    int initIoMemory() {
        // Prefer the runtime native NHWC input layout when it can consume raw RGB888/U8 directly.
        // This avoids the driver's extra NHWC->internal-layout conversion.
        rknn_tensor_attr native_nhwc{};
        native_nhwc.index = 0;
        int native_ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &native_nhwc, sizeof(native_nhwc));
        if (native_ret == 0 && native_nhwc.fmt == RKNN_TENSOR_NHWC && native_nhwc.type == RKNN_TENSOR_UINT8) {
            input_bind_attr_ = native_nhwc;
            input_bind_attr_.pass_through = 1;
            native_input_passthrough_ = true;
        } else {
            // Safe fallback: still zero-copy at the userspace/RKNN IO boundary; runtime may do layout conversion.
            input_bind_attr_ = input_attrs_[0];
            input_bind_attr_.type = RKNN_TENSOR_UINT8;
            input_bind_attr_.fmt = RKNN_TENSOR_NHWC;
            input_bind_attr_.pass_through = 0;
            native_input_passthrough_ = false;
        }

        input_w_stride_ = input_bind_attr_.w_stride > 0 ? static_cast<int>(input_bind_attr_.w_stride) : width_;
        input_h_stride_ = input_bind_attr_.h_stride > 0 ? static_cast<int>(input_bind_attr_.h_stride) : height_;
        // For RGB888 destination RGA stride must be in pixels. Ensure allocation covers it.
        uint32_t rgb_size = static_cast<uint32_t>(input_w_stride_ * input_h_stride_ * channel_);
        uint32_t input_size = std::max(rgb_size,
                                       input_bind_attr_.size_with_stride > 0 ? input_bind_attr_.size_with_stride
                                                                            : input_bind_attr_.size);
        input_mem_ = rknn_create_mem(ctx_, input_size);
        if (!input_mem_) return -1;
        int ret = rknn_set_io_mem(ctx_, input_mem_, &input_bind_attr_);
        if (ret < 0) return ret;

        output_mems_.resize(io_num_.n_output, nullptr);
        output_bind_attrs_.resize(io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            output_bind_attrs_[i] = output_attrs_[i];
            uint32_t bytes = output_bind_attrs_[i].size_with_stride > 0
                                 ? output_bind_attrs_[i].size_with_stride
                                 : output_bind_attrs_[i].size;
            output_mems_[i] = rknn_create_mem(ctx_, bytes);
            if (!output_mems_[i]) return -1;
            ret = rknn_set_io_mem(ctx_, output_mems_[i], &output_bind_attrs_[i]);
            if (ret < 0) return ret;
        }
        return 0;
    }

    void releaseIoMemory() {
        if (!ctx_) return;
        for (auto*& mem : output_mems_) {
            if (mem) {
                rknn_destroy_mem(ctx_, mem);
                mem = nullptr;
            }
        }
        output_mems_.clear();
        if (input_mem_) {
            rknn_destroy_mem(ctx_, input_mem_);
            input_mem_ = nullptr;
        }
    }

   protected:
    std::string model_path_;
    const char* model_name_;
    unsigned char* model_data_ = nullptr;
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr* input_attrs_ = nullptr;
    rknn_tensor_attr* output_attrs_ = nullptr;
    rknn_tensor_attr input_bind_attr_{};
    std::vector<rknn_tensor_attr> output_bind_attrs_;
    rknn_tensor_mem* input_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> output_mems_;

    int channel_ = 3;
    int width_ = 0;
    int height_ = 0;
    int input_w_stride_ = 0;
    int input_h_stride_ = 0;
    int core_id_ = 0;

    float nms_threshold_ = 0.45f;
    float conf_threshold_ = 0.25f;
    ResizeMode resize_mode_ = ResizeMode::Stretch;
    bool zero_copy_io_ = true;
    bool native_input_passthrough_ = false;
    std::vector<std::string> labels_;
};

}  // namespace rknn_deploy
#endif
