/**
 * @file model_wrapper.hpp
 * @brief Thread-safe RKNN context pool using borrow/return semantics.
 * @version 3.0.0
 */
#ifndef MODEL_WRAPPER_HPP
#define MODEL_WRAPPER_HPP

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#include "api/rknn_types.hpp"

namespace rknn_deploy {

class IModelWrapper {
   public:
    virtual ~IModelWrapper() = default;
    virtual int init() = 0;
    virtual std::vector<Detection> detect(const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
    virtual std::vector<Detection> detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                                float nms_threshold) = 0;
    virtual void setLabels(const std::vector<std::string>& labels) = 0;
    virtual const std::vector<std::string>& getLabels() const = 0;
    virtual void setResizeMode(ResizeMode mode) = 0;
    virtual void setZeroCopyEnabled(bool enabled) = 0;
    virtual int getPoolSize() const = 0;
    virtual int getIdleCount() const = 0;
    virtual int getActiveCount() const = 0;
    virtual int getInputWidth() const = 0;
    virtual int getInputHeight() const = 0;
    virtual int getInputChannels() const = 0;
    virtual int getNumClasses() const = 0;
    virtual bool isZeroCopyEnabled() const = 0;
};

template <typename ModelT>
class ModelWrapper : public IModelWrapper {
   public:
    ModelWrapper(const std::string& model_path, int pool_size)
        : pool_size_(std::max(1, pool_size)), model_path_(model_path) {}

    ~ModelWrapper() override {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        shutting_down_ = true;
        pool_cv_.notify_all();
        while (!idle_queue_.empty()) idle_queue_.pop();
        models_.clear();
    }

    int init() override {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        if (is_initialized_) return 0;
        try {
            models_.reserve(pool_size_);
            for (int i = 0; i < pool_size_; ++i) {
                auto model = std::make_shared<ModelT>(model_path_);
                model->setResizeMode(resize_mode_);
                model->setZeroCopyEnabled(zero_copy_enabled_);
                model->setLabels(labels_);
                int ret = model->init(models_.empty() ? nullptr : models_[0]->get_pctx(), !models_.empty(), i % 3);
                if (ret != 0) {
                    std::fprintf(stderr, "[ModelWrapper] init instance %d failed: %d\n", i, ret);
                    return ret;
                }
                models_.push_back(model);
                model_label_version_[model.get()] = labels_version_;
                idle_queue_.push(model.get());
            }
            is_initialized_ = true;
            std::printf("[ModelWrapper] initialized %d contexts, mapped round-robin to 3 RK3588 NPU cores\n",
                        pool_size_);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ModelWrapper] init exception: %s\n", e.what());
            return -1;
        }
    }

    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold, float nms_threshold) override {
        ModelT* model = acquire();
        if (!model) return {};
        Lease lease(this, model);
        syncLabelsIfNeeded(model);
        model->setConfThreshold(conf_threshold);
        model->setNmsThreshold(nms_threshold);
        return model->detect(image, conf_threshold, nms_threshold);
    }

    std::vector<Detection> detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                        float nms_threshold) override {
        ModelT* model = acquire();
        if (!model) return {};
        Lease lease(this, model);
        syncLabelsIfNeeded(model);
        model->setConfThreshold(conf_threshold);
        model->setNmsThreshold(nms_threshold);
        return model->detectDmaBuf(image, conf_threshold, nms_threshold);
    }

    void setLabels(const std::vector<std::string>& labels) override {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        labels_ = labels;
        ++labels_version_;
        // Active contexts are not modified here. Each context applies the new labels the next time it is borrowed.
    }
    const std::vector<std::string>& getLabels() const override { return labels_; }

    void setResizeMode(ResizeMode mode) override {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        resize_mode_ = mode;
        for (auto& model : models_) model->setResizeMode(mode);
    }

    void setZeroCopyEnabled(bool enabled) override {
        // Must be set before init because IO memory is bound at context initialization.
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if (is_initialized_) return;
        zero_copy_enabled_ = true;
        (void)enabled;
    }

    int getPoolSize() const override { return pool_size_; }
    int getIdleCount() const override {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return static_cast<int>(idle_queue_.size());
    }
    int getActiveCount() const override { return active_count_.load(); }
    int getInputWidth() const override { return models_.empty() ? 0 : models_[0]->getInputWidth(); }
    int getInputHeight() const override { return models_.empty() ? 0 : models_[0]->getInputHeight(); }
    int getInputChannels() const override { return models_.empty() ? 0 : models_[0]->getInputChannels(); }
    int getNumClasses() const override { return models_.empty() ? 0 : models_[0]->getNumClasses(); }
    bool isZeroCopyEnabled() const override {
        return !models_.empty() ? models_[0]->isZeroCopyEnabled() : zero_copy_enabled_;
    }

   private:
    struct Lease {
        ModelWrapper* owner;
        ModelT* model;
        Lease(ModelWrapper* o, ModelT* m) : owner(o), model(m) { owner->active_count_.fetch_add(1); }
        ~Lease() {
            owner->active_count_.fetch_sub(1);
            owner->release(model);
        }
    };

    void syncLabelsIfNeeded(ModelT* model) {
        std::vector<std::string> labels_copy;
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            auto it = model_label_version_.find(model);
            if (it != model_label_version_.end() && it->second == labels_version_) return;
            labels_copy = labels_;
            model_label_version_[model] = labels_version_;
        }
        model->setLabels(labels_copy);
    }

    ModelT* acquire() {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        pool_cv_.wait(lock, [this] { return shutting_down_ || !idle_queue_.empty(); });
        if (shutting_down_ || idle_queue_.empty()) return nullptr;
        ModelT* model = idle_queue_.front();
        idle_queue_.pop();
        return model;
    }

    void release(ModelT* model) {
        if (!model) return;
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            if (!shutting_down_) idle_queue_.push(model);
        }
        pool_cv_.notify_one();
    }

    int pool_size_ = 1;
    std::string model_path_;
    std::vector<std::string> labels_;
    ResizeMode resize_mode_ = ResizeMode::Stretch;
    bool zero_copy_enabled_ = true;
    bool is_initialized_ = false;
    bool shutting_down_ = false;
    std::atomic<int> active_count_{0};

    std::vector<std::shared_ptr<ModelT>> models_;
    std::queue<ModelT*> idle_queue_;
    std::unordered_map<ModelT*, uint64_t> model_label_version_;
    uint64_t labels_version_ = 0;
    mutable std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
};

class rkYolov5s;
class rkYolov8;
class rkYolov11;
using Yolov5Wrapper = ModelWrapper<rkYolov5s>;
using Yolov8Wrapper = ModelWrapper<rkYolov8>;
using Yolov11Wrapper = ModelWrapper<rkYolov11>;

}  // namespace rknn_deploy
#endif
