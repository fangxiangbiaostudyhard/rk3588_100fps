#include "api/rknn_detector.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unistd.h>

#include "api/model_wrapper.hpp"
#include "core/thread_pool.hpp"
#include "models/yolov11/rknn_yolov11.hpp"
#include "models/yolov5/rknn_yolov5.hpp"
#include "models/yolov8/rknn_yolov8.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace rknn_deploy {
static const char* SDK_VERSION = "2.0.0-zero-copy";

using Yolov5Wrapper = ModelWrapper<rkYolov5s>;
using Yolov8Wrapper = ModelWrapper<rkYolov8>;
using Yolov11Wrapper = ModelWrapper<rkYolov11>;

class RknnDetectorImpl final : public RknnDetector {
    using WrapperCreator = std::function<std::shared_ptr<IModelWrapper>(const std::string&, int)>;

    static std::unordered_map<ModelType, WrapperCreator>& Registry() {
        static std::unordered_map<ModelType, WrapperCreator> reg;
        return reg;
    }

   public:
    explicit RknnDetectorImpl(int worker_count)
        : worker_count_(std::max(1, worker_count)),
          thread_pool_(std::make_unique<dpool::ThreadPool>(worker_count_, worker_count_ * 2)) {
        auto& reg = Registry();
        if (!reg.count(ModelType::YOLOV5))
            reg[ModelType::YOLOV5] = [](const std::string& p, int n) { return std::make_shared<Yolov5Wrapper>(p, n); };
        if (!reg.count(ModelType::YOLOV8))
            reg[ModelType::YOLOV8] = [](const std::string& p, int n) { return std::make_shared<Yolov8Wrapper>(p, n); };
        if (!reg.count(ModelType::YOLOV11))
            reg[ModelType::YOLOV11] = [](const std::string& p, int n) { return std::make_shared<Yolov11Wrapper>(p, n); };
    }

    ~RknnDetectorImpl() override {
        if (thread_pool_) thread_pool_->stop();
        UnloadAllModels();
    }

    ErrorCode LoadModel(const std::string& name, const std::string& path, ModelType type, int pool_size) override {
        ModelConfig cfg;
        cfg.model_path = path;
        cfg.type = type;
        cfg.pool_size = pool_size > 0 ? pool_size : worker_count_;
        return LoadModel(name, cfg);
    }

    ErrorCode LoadModel(const std::string& name, const ModelConfig& config) override {
        if (name.empty() || config.model_path.empty()) return ErrorCode::InvalidArgument;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (models_.count(name)) return ErrorCode::ModelAlreadyExists;
        }
        std::ifstream model_file(config.model_path, std::ios::binary);
        if (!model_file.good()) return ErrorCode::FileNotExist;

        auto it = Registry().find(config.type);
        if (it == Registry().end()) return ErrorCode::UnsupportedModelType;

        auto entry = std::make_shared<ModelEntry>();
        entry->config = config;
        entry->config.pool_size = std::max(1, config.pool_size > 0 ? config.pool_size : worker_count_);
        entry->type = config.type;
        entry->path = config.model_path;
        entry->wrapper = it->second(config.model_path, entry->config.pool_size);
        entry->wrapper->setResizeMode(config.resize_mode);
        entry->wrapper->setZeroCopyEnabled(config.zero_copy_io);

        std::printf("[RknnDetector] load '%s': %s, type=%s, contexts=%d, zero-copy=%s\n",
                    name.c_str(), config.model_path.c_str(), GetModelTypeString(config.type),
                    entry->config.pool_size, config.zero_copy_io ? "ON" : "OFF");
        if (entry->wrapper->init() != 0) return ErrorCode::InitFailed;

        if (!config.labels_path.empty()) {
            if (!readLabels(config.labels_path, entry->labels)) return ErrorCode::FileNotExist;
            const int classes = entry->wrapper->getNumClasses();
            if (classes > 0 && static_cast<int>(entry->labels.size()) != classes) {
                std::fprintf(stderr, "[RknnDetector] labels count %zu != model classes %d\n", entry->labels.size(), classes);
                return ErrorCode::InvalidArgument;
            }
            entry->wrapper->setLabels(entry->labels);
        }
        entry->config.zero_copy_io = true;

        std::lock_guard<std::mutex> lock(mutex_);
        if (models_.count(name)) return ErrorCode::ModelAlreadyExists;
        models_[name] = entry;
        model_names_.push_back(name);
        return ErrorCode::Success;
    }

    ErrorCode UnloadModel(const std::string& name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(name);
        if (it == models_.end()) return ErrorCode::ModelNotFound;
        models_.erase(it);
        model_names_.erase(std::remove(model_names_.begin(), model_names_.end(), name), model_names_.end());
        return ErrorCode::Success;
    }

    void UnloadAllModels() override {
        std::lock_guard<std::mutex> lock(mutex_);
        models_.clear();
        model_names_.clear();
    }

    bool HasModel(const std::string& name) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return models_.count(name) != 0;
    }
    std::vector<std::string> GetModelNames() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return model_names_;
    }
    int GetModelCount() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(models_.size());
    }

    ModelInfo GetModelInfo(const std::string& name) const override {
        ModelInfo info;
        auto entry = findEntry(name);
        if (!entry || !entry->wrapper) return info;
        info.name = name;
        info.path = entry->path;
        info.type = entry->type;
        info.pool_info = getPoolInfo(entry);
        info.input_width = entry->wrapper->getInputWidth();
        info.input_height = entry->wrapper->getInputHeight();
        info.input_channels = entry->wrapper->getInputChannels();
        info.num_classes = entry->wrapper->getNumClasses();
        info.zero_copy_io = entry->wrapper->isZeroCopyEnabled();
        return info;
    }

    DetectionResult Detect(const std::string& name, const cv::Mat& image, float threshold) override {
        DetectionResult result;
        result.image_width = image.cols;
        result.image_height = image.rows;
        if (image.empty()) {
            result.error_code = ErrorCode::InvalidArgument;
            result.error_message = "empty image";
            return result;
        }
        auto entry = findEntry(name);
        if (!entry) return modelNotFound(name, result);
        float conf, nms;
        {
            std::lock_guard<std::mutex> cfg_lock(entry->config_mutex);
            conf = threshold >= 0 ? threshold : entry->config.conf_threshold;
            nms = entry->config.nms_threshold;
        }
        auto t0 = std::chrono::steady_clock::now();
        try {
            result.detections = entry->wrapper->detect(image, conf, nms);
        } catch (const std::exception& e) {
            result.error_code = ErrorCode::InferenceFailed;
            result.error_message = e.what();
        }
        result.inference_time_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return result;
    }

    DetectionResult DetectDmaBuf(const std::string& name, const DmaBufImage& image, float threshold) override {
        DetectionResult result;
        result.image_width = image.width;
        result.image_height = image.height;
        if (!image.Valid()) {
            result.error_code = ErrorCode::InvalidArgument;
            result.error_message = "invalid DMA-BUF image";
            return result;
        }
        auto entry = findEntry(name);
        if (!entry) return modelNotFound(name, result);
        float conf, nms;
        {
            std::lock_guard<std::mutex> cfg_lock(entry->config_mutex);
            conf = threshold >= 0 ? threshold : entry->config.conf_threshold;
            nms = entry->config.nms_threshold;
        }
        auto t0 = std::chrono::steady_clock::now();
        try {
            result.detections = entry->wrapper->detectDmaBuf(image, conf, nms);
        } catch (const std::exception& e) {
            result.error_code = ErrorCode::InferenceFailed;
            result.error_message = e.what();
        }
        result.inference_time_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return result;
    }

    ErrorCode DetectAsync(const std::string& name, const cv::Mat& image, DetectCallback callback,
                          float threshold) override {
        if (image.empty()) return ErrorCode::InvalidArgument;
        cv::Mat shared = image; // refcounted shallow copy: no full-frame memcpy
        try {
            const uint64_t request_id = request_seq_.fetch_add(1) + 1;
            thread_pool_->submit([this, name, shared, callback = std::move(callback), threshold, request_id]() mutable {
                auto result = Detect(name, shared, threshold);
                result.request_id = request_id;
                if (callback) callback(result);
            });
        } catch (...) {
            return ErrorCode::QueueFull;
        }
        return ErrorCode::Success;
    }

    ErrorCode DetectDmaBufAsync(const std::string& name, const DmaBufImage& image, DetectCallback callback,
                                float threshold) override {
        if (!image.Valid()) return ErrorCode::InvalidArgument;
        int owned_fd = ::dup(image.fd);
        if (owned_fd < 0) return ErrorCode::InvalidArgument;
        DmaBufImage owned = image;
        owned.fd = owned_fd;
        try {
            const uint64_t request_id = request_seq_.fetch_add(1) + 1;
            thread_pool_->submit([this, name, owned, callback = std::move(callback), threshold, request_id]() mutable {
                auto result = DetectDmaBuf(name, owned, threshold);
                result.request_id = request_id;
                ::close(owned.fd);
                if (callback) callback(result);
            });
        } catch (...) {
            ::close(owned_fd);
            return ErrorCode::QueueFull;
        }
        return ErrorCode::Success;
    }

    ErrorCode SetNmsThreshold(const std::string& name, float threshold) override {
        if (threshold < 0 || threshold > 1) return ErrorCode::InvalidArgument;
        auto entry = findEntry(name);
        if (!entry) return ErrorCode::ModelNotFound;
        { std::lock_guard<std::mutex> cfg_lock(entry->config_mutex); entry->config.nms_threshold = threshold; }
        return ErrorCode::Success;
    }
    ErrorCode SetConfThreshold(const std::string& name, float threshold) override {
        if (threshold < 0 || threshold > 1) return ErrorCode::InvalidArgument;
        auto entry = findEntry(name);
        if (!entry) return ErrorCode::ModelNotFound;
        { std::lock_guard<std::mutex> cfg_lock(entry->config_mutex); entry->config.conf_threshold = threshold; }
        return ErrorCode::Success;
    }
    ErrorCode LoadLabels(const std::string& name, const std::string& path) override {
        auto entry = findEntry(name);
        if (!entry) return ErrorCode::ModelNotFound;
        std::vector<std::string> labels;
        if (!readLabels(path, labels)) return ErrorCode::FileNotExist;
        const int classes = entry->wrapper->getNumClasses();
        if (classes > 0 && static_cast<int>(labels.size()) != classes) return ErrorCode::InvalidArgument;
        { std::lock_guard<std::mutex> cfg_lock(entry->config_mutex); entry->labels = labels; }
        entry->wrapper->setLabels(labels);
        return ErrorCode::Success;
    }

    int GetNpuCoreCount() const override { return 3; }
    int GetWorkerCount() const override { return worker_count_; }
    size_t GetPendingTaskCount() const override { return thread_pool_ ? thread_pool_->queue_size() : 0; }
    PoolInfo GetPoolInfo(const std::string& name) const override {
        auto entry = findEntry(name);
        return entry ? getPoolInfo(entry) : PoolInfo{};
    }

   private:
    struct ModelEntry {
        ModelConfig config;
        ModelType type = ModelType::CUSTOM;
        std::string path;
        std::vector<std::string> labels;
        std::shared_ptr<IModelWrapper> wrapper;
        mutable std::mutex config_mutex;
    };

    static bool readLabels(const std::string& path, std::vector<std::string>& labels) {
        std::ifstream file(path);
        if (!file.good()) return false;
        labels.clear();
        std::string line;
        while (std::getline(file, line)) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            const auto last = line.find_last_not_of(" \t\r\n");
            labels.push_back(line.substr(first, last - first + 1));
        }
        return true;
    }

    std::shared_ptr<ModelEntry> findEntry(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(name);
        return it == models_.end() ? nullptr : it->second;
    }
    static PoolInfo getPoolInfo(const std::shared_ptr<ModelEntry>& entry) {
        PoolInfo p;
        if (!entry || !entry->wrapper) return p;
        p.max_size = entry->wrapper->getPoolSize();
        p.idle_count = entry->wrapper->getIdleCount();
        p.active_count = entry->wrapper->getActiveCount();
        return p;
    }
    static DetectionResult modelNotFound(const std::string& name, DetectionResult result) {
        result.error_code = ErrorCode::ModelNotFound;
        result.error_message = "Model not found: " + name;
        return result;
    }

    int worker_count_;
    std::unique_ptr<dpool::ThreadPool> thread_pool_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ModelEntry>> models_;
    std::vector<std::string> model_names_;
    std::atomic<uint64_t> request_seq_{0};
};

std::shared_ptr<RknnDetector> RknnDetector::Create(int worker_count) {
    return std::make_shared<RknnDetectorImpl>(worker_count);
}
const char* RknnDetector::GetVersion() { return SDK_VERSION; }

cv::Scalar GetClassColor(int class_id) {
    float hue = static_cast<float>((class_id * 37) % 180);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto v = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(v[0], v[1], v[2]);
}

void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections, bool draw_label, int thickness) {
    for (const auto& det : detections) {
        cv::Scalar color = GetClassColor(det.label_id);
        if (det.bbox.Area() <= 0) continue;
        cv::rectangle(image, cv::Point(static_cast<int>(det.bbox.left), static_cast<int>(det.bbox.top)),
                      cv::Point(static_cast<int>(det.bbox.right), static_cast<int>(det.bbox.bottom)), color, thickness);
        if (!draw_label) continue;
        std::string name = det.label_name.empty() ? ("Class" + std::to_string(det.label_id)) : det.label_name;
        char text[192];
        std::snprintf(text, sizeof(text), "%s %.0f%%", name.c_str(), det.score * 100.0f);
        int baseline = 0;
        auto sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = std::max(0, static_cast<int>(det.bbox.left));
        int y = std::max(sz.height + 4, static_cast<int>(det.bbox.top));
        cv::rectangle(image, cv::Point(x, y - sz.height - 4), cv::Point(x + sz.width, y + 2), color, cv::FILLED);
        cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }
}

}  // namespace rknn_deploy
