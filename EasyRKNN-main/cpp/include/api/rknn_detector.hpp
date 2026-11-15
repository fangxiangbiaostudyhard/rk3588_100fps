#ifndef RKNN_DETECTOR_HPP
#define RKNN_DETECTOR_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "api/rknn_types.hpp"
#include "opencv2/core/core.hpp"

namespace rknn_deploy {

class RknnDetector {
   public:
    using DetectCallback = std::function<void(const DetectionResult&)>;

    // worker_count is the CPU/request worker count. RK3588 still has 3 physical NPU cores.
    static std::shared_ptr<RknnDetector> Create(int worker_count = 3);
    virtual ~RknnDetector() = default;

    virtual ErrorCode LoadModel(const std::string& model_name, const std::string& model_path, ModelType type,
                                int pool_size = 0) = 0;
    virtual ErrorCode LoadModel(const std::string& model_name, const ModelConfig& config) = 0;
    virtual ErrorCode UnloadModel(const std::string& model_name) = 0;
    virtual void UnloadAllModels() = 0;
    virtual bool HasModel(const std::string& model_name) const = 0;
    virtual std::vector<std::string> GetModelNames() const = 0;
    virtual int GetModelCount() const = 0;
    virtual ModelInfo GetModelInfo(const std::string& model_name) const = 0;

    // threshold < 0 => use the configured per-model confidence threshold.
    virtual DetectionResult Detect(const std::string& model_name, const cv::Mat& image,
                                   float score_threshold = -1.0f) = 0;
    virtual DetectionResult DetectDmaBuf(const std::string& model_name, const DmaBufImage& image,
                                         float score_threshold = -1.0f) = 0;

    /**
     * OpenCV async path is copy-minimized: cv::Mat header/refcount is retained, no image.clone().
     * Do not mutate externally-backed image memory until callback completes.
     */
    virtual ErrorCode DetectAsync(const std::string& model_name, const cv::Mat& image, DetectCallback callback,
                                  float score_threshold = -1.0f) = 0;
    /** Async DMA-BUF path duplicates fd so the caller may close its original fd immediately. */
    virtual ErrorCode DetectDmaBufAsync(const std::string& model_name, const DmaBufImage& image,
                                        DetectCallback callback, float score_threshold = -1.0f) = 0;

    virtual ErrorCode SetNmsThreshold(const std::string& model_name, float threshold) = 0;
    virtual ErrorCode SetConfThreshold(const std::string& model_name, float threshold) = 0;
    virtual ErrorCode LoadLabels(const std::string& model_name, const std::string& labels_path) = 0;

    static const char* GetVersion();
    virtual int GetNpuCoreCount() const = 0;
    virtual int GetWorkerCount() const = 0;
    virtual size_t GetPendingTaskCount() const = 0;
    virtual PoolInfo GetPoolInfo(const std::string& model_name) const = 0;

   protected:
    RknnDetector() = default;
};

void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections, bool draw_label = true,
                    int thickness = 2);
cv::Scalar GetClassColor(int class_id);

}  // namespace rknn_deploy
#endif
