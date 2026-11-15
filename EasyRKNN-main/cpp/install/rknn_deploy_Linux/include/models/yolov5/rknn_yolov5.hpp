#ifndef RKNN_YOLOV5_HPP
#define RKNN_YOLOV5_HPP

#include "api/model_base.hpp"

namespace rknn_deploy {

class rkYolov5s : public RknnModelBase {
   public:
    explicit rkYolov5s(const std::string& model_path) : RknnModelBase(model_path, "YOLOv5") {}

    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold, float nms_threshold) override;
    std::vector<Detection> detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                        float nms_threshold) override;
    int getNumClasses() const override { return num_classes_; }
    static ModelType getModelType() { return ModelType::YOLOV5; }

   protected:
    int initModel() override;

   private:
    int postprocess(const PreprocessMeta& meta, float conf_threshold, float nms_threshold,
                    std::vector<Detection>& detections);
    int decodeBranch(int output_index, int stride, const int anchors[6], const PreprocessMeta& meta,
                     float conf_threshold, std::vector<Detection>& detections);
    int num_classes_ = 0;
};

}  // namespace rknn_deploy
#endif
