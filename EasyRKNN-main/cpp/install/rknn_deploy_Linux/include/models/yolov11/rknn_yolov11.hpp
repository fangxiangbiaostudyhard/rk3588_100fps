#ifndef RKNN_YOLOV11_HPP
#define RKNN_YOLOV11_HPP

#include "api/model_base.hpp"

namespace rknn_deploy {

class rkYolov11 : public RknnModelBase {
   public:
    explicit rkYolov11(const std::string& model_path) : RknnModelBase(model_path, "YOLOv11") {}

    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold, float nms_threshold) override;
    std::vector<Detection> detectDmaBuf(const DmaBufImage& image, float conf_threshold,
                                        float nms_threshold) override;
    int getNumClasses() const override { return num_classes_; }
    static ModelType getModelType() { return ModelType::YOLOV11; }

   protected:
    int initModel() override;

   private:
    int postprocess(const PreprocessMeta& meta, float conf_threshold, float nms_threshold,
                    std::vector<Detection>& detections);
    int num_classes_ = 0;
    int output_per_branch_ = 0;
    bool multi_output_ = false;
};

}  // namespace rknn_deploy
#endif
