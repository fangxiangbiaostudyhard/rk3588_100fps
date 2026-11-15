#include "api/rknn_detector_c.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "api/rknn_detector.hpp"
#include "opencv2/imgcodecs.hpp"

using rknn_deploy::ErrorCode;
using rknn_deploy::ModelType;
using rknn_deploy::RknnDetector;

namespace {
struct CHandle { std::shared_ptr<RknnDetector> detector; };

rknn_error_code_t ec(ErrorCode c) { return static_cast<rknn_error_code_t>(static_cast<int>(c)); }
ModelType mt(rknn_model_type_t t) { return static_cast<ModelType>(static_cast<int>(t)); }

rknn_error_code_t copy_results(const rknn_deploy::DetectionResult& r, rknn_detection_t** out, int* count) {
    if (!out || !count) return RKNN_ERR_INVALID_ARG;
    *out = nullptr;
    *count = 0;
    if (!r.IsSuccess()) return ec(r.error_code);
    if (r.detections.empty()) return RKNN_SUCCESS;
    auto* arr = static_cast<rknn_detection_t*>(std::calloc(r.detections.size(), sizeof(rknn_detection_t)));
    if (!arr) return RKNN_ERR_OUT_OF_MEMORY;
    for (size_t i = 0; i < r.detections.size(); ++i) {
        const auto& d = r.detections[i];
        arr[i].label_id = d.label_id;
        arr[i].score = d.score;
        arr[i].bbox.left = d.bbox.left;
        arr[i].bbox.top = d.bbox.top;
        arr[i].bbox.right = d.bbox.right;
        arr[i].bbox.bottom = d.bbox.bottom;
        std::snprintf(arr[i].label_name, sizeof(arr[i].label_name), "%s", d.label_name.c_str());
    }
    *out = arr;
    *count = static_cast<int>(r.detections.size());
    return RKNN_SUCCESS;
}
}

extern "C" {

rknn_error_code_t rknn_detector_create(rknn_detector_t* detector, int worker_count) {
    if (!detector) return RKNN_ERR_INVALID_ARG;
    try {
        auto* h = new CHandle;
        h->detector = RknnDetector::Create(worker_count > 0 ? worker_count : 3);
        *detector = h;
        return RKNN_SUCCESS;
    } catch (...) { return RKNN_ERR_OUT_OF_MEMORY; }
}

void rknn_detector_destroy(rknn_detector_t detector) { delete static_cast<CHandle*>(detector); }

rknn_error_code_t rknn_detector_load_model(rknn_detector_t detector, const char* name, const char* path,
                                            rknn_model_type_t type, int pool_size) {
    if (!detector || !name || !path) return RKNN_ERR_INVALID_ARG;
    return ec(static_cast<CHandle*>(detector)->detector->LoadModel(name, path, mt(type), pool_size));
}

rknn_error_code_t rknn_detector_unload_model(rknn_detector_t detector, const char* name) {
    if (!detector || !name) return RKNN_ERR_INVALID_ARG;
    return ec(static_cast<CHandle*>(detector)->detector->UnloadModel(name));
}
int rknn_detector_has_model(rknn_detector_t detector, const char* name) {
    return detector && name && static_cast<CHandle*>(detector)->detector->HasModel(name);
}
int rknn_detector_get_model_count(rknn_detector_t detector) {
    return detector ? static_cast<CHandle*>(detector)->detector->GetModelCount() : 0;
}

rknn_error_code_t rknn_detector_detect_from_file(rknn_detector_t detector, const char* name, const char* path,
                                                  float threshold, rknn_detection_t** results, int* count) {
    if (!detector || !name || !path || !results || !count) return RKNN_ERR_INVALID_ARG;
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return RKNN_ERR_FILE_NOT_EXIST;
    return copy_results(static_cast<CHandle*>(detector)->detector->Detect(name, img, threshold), results, count);
}

rknn_error_code_t rknn_detector_detect_from_ptr(rknn_detector_t detector, const char* name, uint8_t* data,
                                                 int width, int height, int channels, float threshold,
                                                 rknn_detection_t** results, int* count) {
    if (!detector || !name || !data || width <= 0 || height <= 0 || channels != 3 || !results || !count)
        return RKNN_ERR_INVALID_ARG;
    cv::Mat img(height, width, CV_8UC3, data);
    return copy_results(static_cast<CHandle*>(detector)->detector->Detect(name, img, threshold), results, count);
}

rknn_error_code_t rknn_detector_detect_from_dmabuf(rknn_detector_t detector, const char* name,
                                                     const rknn_dmabuf_image_t* image, float threshold,
                                                     rknn_detection_t** results, int* count) {
    if (!detector || !name || !image || !results || !count || image->fd < 0) return RKNN_ERR_INVALID_ARG;
    rknn_deploy::DmaBufImage d;
    d.fd = image->fd;
    d.virt_addr = image->virt_addr;
    d.width = image->width;
    d.height = image->height;
    d.w_stride = image->w_stride;
    d.h_stride = image->h_stride;
    d.format = static_cast<rknn_deploy::PixelFormat>(static_cast<int>(image->format));
    return copy_results(static_cast<CHandle*>(detector)->detector->DetectDmaBuf(name, d, threshold), results, count);
}

void rknn_detector_free_results(rknn_detection_t* results) { std::free(results); }
const char* rknn_detector_get_version(void) { return RknnDetector::GetVersion(); }
const char* rknn_detector_get_error_string(rknn_error_code_t code) {
    return rknn_deploy::GetErrorString(static_cast<ErrorCode>(static_cast<int>(code)));
}
const char* rknn_detector_get_model_type_string(rknn_model_type_t type) {
    return rknn_deploy::GetModelTypeString(mt(type));
}
rknn_error_code_t rknn_detector_load_labels(rknn_detector_t detector, const char* name, const char* path) {
    if (!detector || !name || !path) return RKNN_ERR_INVALID_ARG;
    return ec(static_cast<CHandle*>(detector)->detector->LoadLabels(name, path));
}
rknn_error_code_t rknn_detector_set_nms_threshold(rknn_detector_t detector, const char* name, float threshold) {
    if (!detector || !name) return RKNN_ERR_INVALID_ARG;
    return ec(static_cast<CHandle*>(detector)->detector->SetNmsThreshold(name, threshold));
}
rknn_error_code_t rknn_detector_set_conf_threshold(rknn_detector_t detector, const char* name, float threshold) {
    if (!detector || !name) return RKNN_ERR_INVALID_ARG;
    return ec(static_cast<CHandle*>(detector)->detector->SetConfThreshold(name, threshold));
}

}  // extern C
