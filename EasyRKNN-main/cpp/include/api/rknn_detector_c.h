/**
 * @file rknn_detector_c.h
 * @brief RKNN Deploy SDK - C ABI 接口
 * @version 1.0.0
 *
 * 提供 C 风格接口，支持跨语言调用 (Python/C#/etc.)
 *
 * Usage:
 * @code
 *   // 创建检测器
 *   rknn_detector_t detector;
 *   rknn_detector_create(&detector, 3);
 *
 *   // 加载模型
 *   rknn_detector_load_model(detector, "yolo", "model.rknn", RKNN_MODEL_YOLOV5, 3);
 *
 *   // 推理
 *   rknn_detection_t* results;
 *   int count;
 *   rknn_detector_detect_from_file(detector, "yolo", "test.jpg", 0.5f, &results, &count);
 *
 *   // 处理结果
 *   for (int i = 0; i < count; i++) {
 *       printf("Class %d: %.1f%%\n", results[i].label_id, results[i].score * 100);
 *   }
 *
 *   // 释放结果
 *   rknn_detector_free_results(results);
 *
 *   // 销毁检测器
 *   rknn_detector_destroy(detector);
 * @endcode
 */

#ifndef RKNN_DETECTOR_C_H
#define RKNN_DETECTOR_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 平台导出宏
// ============================================================================
#ifdef _WIN32
#ifdef RKNN_DEPLOY_EXPORTS
#define RKNN_API __declspec(dllexport)
#else
#define RKNN_API __declspec(dllimport)
#endif
#else
#define RKNN_API __attribute__((visibility("default")))
#endif

// ============================================================================
// 类型定义
// ============================================================================

/** 检测器句柄 */
typedef void* rknn_detector_t;

/** 错误码 */
typedef enum {
    RKNN_SUCCESS = 0,
    RKNN_ERR_INVALID_ARG = -1,
    RKNN_ERR_MODEL_NOT_FOUND = -2,
    RKNN_ERR_MODEL_EXISTS = -3,
    RKNN_ERR_INFERENCE_FAILED = -4,
    RKNN_ERR_OUT_OF_MEMORY = -5,
    RKNN_ERR_FILE_NOT_EXIST = -6,
    RKNN_ERR_INIT_FAILED = -7,
    RKNN_ERR_UNSUPPORTED = -8,
    RKNN_ERR_UNKNOWN = -99
} rknn_error_code_t;

/** 模型类型 */
typedef enum {
    RKNN_MODEL_YOLOV5 = 0,
    RKNN_MODEL_YOLOV7 = 1,
    RKNN_MODEL_YOLOV8 = 2,
    RKNN_MODEL_YOLOV8_SEG = 3,
    RKNN_MODEL_YOLOV8_POSE = 4,
    RKNN_MODEL_YOLOV9 = 5,
    RKNN_MODEL_YOLOV10 = 6,
    RKNN_MODEL_YOLOV11 = 7,
    RKNN_MODEL_CUSTOM = 99
} rknn_model_type_t;


/** DMA-BUF pixel format */
typedef enum {
    RKNN_PIXFMT_BGR888 = 0,
    RKNN_PIXFMT_RGB888 = 1,
    RKNN_PIXFMT_NV12 = 2,
    RKNN_PIXFMT_NV21 = 3
} rknn_pixel_format_t;

/** External DMA-BUF image descriptor. fd ownership remains with caller for sync calls. */
typedef struct {
    int fd;
    void* virt_addr;
    int width;
    int height;
    int w_stride;
    int h_stride;
    rknn_pixel_format_t format;
} rknn_dmabuf_image_t;

/** 边界框 */
typedef struct {
    float left;
    float top;
    float right;
    float bottom;
} rknn_bbox_t;

/** 单个检测结果 */
typedef struct {
    int label_id;        /**< 类别ID */
    float score;         /**< 置信度 [0, 1] */
    rknn_bbox_t bbox;    /**< 边界框 */
    char label_name[64]; /**< 类别名称 */
} rknn_detection_t;

/** 检测结果集合 */
typedef struct {
    rknn_detection_t* detections; /**< 检测结果数组 */
    int count;                    /**< 检测数量 */
    int image_width;              /**< 图像宽度 */
    int image_height;             /**< 图像高度 */
    float inference_time_ms;      /**< 推理耗时 */
    rknn_error_code_t error_code; /**< 错误码 */
} rknn_result_t;

/** 模型信息 */
typedef struct {
    char name[64];    /**< 模型名称 */
    char path[256];   /**< 模型路径 */
    int model_type;   /**< 模型类型 */
    int pool_size;    /**< 池大小 */
    int input_width;  /**< 输入宽度 */
    int input_height; /**< 输入高度 */
} rknn_model_info_t;

// ============================================================================
// 核心 API
// ============================================================================

/**
 * @brief 创建检测器实例
 * @param[out] detector 检测器句柄
 * @param[in] npu_core_count worker/context concurrency hint (physical RK3588 NPU cores remain 3)
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_create(rknn_detector_t* detector, int npu_core_count);

/**
 * @brief 销毁检测器并释放所有资源
 * @param[in] detector 检测器句柄
 */
RKNN_API void rknn_detector_destroy(rknn_detector_t detector);

/**
 * @brief 加载模型
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称 (唯一标识)
 * @param[in] model_path 模型文件路径 (.rknn)
 * @param[in] model_type 模型类型
 * @param[in] pool_size 模型池大小 (0 = 使用NPU核心数)
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_load_model(rknn_detector_t detector, const char* model_name,
                                                    const char* model_path, rknn_model_type_t model_type,
                                                    int pool_size);

/**
 * @brief 卸载模型
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_unload_model(rknn_detector_t detector, const char* model_name);

/**
 * @brief 检查模型是否已加载
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @return 1 = 已加载, 0 = 未加载
 */
RKNN_API int rknn_detector_has_model(rknn_detector_t detector, const char* model_name);

/**
 * @brief 获取已加载模型数量
 * @param[in] detector 检测器句柄
 * @return 模型数量
 */
RKNN_API int rknn_detector_get_model_count(rknn_detector_t detector);

// ============================================================================
// 推理 API
// ============================================================================

/**
 * @brief 从文件执行检测
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @param[in] image_path 图像文件路径
 * @param[in] threshold 置信度阈值
 * @param[out] results 检测结果数组 (需调用 rknn_detector_free_results 释放)
 * @param[out] count 检测数量
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_detect_from_file(rknn_detector_t detector, const char* model_name,
                                                          const char* image_path, float threshold,
                                                          rknn_detection_t** results, int* count);

/**
 * @brief 从内存执行检测
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @param[in] image_data 图像数据指针 (BGR格式)
 * @param[in] width 图像宽度
 * @param[in] height 图像高度
 * @param[in] channels 通道数 (通常为3)
 * @param[in] threshold 置信度阈值
 * @param[out] results 检测结果数组
 * @param[out] count 检测数量
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_detect_from_ptr(rknn_detector_t detector, const char* model_name,
                                                         uint8_t* image_data, int width, int height, int channels,
                                                         float threshold, rknn_detection_t** results, int* count);


/**
 * @brief DMA-BUF zero-copy source detection. RGA reads source fd directly and writes to RKNN input DMA memory.
 */
RKNN_API rknn_error_code_t rknn_detector_detect_from_dmabuf(rknn_detector_t detector, const char* model_name,
                                                             const rknn_dmabuf_image_t* image, float threshold,
                                                             rknn_detection_t** results, int* count);

/**
 * @brief 释放检测结果内存
 * @param[in] results 检测结果数组
 */
RKNN_API void rknn_detector_free_results(rknn_detection_t* results);

// ============================================================================
// 辅助 API
// ============================================================================

/**
 * @brief 获取 SDK 版本号
 * @return 版本字符串 (静态字符串，无需释放)
 */
RKNN_API const char* rknn_detector_get_version(void);

/**
 * @brief 获取错误码描述
 * @param[in] code 错误码
 * @return 描述字符串
 */
RKNN_API const char* rknn_detector_get_error_string(rknn_error_code_t code);

/**
 * @brief 获取模型类型名称
 * @param[in] type 模型类型
 * @return 类型名称字符串
 */
RKNN_API const char* rknn_detector_get_model_type_string(rknn_model_type_t type);

/**
 * @brief 加载标签文件
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @param[in] labels_path 标签文件路径
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_load_labels(rknn_detector_t detector, const char* model_name,
                                                     const char* labels_path);

/**
 * @brief 设置 NMS 阈值
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @param[in] threshold NMS 阈值
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_set_nms_threshold(rknn_detector_t detector, const char* model_name,
                                                           float threshold);

/**
 * @brief 设置置信度阈值
 * @param[in] detector 检测器句柄
 * @param[in] model_name 模型名称
 * @param[in] threshold 置信度阈值
 * @return 错误码
 */
RKNN_API rknn_error_code_t rknn_detector_set_conf_threshold(rknn_detector_t detector, const char* model_name,
                                                            float threshold);

#ifdef __cplusplus
}
#endif

#endif  // RKNN_DETECTOR_C_H
