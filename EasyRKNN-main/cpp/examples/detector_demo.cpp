#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <vector>

#include "api/rknn_detector.hpp"
#include "core/safe_queue.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"

using namespace rknn_deploy;

struct ResultPackage {
    cv::Mat frame;
    DetectionResult result;
    uint64_t frame_id = 0;
};

class DetectorApp {
   public:
    DetectorApp(const std::string& model_path, const std::string& type, int concurrency, bool show_ui,
                const std::string& labels, ResizeMode resize_mode, bool print_results)
        : concurrency_(std::max(1, concurrency)), show_ui_(show_ui),
          print_results_(print_results), result_queue_(show_ui ? 4 : 0) {
        ModelType mt;
        if (type == "v5") mt = ModelType::YOLOV5;
        else if (type == "v8") mt = ModelType::YOLOV8;
        else if (type == "v11") mt = ModelType::YOLOV11;
        else throw std::invalid_argument("model_type must be v5/v8/v11");

        detector_ = RknnDetector::Create(concurrency_);
        ModelConfig cfg;
        cfg.model_path = model_path;
        cfg.type = mt;
        cfg.pool_size = concurrency_;
        cfg.resize_mode = resize_mode;
        cfg.zero_copy_io = true;
        cfg.labels_path = labels;
        auto ec = detector_->LoadModel("main", cfg);
        if (ec != ErrorCode::Success) throw std::runtime_error(GetErrorString(ec));

        auto info = detector_->GetModelInfo("main");
        std::printf("[APP] model=%s input=%dx%d classes=%d contexts=%d zero-copy=%s\n",
                    model_path.c_str(), info.input_width, info.input_height, info.num_classes,
                    info.pool_info.max_size, info.zero_copy_io ? "ON" : "OFF");
    }

    int run(const std::string& source, int decode_width, int decode_height) {
        cv::VideoCapture cap;
        std::string lower_source = source;
        std::transform(lower_source.begin(), lower_source.end(), lower_source.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (lower_source.size() >= 4 && lower_source.compare(lower_source.size() - 4, 4, ".mp4") == 0) {
            // Use software H.264 decode and scale before RGA. The board's direct MP4 path is unstable.
            std::string escaped_source;
            for (char ch : source) {
                if (ch == '"' || ch == '\\') escaped_source += '\\';
                escaped_source += ch;
            }
            const std::string pipeline =
                "filesrc location=\"" + escaped_source + "\" ! qtdemux ! h264parse ! avdec_h264 ! "
                "videoscale ! video/x-raw,width=" + std::to_string(decode_width) +
                ",height=" + std::to_string(decode_height) + " ! videoconvert ! "
                "video/x-raw,format=BGR ! appsink sync=false";
            cap.open(pipeline, cv::CAP_GSTREAMER);
        } else if (source.size() == 1 && std::isdigit(source[0])) {
            cap.open(source[0] - '0');
        } else {
            cap.open(source);
        }
        if (!cap.isOpened()) {
            std::fprintf(stderr, "cannot open %s\n", source.c_str());
            return -1;
        }
        if (show_ui_) {
            cv::namedWindow("EasyRKNN zero-copy", cv::WINDOW_NORMAL);
            cv::resizeWindow("EasyRKNN zero-copy", 1280, 720);
        }
        completed_fps_.store(0.0);
        latest_ui_.reset();
        last_ui_pump_ = {};

        auto stat_t0 = std::chrono::steady_clock::now();
        uint64_t submitted = 0;
        uint64_t last_completed = 0;
        bool quit = false;

        while (!quit) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            const uint64_t id = ++submitted;
            pending_.fetch_add(1);

            // No frame.clone(): cv::Mat refcount keeps OpenCV-owned memory alive for the async task.
            cv::Mat frame_ref = frame;
            auto ec = detector_->DetectAsync("main", frame_ref,
                [this, id, frame_ref](const DetectionResult& result) mutable {
                    if (print_results_) {
                        std::lock_guard<std::mutex> lock(output_mutex_);
                        if (result.IsSuccess()) {
                            std::printf("[DETECT] frame=%llu count=%zu inference=%.1fms\n",
                                        static_cast<unsigned long long>(id),
                                        result.detections.size(), result.inference_time_ms);
                            for (const auto& det : result.detections) {
                                const std::string name = det.label_name.empty()
                                    ? "class " + std::to_string(det.label_id) : det.label_name;
                                std::printf("  %s id=%d score=%.3f box=(%.1f, %.1f, %.1f, %.1f)\n",
                                            name.c_str(), det.label_id, det.score,
                                            det.bbox.left, det.bbox.top,
                                            det.bbox.right, det.bbox.bottom);
                            }
                        } else {
                            std::fprintf(stderr, "[DETECT] frame=%llu error=%s\n",
                                         static_cast<unsigned long long>(id),
                                         result.error_message.c_str());
                        }
                    }
                    if (show_ui_) {
                        ResultPackage p;
                        p.frame = frame_ref;
                        p.result = result;
                        p.frame_id = id;
                        // UI queue is deliberately bounded. If full, drop display work rather than stall inference.
                        result_queue_.try_push(std::move(p));
                    }
                    if (result.IsSuccess()) completed_.fetch_add(1);
                    pending_.fetch_sub(1);
                });
            if (ec != ErrorCode::Success) pending_.fetch_sub(1);

            if (show_ui_) quit = pump_ui();

            auto now = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - stat_t0).count();
            if (ms >= 1000) {
                uint64_t done = completed_.load();
                double fps = (done - last_completed) * 1000.0 / ms;
                completed_fps_.store(fps);
                auto pool = detector_->GetPoolInfo("main");
                std::printf("[STATS] throughput=%.2f FPS pending=%d queue=%zu ctx(active/idle/max)=%d/%d/%d\n",
                            fps, pending_.load(), detector_->GetPendingTaskCount(),
                            pool.active_count, pool.idle_count, pool.max_size);
                last_completed = done;
                stat_t0 = now;
            }
        }

        while (pending_.load() > 0) {
            if (show_ui_) pump_ui();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }

   private:
    bool pump_ui() {
        ResultPackage p;
        // Keep only the newest result so UI rendering cannot build up a backlog.
        while (result_queue_.try_pop(p)) {
            latest_ui_ = std::move(p);
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_ui_pump_ != std::chrono::steady_clock::time_point{} &&
            now - last_ui_pump_ < std::chrono::milliseconds(33)) {
            return false;
        }
        last_ui_pump_ = now;
        if (latest_ui_) {
            ResultPackage newest = std::move(*latest_ui_);
            latest_ui_.reset();
            DrawDetections(newest.frame, newest.result.detections);
            std::map<int, std::pair<std::string, int>> class_counts;
            for (const auto& detection : newest.result.detections) {
                auto& entry = class_counts[detection.label_id];
                ++entry.second;
                if (!detection.label_name.empty()) entry.first = detection.label_name;
            }
            std::vector<std::pair<std::string, int>> classes;
            classes.reserve(class_counts.size());
            for (const auto& item : class_counts) {
                const std::string name = item.second.first.empty()
                    ? "class " + std::to_string(item.first) : item.second.first;
                classes.emplace_back(name, item.second.second);
            }
            std::sort(classes.begin(), classes.end(), [](const auto& a, const auto& b) {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            });

            char line[128];
            std::vector<std::string> lines;
            std::snprintf(line, sizeof(line), "Inference FPS: %.1f", completed_fps_.load());
            lines.emplace_back(line);
            std::snprintf(line, sizeof(line), "Detect: %.1f ms   Pending: %d",
                          newest.result.inference_time_ms, pending_.load());
            lines.emplace_back(line);
            std::snprintf(line, sizeof(line), "Frame: %llu   Objects: %zu",
                          static_cast<unsigned long long>(newest.frame_id),
                          newest.result.detections.size());
            lines.emplace_back(line);
            if (classes.empty()) lines.emplace_back("Classes: none");
            for (size_t i = 0; i < std::min<size_t>(classes.size(), 3); ++i) {
                lines.emplace_back(classes[i].first + ": " + std::to_string(classes[i].second));
            }

            const int line_height = 25;
            const int panel_width = std::min(newest.frame.cols, 440);
            const int panel_height = std::min(newest.frame.rows,
                                              static_cast<int>(lines.size()) * line_height + 8);
            cv::rectangle(newest.frame, cv::Rect(0, 0, panel_width, panel_height),
                          cv::Scalar(20, 20, 20), cv::FILLED);
            for (size_t i = 0; i < lines.size(); ++i) {
                const int y = 21 + static_cast<int>(i) * line_height;
                if (y >= newest.frame.rows) break;
                cv::putText(newest.frame, lines[i], cv::Point(10, y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255),
                            1, cv::LINE_AA);
            }
            cv::imshow("EasyRKNN zero-copy", newest.frame);
        }
        return cv::waitKey(1) == 'q';
    }

    std::shared_ptr<RknnDetector> detector_;
    int concurrency_ = 1;
    bool show_ui_ = false;
    bool print_results_ = false;
    std::mutex output_mutex_;
    std::atomic<int> pending_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<double> completed_fps_{0.0};
    SafeQueue<ResultPackage> result_queue_;
    std::optional<ResultPackage> latest_ui_;
    std::chrono::steady_clock::time_point last_ui_pump_{};
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("Usage: %s <model.rknn> <v5|v8|v11> <video|camera_id> [contexts=3] [show_ui=0] [labels=''] [stretch|letterbox] [decode_width=640] [decode_height=640] [print_results=0]\n", argv[0]);
        return 1;
    }
    try {
        int contexts = argc > 4 ? std::atoi(argv[4]) : 3;
        bool show = argc > 5 ? std::atoi(argv[5]) != 0 : false;
        std::string labels = argc > 6 ? argv[6] : "";
        std::string mode = argc > 7 ? argv[7] : "stretch";
        ResizeMode resize_mode = mode == "letterbox" ? ResizeMode::Letterbox : ResizeMode::Stretch;
        int decode_width = argc > 8 ? std::atoi(argv[8]) : 640;
        int decode_height = argc > 9 ? std::atoi(argv[9]) : 640;
        bool print_results = argc > 10 ? std::atoi(argv[10]) != 0 : false;
        if (decode_width <= 0 || decode_height <= 0)
            throw std::invalid_argument("decode width/height must be positive");
        DetectorApp app(argv[1], argv[2], contexts, show, labels, resize_mode, print_results);
        return app.run(argv[3], decode_width, decode_height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 2;
    }
}
