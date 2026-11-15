# EasyRKNN RK3588 实用版 / RK3588 Edition

[中文](#中文) · [English](#english)

<a id="中文"></a>

## 中文

**本项目作者：方向标studyhard。** 本项目基于 [Onelevenvy/EasyRKNN](https://github.com/Onelevenvy/EasyRKNN) 二次开发，面向 RK3588 板卡上的 YOLOv5、YOLOv8 和 YOLO11 目标检测。模型推理、视频处理和画面绘制由 C++ 程序完成；`run.py` 只负责读取配置并启动该程序。

目前仅在一块 RK3588 香橙派板卡上验证。其他 Rockchip 芯片、其他系统镜像、任意自训模型及摄像头输入需要自行验证。项目沿用原项目的 [Apache-2.0 许可证](LICENSE)。

### 相对原版的主要改动

- **修正 RGA 预处理调用：** 调整 `wrapbuffer_virtualaddr` 和 `wrapbuffer_fd` 的参数顺序，使像素格式及 stride 参数与当前 RGA 头文件一致。
- **增加适合本板卡的 MP4 路径：** H.264 MP4 使用 GStreamer 软件解码、预缩放，再进入 RGA 和 RKNN。默认解码尺寸为 640×640；超过 640 的尺寸需显式放行，因为本板卡上较大的 RGA 输入曾导致失联。摄像头和其他文件类型仍使用 OpenCV 常规输入路径。
- **改善显示：** 显示队列保留最新结果，窗口约每 33 ms 更新一次，并显示推理 FPS、耗时、目标数量和类别统计。按 `q` 退出窗口。
- **增加可选逐帧结果输出：** 可打印帧号、目标类别、置信度和检测框坐标；关闭时不产生逐帧检测日志。FPS 统计只计入成功的检测。
- **简化板端使用：** 增加可直接编辑配置的 `run.py`、带参数检查的 `run_cpp.sh`，并让编译脚本能处理搬迁项目后失效的 CMake 缓存。

YOLO 后处理、核心检测器和仓库内三个 `.rknn` 模型沿用原版。上述改动不代表模型精度或 FPS 必然高于原版；请用相同硬件、输入和参数进行性能对比。

### 首次使用前准备

| 项目 | 要求 |
| --- | --- |
| 硬件与系统 | 带可用 RKNPU 驱动的 RK3588 Linux 板卡。已验证环境：Orange Pi、Ubuntu 20.04.6 LTS、aarch64。 |
| 编译工具 | CMake ≥ 3.13、支持 C++17 的 GCC/G++、`make`、`pkg-config`。已验证 CMake 3.16.3、G++ 9.4。 |
| 图像与视频 | OpenCV C++ 开发包，且 OpenCV 启用了 GStreamer；GStreamer 的 `qtdemux`、`h264parse`、`avdec_h264`、`videoscale`、`videoconvert`、`appsink` 插件。已验证 OpenCV 4.2.0、GStreamer 1.16.3。 |
| Python | Python 3 用于启动脚本；`run.py` 只用标准库，无需 `pip install`。纯 C++ 运行可使用 `run_cpp.sh`。 |
| 仓库文件 | 检查 `cpp/include/librknnrt.so`、`cpp/include/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so`、`model/rk3588/*.rknn` 是否随仓库下载。当前项目包含这些文件；已验证 RKNN runtime 为 2.3.2。 |

先下载或克隆**本仓库**到 RK3588 板卡，进入仓库根目录。板卡系统镜像及 RKNPU 驱动应从板卡厂商获取；如果需要另行获取或更新 RKNN runtime，请查看 Rockchip 的 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) 与 [RKNPU2](https://github.com/airockchip/rknpu2)，并核对模型、运行库和驱动的兼容性。**仅运行仓库自带模型时，无需在板卡上安装 RKNN-Toolkit2 Python 包。**

在 Ubuntu/Debian 系统上，可安装构建和视频依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

检查环境，以下命令在**仓库根目录**执行：

```bash
uname -m
cmake --version
pkg-config --modversion opencv4
opencv_version --verbose | grep -i GStreamer
for p in qtdemux h264parse avdec_h264 videoscale videoconvert appsink; do
  gst-inspect-1.0 "$p" >/dev/null || echo "缺少插件: $p"
done
ls cpp/include/librknnrt.so cpp/include/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so
```

`uname -m` 应为 `aarch64`，OpenCV 的 GStreamer 项应为 `YES`。若板卡系统缺少 NPU 驱动，仅安装用户态软件不能使 NPU 推理工作。

### 编译与首次验证

在**仓库根目录**运行：

```bash
bash cpp/build-linux_RK3588.sh
```

编译结果位于 `cpp/install/rknn_deploy_Linux/bin/rknn_detector_demo`。不要直接使用另一台机器留下的 `cpp/build/` 或 `cpp/install/` 产物；在目标板卡重新编译。编译脚本发现旧 CMake 路径缓存时，会将旧构建目录移入用户缓存目录后重新配置。

首次验证可用仓库内的单张图片，**无需摄像头、桌面或额外视频文件**：

```bash
python3 run.py --video datasets/bus.jpg --threads 1 --hide --print-results
```

正常情况下会看到 `[APP]` 模型信息和 `[DETECT]` 检测结果。若只想走 Shell 入口：

```bash
./run_cpp.sh --video datasets/bus.jpg --threads 1 --labels model/coco_80_labels_list.txt --print-results
```

### 日常使用

直接运行 `python3 run.py` 不会询问模型或输入源。请先编辑该文件开头的配置：

| 配置项 | 作用 |
| --- | --- |
| `MODEL_PATH` / `MODEL_TYPE` | `.rknn` 路径及对应的 `v5`、`v8` 或 `v11` 类型。 |
| `SOURCE_MODE` | `video` 或 `camera`。视频使用 `VIDEO_PATH`；摄像头使用 0–9 的 `CAMERA_INDEX`。 |
| `THREADS` | 推理工作线程数，同时也是 RKNN context 数；不等于物理 NPU 核心数。 |
| `SHOW_WINDOW` | 是否在板卡图形桌面显示检测画面。 |
| `PRINT_RESULTS` | 是否逐帧打印检测类别、置信度和框坐标。输出大量日志可能降低吞吐。 |
| `LABELS_PATH` | 类别文件。自训模型应使用与模型类别数量一致的文件；不提供时显示类别 ID。 |
| `RESIZE_MODE` / `DECODE_SIZE` | 预处理方式与 H.264 MP4 软件解码后的尺寸。`DECODE_SIZE` 不会改变模型本身的输入尺寸。 |

`VIDEO_PATH` 的预设值 `../test.mp4` 是开发板上的示例路径；首次使用请改成自己的文件，或用命令行指定。命令行参数可临时覆盖文件中的设置：

```bash
python3 run.py --video /path/to/video.mp4 --threads 3 --hide
python3 run.py --camera 0 --threads 3 --show --print-results
python3 run.py --model /path/to/custom.rknn --type v11 \
  --labels /path/to/classes.txt --video /path/to/video.mp4
python3 run.py --help
```

`--show` 需要板卡上可访问的图形桌面；从 SSH 启动时可尝试 `DISPLAY=:0 python3 run.py --show`。无图形环境请使用 `--hide`。摄像头模式接受 OpenCV 的单数字设备编号，尚未在本板卡完成同等程度的稳定性验证。

当前 H.264 MP4 路径会先把画面缩放到 `DECODE_SIZE`，默认 640×640；如果原视频不是正方形，这一步会改变宽高比。需要保留原始比例或使用大尺寸输入时，应针对模型和板卡重新验证预处理及框坐标。程序只在窗口和终端显示结果，默认不保存标注视频或 JSON 文件。

### 自定义模型与排障

仓库内有 YOLOv5s、YOLOv8n、YOLO11n 的 RK3588 `.rknn` 文件。`.pt` 或 `.onnx` 不能直接传给板端检测程序。需要转换自训 ONNX 时，在与官方工具兼容的开发机上按 [RKNN-Toolkit2 安装说明](https://github.com/airockchip/rknn-toolkit2)安装工具，并准备量化校准图片列表，再运行：

```bash
python3 convert_rknn.py /path/to/model.onnx rk3588 \
  --dtype i8 --preset common --dataset /path/to/dataset.txt \
  -o model/rk3588/custom.rknn
```

转换时的输入归一化、尺寸及输出布局必须与训练和本项目后处理代码匹配。现有示例面向 YOLO 检测，不直接支持分割、姿态等任务；自训模型输出布局不同则需修改对应的 `cpp/src/models/` 后处理代码。

- **模型加载失败：** 核对目标芯片、模型输出布局、RKNN runtime 与驱动的兼容性。
- **MP4 无法打开：** 核对文件路径、H.264 编码、OpenCV GStreamer 支持和上述插件。
- **没有画面：** 检查 `SHOW_WINDOW`、图形桌面、`DISPLAY` 与 X 会话权限。
- **FPS 与预期不同：** 视频解码、线程数、窗口绘制和逐帧日志都会影响吞吐；`[STATS]` 显示的是处理吞吐，不是屏幕刷新率。

### 致谢与许可

感谢以下项目。本仓库是原版 EasyRKNN 的衍生开发，保留原项目许可与其对依赖项目的致谢：

- [Onelevenvy/EasyRKNN](https://github.com/Onelevenvy/EasyRKNN)：本项目的上游基础。
- [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)：Rockchip 模型与推理示例。
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)：RKNN 模型转换工具链。
- [leafqycc/rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading)：多线程 C++ 部署思路参考。

本项目作者：**方向标studyhard**。许可证见 [LICENSE](LICENSE)（Apache-2.0）。

---

<a id="english"></a>

## English

**Project author: 方向标studyhard (Fangxiangbiao studyhard).** This project is a derivative of [Onelevenvy/EasyRKNN](https://github.com/Onelevenvy/EasyRKNN) for YOLOv5, YOLOv8, and YOLO11 object detection on RK3588 boards. C++ handles video, preprocessing, NPU inference, and rendering. `run.py` only reads configuration and starts the C++ executable.

The project has been verified on one Orange Pi RK3588 board. Other Rockchip chips, OS images, custom models, and camera inputs require separate validation. It retains the upstream [Apache-2.0 license](LICENSE).

### Main changes from upstream

- **RGA preprocessing fix:** Corrected the argument order of `wrapbuffer_virtualaddr` and `wrapbuffer_fd` to match the bundled RGA headers.
- **Board-specific MP4 path:** H.264 MP4 files use GStreamer software decoding and scaling before RGA and RKNN. The default decoded size is 640×640. Larger sizes require an explicit override because large RGA inputs previously caused this board to disconnect. Cameras and other file types still use the regular OpenCV input path.
- **Display updates:** The UI keeps the newest result, refreshes about every 33 ms, and overlays FPS, inference time, object count, and class counts. Press `q` to close it.
- **Optional per-frame logs:** Print frame number, class, confidence, and bounding box coordinates when enabled. Throughput counts successful detections only.
- **Simpler deployment:** Added editable `run.py` settings, a checked `run_cpp.sh` launcher, and handling for stale CMake paths after moving the project.

The core detector, YOLO postprocessors, and three bundled `.rknn` models are unchanged from upstream. These changes do not establish better accuracy or FPS; compare both versions under the same conditions.

### Requirements and downloads

| Item | Requirement |
| --- | --- |
| Hardware and OS | RK3588 Linux board with a working RKNPU driver. Verified: Orange Pi, Ubuntu 20.04.6 LTS, aarch64. |
| Build tools | CMake ≥ 3.13, C++17-capable GCC/G++, `make`, and `pkg-config`. Verified: CMake 3.16.3 and G++ 9.4. |
| Image and video | OpenCV C++ development files with GStreamer enabled, plus the `qtdemux`, `h264parse`, `avdec_h264`, `videoscale`, `videoconvert`, and `appsink` GStreamer elements. Verified: OpenCV 4.2.0 and GStreamer 1.16.3. |
| Python | Python 3 for `run.py`. It uses only the standard library; no `pip install` is needed. Use `run_cpp.sh` for a pure Shell entry point. |
| Bundled files | Make sure `cpp/include/librknnrt.so`, `cpp/include/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so`, and `model/rk3588/*.rknn` are present. The current project includes them; the verified RKNN runtime is 2.3.2. |

Download or clone **this repository** onto the RK3588 board and enter its root directory. Obtain the board OS image and RKNPU driver from the board vendor. If you must install or update the runtime separately, consult Rockchip's [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) and [RKNPU2](https://github.com/airockchip/rknpu2) repositories and check compatibility among the model, runtime, and driver. **The RKNN-Toolkit2 Python package is not required on the board to run the bundled models.**

On Ubuntu/Debian, install build and video dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

Check the environment from the **repository root**:

```bash
uname -m
cmake --version
pkg-config --modversion opencv4
opencv_version --verbose | grep -i GStreamer
for p in qtdemux h264parse avdec_h264 videoscale videoconvert appsink; do
  gst-inspect-1.0 "$p" >/dev/null || echo "Missing element: $p"
done
ls cpp/include/librknnrt.so cpp/include/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so
```

`uname -m` should report `aarch64`, and OpenCV should report GStreamer `YES`. Installing user-space packages alone does not replace a missing board NPU driver.

### Build and first test

Run from the **repository root**:

```bash
bash cpp/build-linux_RK3588.sh
```

The executable is installed at `cpp/install/rknn_deploy_Linux/bin/rknn_detector_demo`. Rebuild on the target board rather than reusing `cpp/build/` or `cpp/install/` artifacts from another machine. If the build script finds a CMake cache pointing to an old directory, it archives that build directory in the user's cache and configures a fresh build.

The bundled image gives you a first test **without a camera, desktop, or extra video file**:

```bash
python3 run.py --video datasets/bus.jpg --threads 1 --hide --print-results
```

A successful run prints `[APP]` model information and `[DETECT]` results. To use the Shell launcher directly:

```bash
./run_cpp.sh --video datasets/bus.jpg --threads 1 --labels model/coco_80_labels_list.txt --print-results
```

### Everyday use

`python3 run.py` does not prompt for a model or source. Edit the settings at the top of the file:

| Setting | Purpose |
| --- | --- |
| `MODEL_PATH` / `MODEL_TYPE` | Path to a `.rknn` model and its matching `v5`, `v8`, or `v11` type. |
| `SOURCE_MODE` | `video` or `camera`. Set `VIDEO_PATH` for a file, or `CAMERA_INDEX` (0–9) for a camera. |
| `THREADS` | Worker thread and RKNN context count; this is not the number of physical NPU cores. |
| `SHOW_WINDOW` | Show detection frames on the board's graphical desktop. |
| `PRINT_RESULTS` | Print per-frame classes, confidence scores, and boxes. Heavy terminal output can reduce throughput. |
| `LABELS_PATH` | Class-name file. For a custom model, use a file matching its class count; otherwise class IDs are shown. |
| `RESIZE_MODE` / `DECODE_SIZE` | Preprocessing mode and decoded H.264 MP4 frame size. `DECODE_SIZE` does not change the model's input size. |

The preset `VIDEO_PATH`, `../test.mp4`, is specific to the development board. Change it to your own file or pass a path on the command line. CLI options temporarily override file settings:

```bash
python3 run.py --video /path/to/video.mp4 --threads 3 --hide
python3 run.py --camera 0 --threads 3 --show --print-results
python3 run.py --model /path/to/custom.rknn --type v11 \
  --labels /path/to/classes.txt --video /path/to/video.mp4
python3 run.py --help
```

`--show` requires an accessible graphical desktop. From SSH, `DISPLAY=:0 python3 run.py --show` may work if your session has X access; use `--hide` for headless operation. Camera mode accepts a single-digit OpenCV device index and has not received the same stability testing as the MP4 path on this board.

The H.264 MP4 path scales frames to `DECODE_SIZE` before model preprocessing, defaulting to 640×640. This changes the aspect ratio of non-square source videos. If your model needs the original aspect ratio or larger frames, validate preprocessing and box coordinates on your board. The program displays results in the window or terminal; it does not save an annotated video or JSON by default.

### Custom models and troubleshooting

The repository includes RK3588 `.rknn` files for YOLOv5s, YOLOv8n, and YOLO11n. The board-side detector cannot load `.pt` or `.onnx` directly. To convert a custom ONNX model, install [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) on a compatible development machine, prepare a calibration image list, and run:

```bash
python3 convert_rknn.py /path/to/model.onnx rk3588 \
  --dtype i8 --preset common --dataset /path/to/dataset.txt \
  -o model/rk3588/custom.rknn
```

Conversion preprocessing, input size, and output layout must match the trained model and this project's postprocessor. The demo handles YOLO detection; segmentation and pose models are not directly supported. Different output layouts require changes under `cpp/src/models/`.

- **Model fails to load:** Check target chip, output layout, and compatibility among the model, RKNN runtime, and driver.
- **MP4 fails to open:** Check the path, H.264 encoding, OpenCV GStreamer support, and the listed elements.
- **No window appears:** Check `SHOW_WINDOW`, the board desktop, `DISPLAY`, and X session access.
- **Unexpected FPS:** Decoding, thread count, rendering, and per-frame printing affect throughput. `[STATS]` reports processing throughput, not display refresh rate.

### Acknowledgments and license

Thanks to these projects. This repository derives from EasyRKNN and retains its license and acknowledgments:

- [Onelevenvy/EasyRKNN](https://github.com/Onelevenvy/EasyRKNN): upstream project.
- [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo): Rockchip models and inference examples.
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2): RKNN model conversion toolkit.
- [leafqycc/rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading): inspiration for multithreaded C++ deployment.

Project author: **方向标studyhard**. See [LICENSE](LICENSE) for Apache-2.0.
