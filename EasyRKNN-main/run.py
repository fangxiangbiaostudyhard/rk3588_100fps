#!/usr/bin/env python3
"""Configure and launch the RK3588 C++ detector."""

import argparse
import subprocess
import sys
from pathlib import Path


# 修改这里即可；直接运行 python3 run.py 不会询问参数。
MODEL_PATH = "model/rk3588/yolo11n.rknn"
MODEL_TYPE = "v11"             # v5 / v8 / v11，须与模型匹配
SOURCE_MODE = "video"          # video 或 camera
VIDEO_PATH = "../test.mp4"
CAMERA_INDEX = 0               # 0～9
THREADS = 12                    # 推理线程和 RKNN context 数
SHOW_WINDOW = True            # True：显示检测画面
PRINT_RESULTS = False          # True：逐帧打印类别、置信度和检测框
LABELS_PATH = "model/coco_80_labels_list.txt"  # 内置 COCO 模型的类别名称
RESIZE_MODE = "stretch"        # stretch 或 letterbox
DECODE_SIZE = "640x640"        # MP4 软件解码尺寸
ALLOW_LARGE_FRAME = False      # 本板卡未验证大于 640 的输入


ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "run_cpp.sh"


def main():
    parser = argparse.ArgumentParser(description="运行 RK3588 目标检测；默认值写在 run.py 顶部。")
    parser.add_argument("--model", default=MODEL_PATH, help="RKNN 模型路径")
    parser.add_argument("--type", choices=("v5", "v8", "v11"), default=MODEL_TYPE,
                        help="模型类型")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="视频文件路径")
    source.add_argument("--camera", type=int, help="摄像头编号 0–9")
    parser.add_argument("--threads", type=int, default=THREADS,
                        help="推理线程/context 数")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true", default=SHOW_WINDOW,
                         help="显示检测画面")
    display.add_argument("--hide", dest="show", action="store_false", help="不显示检测画面")
    printing = parser.add_mutually_exclusive_group()
    printing.add_argument("--print-results", dest="print_results", action="store_true",
                          default=PRINT_RESULTS, help="逐帧打印检测结果")
    printing.add_argument("--no-print-results", dest="print_results", action="store_false",
                          help="不打印逐帧检测结果")
    parser.add_argument("--labels", default=LABELS_PATH, help="类别名称文件路径")
    parser.add_argument("--resize", choices=("stretch", "letterbox"),
                        default=RESIZE_MODE, help="预处理缩放方式")
    parser.add_argument("--decode-size", default=DECODE_SIZE,
                        help="MP4 软件解码尺寸")
    parser.add_argument("--allow-large-frame", action="store_true",
                        default=ALLOW_LARGE_FRAME, help="允许大于 640 的解码尺寸")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads 必须是正整数")
    if not LAUNCHER.is_file():
        parser.error(f"找不到启动脚本：{LAUNCHER}")
    if args.video is not None:
        video = args.video
    elif args.camera is not None:
        if not 0 <= args.camera <= 9:
            parser.error("--camera 必须是 0–9")
        video = str(args.camera)
    elif SOURCE_MODE == "video":
        video = VIDEO_PATH
    elif SOURCE_MODE == "camera":
        if not 0 <= CAMERA_INDEX <= 9:
            parser.error("CAMERA_INDEX 必须是 0–9")
        video = str(CAMERA_INDEX)
    else:
        parser.error("SOURCE_MODE 必须是 video 或 camera")

    command = [str(LAUNCHER), "--model", args.model, "--type", args.type,
               "--video", video, "--threads", str(args.threads),
               "--resize", args.resize, "--decode-size", args.decode_size]
    if args.show:
        command.append("--show")
    if args.print_results:
        command.append("--print-results")
    if args.labels:
        command.extend(("--labels", args.labels))
    if args.allow_large_frame:
        command.append("--allow-large-frame")
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
