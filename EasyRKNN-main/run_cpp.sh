#!/usr/bin/env bash
# Portable entry point for the RK3588 C++ detector. Run from any directory.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${PROJECT_DIR}/cpp/install/rknn_deploy_Linux/bin/rknn_detector_demo"
model="${PROJECT_DIR}/model/rk3588/yolo11n.rknn"
video="${PROJECT_DIR}/../test.mp4"
type=v11
threads=3
show=0
print_results=0
labels=""
resize=stretch
decode_size=640x640
allow_large_frame=0

usage() {
    cat <<'HELP'
用法: ./run_cpp.sh [选项]

不带参数：使用 model/rk3588/yolo11n.rknn 和项目上级的 test.mp4。
  --video PATH          MP4 文件路径；也可传单数字摄像头编号（默认 ../test.mp4）
  --model PATH          RKNN 模型路径（默认 model/rk3588/yolo11n.rknn）
  --type v5|v8|v11      检测模型类型（默认 v11）
  --threads N           推理工作线程和 RKNN context 数（默认 3）
  --show                在板卡的图形桌面显示画面（默认关闭）
  --print-results       逐帧打印检测类别、置信度和检测框（默认关闭）
  --labels PATH         类别文件，每行一个名称
  --resize stretch|letterbox  模型预处理模式（默认 stretch）
  --decode-size WxH     MP4 软件解码后的尺寸（默认 640x640）
  --allow-large-frame   允许大于 640 的解码尺寸；当前板卡未验证
  -h, --help            显示帮助

示例:
  ./run_cpp.sh
  ./run_cpp.sh --video /data/video.mp4 --threads 2 --show
  ./run_cpp.sh --model /data/my.rknn --type v11 --labels /data/classes.txt --video /data/video.mp4
HELP
}

fail() { printf '错误: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "$1 缺少参数"; }
resolve_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        realpath -e -- "$path"
    elif [[ -f "${PROJECT_DIR}/${path}" ]]; then
        realpath -e -- "${PROJECT_DIR}/${path}"
    else
        fail "找不到文件: ${path}"
    fi
}

while (($#)); do
    case "$1" in
        --video) need_value "$@"; video="$2"; shift 2 ;;
        --model) need_value "$@"; model="$2"; shift 2 ;;
        --type) need_value "$@"; type="$2"; shift 2 ;;
        --threads) need_value "$@"; threads="$2"; shift 2 ;;
        --show) show=1; shift ;;
        --print-results) print_results=1; shift ;;
        --labels) need_value "$@"; labels="$2"; shift 2 ;;
        --resize) need_value "$@"; resize="$2"; shift 2 ;;
        --decode-size) need_value "$@"; decode_size="$2"; shift 2 ;;
        --allow-large-frame) allow_large_frame=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "未知选项: $1（运行 --help 查看用法）" ;;
    esac
done

[[ "$type" == v5 || "$type" == v8 || "$type" == v11 ]] || fail "--type 只能是 v5、v8、v11"
[[ "$resize" == stretch || "$resize" == letterbox ]] || fail "--resize 只能是 stretch 或 letterbox"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || fail "--threads 必须是正整数"
[[ "$decode_size" =~ ^([1-9][0-9]*)[xX]([1-9][0-9]*)$ ]] || fail "--decode-size 格式应为 640x640"
decode_width="${BASH_REMATCH[1]}"
decode_height="${BASH_REMATCH[2]}"
if (( (decode_width > 640 || decode_height > 640) && !allow_large_frame )); then
    fail "本板卡大于 640 的 RGA 输入曾导致失联；确认目标机可用后加 --allow-large-frame"
fi

[[ -x "$BINARY" ]] || fail "未找到可执行文件，请先运行 ${PROJECT_DIR}/cpp/build-linux_RK3588.sh"
model="$(resolve_file "$model")"
if [[ "$video" =~ ^[0-9]$ ]]; then
    : # OpenCV camera index
else
    video="$(resolve_file "$video")"
fi
if [[ -n "$labels" ]]; then labels="$(resolve_file "$labels")"; fi
if ((show)) && [[ -z "${DISPLAY:-}" ]] && [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
fi
if ((show)) && [[ -z "${DISPLAY:-}" ]]; then
    fail "没有图形桌面；请在板卡桌面运行，或设置 DISPLAY=:0"
fi

printf '模型: %s\n输入: %s\n类型: %s  线程/context: %s  显示: %s  解码尺寸: %sx%s\n' \
    "$model" "$video" "$type" "$threads" "$show" "$decode_width" "$decode_height"
exec "$BINARY" "$model" "$type" "$video" "$threads" "$show" "$labels" "$resize" "$decode_width" "$decode_height" "$print_results"
