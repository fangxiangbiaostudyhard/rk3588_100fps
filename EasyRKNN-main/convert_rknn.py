import sys
import os
import argparse
from rknn.api import RKNN


MODEL_PRESETS = {
    'common': { 
        # 适用于 YOLOv5, v8, v11, MobileNet 等大多数模型
        'mean': [0, 0, 0], 
        'std':  [255, 255, 255] 
    },
    'ocr': { 
        # 适用于 PP-OCR, ImageNet 预训练模型 (ResNet 等)
        'mean': [123.675, 116.28, 103.53], 
        'std':  [58.395, 57.12, 57.375]
    }
}

def get_args():
    parser = argparse.ArgumentParser(description="Unified ONNX to RKNN Converter")
    
    # 必选参数
    parser.add_argument('input_model', help='Path to input .onnx model')
    parser.add_argument('platform', help='Target platform (e.g., rk3588, rk3568, rv1126)')
    
    # 常用参数
    parser.add_argument('-o', '--output', help='Path to output .rknn file (Default: same name as input)')
    parser.add_argument('-d', '--dataset', default='./datasets/dataset.txt', help='Path to quantization dataset (Default: ./dataset.txt)')
    parser.add_argument('--dtype', default='i8', choices=['i8', 'u8', 'fp'], help='Data type (i8/u8 for quant, fp for no quant)')
    
    # 高级参数：预设 vs 手动
    parser.add_argument('--preset', default='common', choices=['common', 'ocr', 'none'], 
                        help='Use preset mean/std values. "common"=0/255(YOLO), "ocr"=ImageNet norm')
    
    # 如果选了 preset='none'，则必须手动指定下面两项
    parser.add_argument('--mean', type=float, nargs='+', help='Manual mean values (e.g., --mean 0 0 0)')
    parser.add_argument('--std',  type=float, nargs='+', help='Manual std values (e.g., --std 255 255 255)')

    return parser.parse_args()

def main():
    args = get_args()

    # 1. 检查文件是否存在
    if not os.path.exists(args.input_model):
        print(f"Error: Input model '{args.input_model}' not found!")
        sys.exit(1)
    
    # 2. 确定量化参数
    do_quant = False if args.dtype == 'fp' else True
    if do_quant and not os.path.exists(args.dataset):
         print(f"Warning: Quantization is on ({args.dtype}) but dataset '{args.dataset}' not found.")
         print("Please prepare a dataset.txt file for calibration.")
         sys.exit(1)

    # 3. 确定 Mean/Std
    if args.preset != 'none':
        mean_values = [MODEL_PRESETS[args.preset]['mean']]
        std_values  = [MODEL_PRESETS[args.preset]['std']]
        print(f"--> Using preset '{args.preset}': mean={mean_values}, std={std_values}")
    else:
        if args.mean is None or args.std is None:
            print("Error: When --preset is 'none', you must provide --mean and --std.")
            sys.exit(1)
        mean_values = [args.mean]
        std_values = [args.std]

    # 4. 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.splitext(args.input_model)[0] + '.rknn'

    # ====================================================================
    # RKNN 核心流程
    # ====================================================================
    rknn = RKNN(verbose=False)

    print(f'--> Config model for {args.platform}')
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform=args.platform)

    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=args.input_model)
    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)

    print(f'--> Building model (Quantization: {do_quant}, Type: {args.dtype})')
    # Build 阶段
    ret = rknn.build(do_quantization=do_quant, dataset=args.dataset if do_quant else None)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)

    print(f'--> Exporting RKNN model to {output_path}')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)

    rknn.release()
    print('--> Done!')

if __name__ == '__main__':
    main()