"""
主入口 —— 整合 训练 / 推理 / 大模型融合 全流程
用法:
  python main.py --train                         # 仅训练
  python main.py --infer <图片路径>               # 仅 ResNet 推理
  python main.py --full <图片路径>                # 推理 + 大模型融合
  python main.py --train --full <图片路径>        # 先训练再完整推理
"""

import os
import argparse

import config
from inference import InferencePipeline, get_class_names
from llm_fusion import llm_refine


def check_model_exists():
    """检查训练好的模型权重是否存在"""
    return os.path.exists(config.MODEL_SAVE_PATH)


def run_full_pipeline(image_path):
    """运行完整流程：ResNet 推理 → 大模型精判 → 输出结果"""
    if not check_model_exists():
        print(f"[错误] 未找到模型权重: {config.MODEL_SAVE_PATH}")
        print("请先运行训练: python main.py --train")
        return

    class_names = get_class_names()
    pipeline = InferencePipeline(class_names=class_names)
    result = pipeline.predict(image_path)

    # ResNet 粗分类结果
    print(f"\n{'='*60}")
    print(f"[图片]: {image_path}")
    print(f"{'='*60}")
    print(f"\n[ResNet 粗分类结果]")
    print(f"  Top1: {result['top1']['class']} (置信度: {result['top1']['confidence']:.4f})")
    print(f"  Top3:")
    for i, item in enumerate(result["top3"], 1):
        print(f"    {i}. {item['class']:12s}  置信度: {item['confidence']:.4f}")

    # 大模型融合
    print(f"\n[正在调用大模型进行语义精判...]")
    llm_result = llm_refine(result["top3"], class_names)

    print(f"\n{'='*60}")
    print(f"[最终分类结果]")
    print(f"{'='*60}")
    print(f"  最终类别: {llm_result['final_class']}")
    print(f"  投放指南: {llm_result['disposal_guide']}")
    print(f"  环保科普: {llm_result['eco_tip']}")
    print(f"{'='*60}\n")

    return result, llm_result


def main():
    parser = argparse.ArgumentParser(description="基于 ResNet18 + 大模型融合的垃圾分类系统")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--infer", type=str, default=None, metavar="PATH", help="单张图片推理")
    parser.add_argument("--full", type=str, default=None, metavar="PATH", help="推理 + 大模型融合全流程")
    args = parser.parse_args()

    # 如果没有任何参数，打印帮助
    if not any([args.train, args.infer, args.full]):
        parser.print_help()
        print("\n示例:")
        print("  python main.py --train")
        print("  python main.py --infer test.jpg")
        print("  python main.py --full test.jpg")
        print("  python main.py --train --full test.jpg")
        return

    # 训练
    if args.train:
        from train import train
        train()

    # 仅推理（不调用大模型）
    if args.infer:
        if not check_model_exists():
            print(f"[错误] 未找到模型权重: {config.MODEL_SAVE_PATH}")
            print("请先运行训练: python main.py --train")
            return
        pipeline = InferencePipeline()
        result = pipeline.predict(args.infer)
        print(f"\n{'='*50}")
        print(f"图片: {args.infer}")
        print(f"Top1: {result['top1']['class']} ({result['top1']['confidence']:.4f})")
        for i, item in enumerate(result["top3"], 1):
            print(f"Top{i}: {item['class']} ({item['confidence']:.4f})")
        print(f"{'='*50}")

    # 完整流程（推理 + 大模型）
    if args.full:
        run_full_pipeline(args.full)


if __name__ == "__main__":
    main()
