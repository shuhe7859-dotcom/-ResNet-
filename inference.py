"""
推理脚本 —— 加载训练好的模型，对单张图片输出 Top1 / Top3 分类结果
"""

import os
import paddle
import paddle.nn.functional as F
from PIL import Image

import config
from resnet18 import create_model


def get_class_names():
    """从数据集目录动态获取类别名列表"""
    class_names = sorted([
        d for d in os.listdir(config.DATA_ROOT)
        if os.path.isdir(os.path.join(config.DATA_ROOT, d)) and not d.startswith(".")
    ])
    return class_names


def build_inference_transform():
    """构建推理时的图像预处理 pipeline"""
    import paddle.vision.transforms as T

    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class InferencePipeline:
    """封装模型加载 + 推理逻辑，供内部调用和外部对接"""

    def __init__(self, model_path=None, class_names=None):
        """
        Args:
            model_path: 模型权重 .pdparams 路径，默认使用 config 中的路径
            class_names: 类别名列表，不传则自动扫描数据集目录
        """
        self.class_names = class_names or get_class_names()
        self.num_classes = len(self.class_names)

        self.model = create_model(self.num_classes)
        weight_path = model_path or config.MODEL_SAVE_PATH
        state_dict = paddle.load(weight_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()

        self.transform = build_inference_transform()
        print(f"模型加载完成，类别数={self.num_classes}, 类别={self.class_names}")

    def predict(self, image_path):
        """
        对单张图片进行推理

        Args:
            image_path: 图片文件路径

        Returns:
            dict: {
                "top1": {"class": str, "confidence": float},
                "top3": [{"class": str, "confidence": float}, ...],
                "all_probs": list of float
            }
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)  # (1, 3, 224, 224)

        with paddle.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, axis=1).squeeze(0).numpy()

        # 按概率降序排列的索引
        sorted_idx = probs.argsort()[::-1]

        top1 = {
            "class": self.class_names[sorted_idx[0]],
            "confidence": float(probs[sorted_idx[0]]),
        }
        top3 = [
            {"class": self.class_names[i], "confidence": float(probs[i])}
            for i in sorted_idx[:3]
        ]

        return {
            "top1": top1,
            "top3": top3,
            "all_probs": [float(p) for p in probs],
        }

    def predict_batch(self, image_paths):
        """批量推理"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results


# ==================== 命令行直接调用 ====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python inference.py <图片路径>")
        sys.exit(1)

    pipeline = InferencePipeline()
    result = pipeline.predict(sys.argv[1])

    print(f"\n{'='*50}")
    print(f"图片: {sys.argv[1]}")
    print(f"Top1 预测: {result['top1']['class']} (置信度: {result['top1']['confidence']:.4f})")
    print(f"Top3 预测:")
    for i, item in enumerate(result["top3"], 1):
        print(f"  {i}. {item['class']:12s}  置信度: {item['confidence']:.4f}")
    print(f"{'='*50}")
