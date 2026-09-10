"""
Flask Web 展示系统 —— 垃圾分类 ResNet18 + 大模型融合
"""

import os
import sys

# ---- 必须在 import paddle 之前设置，防止 Windows 上 bf16 AMP 分配内存崩溃 ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import uuid
import time
import argparse
import webbrowser
import threading

from flask import Flask, render_template, request, jsonify, url_for

import config
from inference import InferencePipeline, get_class_names
from llm_fusion import llm_refine


# ==================== 资源路径（兼容 exe 打包） ====================
def resource_path(relative_path):
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# ==================== 初始化 ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=resource_path("templates"),
            static_folder=resource_path("static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}

# ---- 类别名 ----
if os.path.isdir(config.DATA_ROOT):
    class_names = get_class_names()
else:
    class_names = ["cardboard", "glass", "metal", "paper", "plastic"]

# ---- 模型加载 ----
model_path = config.MODEL_SAVE_PATH
if getattr(sys, "frozen", False):
    bundled = resource_path("best_model.pdparams")
    if os.path.exists(bundled):
        model_path = bundled

pipeline = InferencePipeline(model_path=model_path, class_names=class_names)
print(f"[Flask] 模型已加载，类别={class_names}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ==================== 路由 ====================
@app.route("/")
def index():
    return render_template("index.html", class_names=class_names)




@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "未上传图片"}), 400

        file = request.files["image"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "仅支持 jpg/png/bmp/webp"}), 400

        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        use_llm = request.form.get("use_llm", "0") == "1"

        t0 = time.time()
        result = pipeline.predict(filepath)
        resnet_time = round((time.time() - t0) * 1000, 1)

        response = {
            "image_url": url_for("static", filename=f"uploads/{filename}"),
            "resnet": {
                "top1": result["top1"],
                "top3": result["top3"],
                "time_ms": resnet_time,
            },
            "llm": None,
        }

        if use_llm:
            t1 = time.time()
            llm_result = llm_refine(result["top3"], class_names)
            llm_time = round((time.time() - t1) * 1000, 1)
            response["llm"] = {
                "final_class": llm_result["final_class"],
                "disposal_guide": llm_result["disposal_guide"],
                "eco_tip": llm_result["eco_tip"],
                "time_ms": llm_time,
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500


# ==================== 启动 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="垃圾分类 Flask Web 展示系统")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"\n{'='*60}")
    print(f"垃圾分类 ResNet18 + 大模型融合 Web 系统")
    print(f"{'='*60}")
    print(f"访问地址: {url}")
    print(f"{'='*60}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)
