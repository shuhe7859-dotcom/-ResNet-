"""
训练脚本 —— 自动扫描数据集、8:2 随机切分、训练并保存最优模型
附带 VisualDL 可视化 + 训练日志文件记录
"""

import os
import random
import traceback

import numpy as np
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader
from PIL import Image
from visualdl import LogWriter

import config
from resnet18 import create_model


# ==================== 固定随机种子 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# ==================== 自定义数据集 ====================
class GarbageDataset(Dataset):
    """加载图片路径列表和对应标签，带异常处理"""

    def __init__(self, image_paths, labels, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, np.array(self.labels[idx], dtype="int64")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform is not None:
                img = self.transform(img)
            return img, np.array(self.labels[idx], dtype="int64")

    def __len__(self):
        return len(self.image_paths)


# ==================== 数据准备 ====================
def build_transforms():
    """构建训练/验证数据增强 pipeline"""
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def scan_dataset(data_root):
    """扫描数据集目录"""
    class_names = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")
    ])
    print(f"发现 {len(class_names)} 个类别: {class_names}", flush=True)

    all_paths = []
    all_labels = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_root, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label_idx)

    print(f"共扫描到 {len(all_paths)} 张图片", flush=True)
    return class_names, all_paths, all_labels


def split_dataset(paths, labels):
    """随机 8:2 划分训练集和验证集"""
    indices = list(range(len(paths)))
    random.shuffle(indices)
    split_point = int(len(indices) * config.TRAIN_SPLIT)
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"训练集: {len(train_paths)} 张 | 验证集: {len(val_paths)} 张", flush=True)
    return train_paths, train_labels, val_paths, val_labels


# ==================== 训练主流程 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        try:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += float(loss) * images.shape[0]
            pred = paddle.argmax(logits, axis=1)
            correct += int((pred == labels).sum())
            total += images.shape[0]

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"  Epoch {epoch}/{total_epochs} "
                      f"[{batch_idx + 1}/{batch_count}] "
                      f"Loss: {float(loss):.4f}", flush=True)
        except Exception as e:
            print(f"  [警告] Batch {batch_idx} 出错: {e}", flush=True)
            continue

    return total_loss / max(total, 1), correct / max(total, 1)


@paddle.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        try:
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss) * images.shape[0]
            pred = paddle.argmax(logits, axis=1)
            correct += int((pred == labels).sum())
            total += images.shape[0]
        except Exception as e:
            print(f"  [警告] 验证批次出错: {e}", flush=True)
            continue

    return total_loss / max(total, 1), correct / max(total, 1)


def train():
    set_seed(config.RANDOM_SEED)

    # ---- 1. 扫描数据集 ----
    class_names, all_paths, all_labels = scan_dataset(config.DATA_ROOT)
    num_classes = len(class_names)

    # ---- 2. 划分训练/验证集 ----
    train_paths, train_labels, val_paths, val_labels = split_dataset(all_paths, all_labels)

    # ---- 3. 构建 DataLoader ----
    train_transform, val_transform = build_transforms()
    train_dataset = GarbageDataset(train_paths, train_labels, train_transform)
    val_dataset = GarbageDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # ---- 4. 创建模型 ----
    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=config.LEARNING_RATE,
    )

    # ---- 5. 初始化 VisualDL 和日志文件 ----
    writer = LogWriter(logdir=config.VDL_LOG_DIR)
    log_file = open(config.TRAIN_LOG_PATH, "w", encoding="utf-8")

    header = f"{'='*60}\n训练日志 - 基于 ResNet18 + 大模型融合的垃圾分类系统\n{'='*60}\n"
    header += f"类别数: {num_classes} | Batch: {config.BATCH_SIZE} | Epochs: {config.EPOCHS}\n"
    header += f"训练集: {len(train_paths)} 张 | 验证集: {len(val_paths)} 张\n"
    header += f"{'='*60}\n\n"
    header += f"{'Epoch':>6s}  {'Train Loss':>11s}  {'Train Acc':>10s}  {'Val Loss':>9s}  {'Val Acc':>8s}  {'Note':>6s}\n"
    header += f"{'-'*6}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*6}\n"

    print(header, flush=True)
    log_file.write(header)
    log_file.flush()

    # ---- 6. 训练循环 ----
    best_val_acc = 0.0
    best_epoch = 0

    print(f"{'='*60}", flush=True)
    print(f"开始训练 | 类别数={num_classes} | Batch={config.BATCH_SIZE} | Epochs={config.EPOCHS}", flush=True)
    print(f"VisualDL 日志: {config.VDL_LOG_DIR}", flush=True)
    print(f"训练日志文件: {config.TRAIN_LOG_PATH}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(1, config.EPOCHS + 1):
        try:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch, config.EPOCHS)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # 写入 VisualDL（每轮记录 4 个标量）
            writer.add_scalar(tag="train_loss", value=train_loss, step=epoch)
            writer.add_scalar(tag="train_acc", value=train_acc, step=epoch)
            writer.add_scalar(tag="val_loss", value=val_loss, step=epoch)
            writer.add_scalar(tag="val_acc", value=val_acc, step=epoch)

            # 判断是否最佳
            is_best = val_acc > best_val_acc
            note = "[BEST]" if is_best else ""
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch
                paddle.save(model.state_dict(), config.MODEL_SAVE_PATH)

            # 终端输出
            status = (f"Epoch {epoch:3d}/{config.EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            if note:
                status += f"  {note}"
            print(status, flush=True)

            # 写入日志文件
            log_line = (f"{epoch:6d}  {train_loss:11.4f}  {train_acc:10.4f}  "
                        f"{val_loss:9.4f}  {val_acc:8.4f}  {note}\n")
            log_file.write(log_line)
            log_file.flush()

        except Exception as e:
            print(f"Epoch {epoch} 崩溃: {e}", flush=True)
            traceback.print_exc()
            break

    # ---- 7. 训练结束收尾 ----
    log_file.close()
    writer.close()

    print(f"\n{'='*60}", flush=True)
    print(f"训练完成!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"最佳验证准确率: {best_val_acc:.4f}  (Epoch {best_epoch})", flush=True)
    print(f"模型文件: {config.MODEL_SAVE_PATH}", flush=True)
    print(f"日志文件: {config.TRAIN_LOG_PATH}", flush=True)
    print(f"VisualDL 日志: {config.VDL_LOG_DIR}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"启动 VisualDL 可视化面板:", flush=True)
    print(f"  visualdl --logdir={config.VDL_LOG_DIR} --port=8080", flush=True)
    print(f"然后打开浏览器访问 http://localhost:8080 查看训练曲线", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    train()
