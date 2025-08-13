from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import io
import math
import numpy as np
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
from datetime import datetime, timedelta
import base64
from io import BytesIO
import re
import torch.multiprocessing as mp

import csv
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import shutil
import glob
import time
import threading
from multiprocessing import Process, Manager
FIXED_MODEL = "yolov8s.pt"  # 固定模型路径
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

base_path = "/var/www/html/iot/yolo/2025tokyo/yolo"
# 全局训练中断标记（job_id -> threading.Event）
train_cancel_flags = {}

train_processes = {}  # job_id -> Process


def generate_augmented_dataset(
    image_dir,
    label_dir,
    out_img_dir,
    out_label_dir,
    augmentation_settings
):
    # 清空旧目录
    for d in [out_img_dir, out_label_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        label_src = os.path.join(label_dir, f"{basename}.txt")
        if not os.path.exists(label_src):
            continue

        # === Step 1: 保存原图到 augmented 文件夹 ===
        orig_img_name = f"{basename}_original.jpg"
        orig_lbl_name = f"{basename}_original.txt"
        orig_img_path = os.path.join(out_img_dir, orig_img_name)
        orig_lbl_path = os.path.join(out_label_dir, orig_lbl_name)

        image.save(orig_img_path)
        shutil.copy(label_src, orig_lbl_path)

        # === Step 2: 各种增强处理 ===
        for key, setting in augmentation_settings.items():
            for i in range(setting["scale"]):
                if key == "rotate":
                    angle = random.uniform(-abs(setting["param"]), abs(setting["param"]))
                    transform = lambda img: TF.rotate(img, angle, expand=True)
                else:
                    transform = build_transform(setting)

                if transform is None:
                    continue

                aug_img = transform(image)
                aug_name = f"{basename}_{key}{i}.jpg"
                aug_img_path = os.path.join(out_img_dir, aug_name)
                aug_lbl_path = os.path.join(out_label_dir, f"{basename}_{key}{i}.txt")
                aug_img.save(aug_img_path)

                # === 标签处理 ===
                if key == "rotate":
                    orig_w, orig_h = image.size
                    new_w, new_h = aug_img.size
                    new_lines = []
                    with open(label_src, "r") as f:
                        for line in f:
                            try:
                                cls, xc, yc, w, h = line.strip().split()
                                xc, yc, w, h = map(float, [xc, yc, w, h])
                                new_xc, new_yc, new_w_box, new_h_box = rotate_bbox_geom(
                                    xc, yc, w, h, angle, orig_w, orig_h, new_w, new_h
                                )
                                new_line = f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w_box:.6f} {new_h_box:.6f}"
                                new_lines.append(new_line)
                            except ValueError:
                                print(f"[!] Invalid label line in {label_src}: {line.strip()}")
                    with open(aug_lbl_path, "w") as f:
                        f.write("\n".join(new_lines) + "\n")
                else:
                    shutil.copy(label_src, aug_lbl_path)

                print(f"[+] {aug_name} + label → OK")
        # === Step 3: 统计增强结果 ===
        split_train_val(out_img_dir, out_label_dir, train_ratio=0.8)


def split_train_val(image_aug_dir, label_aug_dir, train_ratio=0.8):
    # 获取所有增强后的图像文件名（只取 .jpg）
    all_images = [f for f in os.listdir(image_aug_dir) if f.endswith(".jpg")]
    all_images.sort()
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # 生成新目录
    image_train_dir = os.path.join(os.path.dirname(image_aug_dir), "train")
    image_val_dir = os.path.join(os.path.dirname(image_aug_dir), "val")
    label_train_dir = os.path.join(os.path.dirname(label_aug_dir), "train")
    label_val_dir = os.path.join(os.path.dirname(label_aug_dir), "val")

    for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def move_files(image_list, img_dst, lbl_dst):
        for img_file in image_list:
            name = os.path.splitext(img_file)[0]
            src_img = os.path.join(image_aug_dir, img_file)
            src_lbl = os.path.join(label_aug_dir, name + ".txt")

            dst_img = os.path.join(img_dst, img_file)
            dst_lbl = os.path.join(lbl_dst, name + ".txt")

            shutil.copy(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"[!] Warning: Label not found for {img_file}")

    # 执行复制
    move_files(train_images, image_train_dir, label_train_dir)
    move_files(val_images, image_val_dir, label_val_dir)
def count_augmented_images(out_img_dir):
    counts = defaultdict(int)
    for fname in os.listdir(out_img_dir):
        if not fname.endswith(".jpg"):
            continue
        # 解析出增强类型
        parts = fname.split("_")
        if len(parts) >= 2:
            key = ''.join([c for c in parts[-1] if not c.isdigit()]).replace(".jpg", "")
            counts[key] += 1
        else:
            counts["original"] += 1
    return counts

def build_transform(setting):
    t, p = setting["type"], setting["param"]
    if t == "blur":
        return T.GaussianBlur(kernel_size=p * 2 + 1)
    elif t == "rotate":
            return None
    elif t == "brightness":
        return T.ColorJitter(brightness=p / 10)
    elif t == "contrast":
        return T.ColorJitter(contrast=p / 10)
    elif t == "noise":
        def add_noise(img):
            np_img = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C] or [H, W]
            if np_img.ndim == 2:
                np_img = np_img.unsqueeze(0)
            elif np_img.ndim == 3:
                np_img = np_img.permute(2, 0, 1)  # -> [C, H, W]

            noise = torch.randn_like(np_img) * (p / 100.0)
            noisy_img = torch.clamp(np_img + noise, 0.0, 1.0)
            return T.ToPILImage()(noisy_img)

        return add_noise
    else:
        return None
def rotate_bbox_geom(xc, yc, w, h, angle_deg, orig_w, orig_h, new_w, new_h):
    """
    几何方法实现的 bbox 旋转，输入输出为归一化 [0,1] 的 xc, yc, w, h
    注意旋转中心应为 new_w, new_h 中心
    """
    angle_rad = math.radians(angle_deg)
    print(f"Applied rotation: {angle_deg}° → {angle_rad:.3f} rad")
    cx, cy = new_w / 2, new_h / 2  # ✅ 使用新图中心

    # 归一化坐标转为原图像素坐标
    xc *= orig_w
    yc *= orig_h
    w *= orig_w
    h *= orig_h

    # 4个角点
    corners = np.array([
        [xc - w / 2, yc - h / 2],
        [xc + w / 2, yc - h / 2],
        [xc + w / 2, yc + h / 2],
        [xc - w / 2, yc + h / 2],
    ])

    # 极坐标旋转 + 平移到新图中心
    rotated = []
    for x, y in corners:
        dx, dy = x - (orig_w / 2), y - (orig_h / 2)  # 相对原图中心
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) - angle_rad
        new_x = cx + r * math.cos(theta)
        new_y = cy + r * math.sin(theta)
        rotated.append([new_x, new_y])
    rotated = np.array(rotated)

    # 外接矩形
    x_min, y_min = rotated.min(axis=0)
    x_max, y_max = rotated.max(axis=0)

    new_xc = (x_min + x_max) / 2 / new_w
    new_yc = (y_min + y_max) / 2 / new_h
    new_w_box = (x_max - x_min) / new_w
    new_h_box = (y_max - y_min) / new_h

    return new_xc, new_yc, new_w_box, new_h_box





def get_job_paths(job_id):
    job_dir = os.path.join(base_path, job_id)
    image_dir = os.path.join(job_dir, "images")
    label_dir = os.path.join(job_dir, "labels")
    return {
        "job_dir": job_dir,
        "image_dir": image_dir,
        "label_dir": label_dir,
    }
def parse_time_from_folder(folder_name):
    match = re.search(r'exp_(\d{8})_(\d{6})', folder_name)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        dt_str = f"{date_part}{time_part}"
        try:
            return time.mktime(time.strptime(dt_str, "%Y%m%d%H%M%S"))
        except:
            return 0
    return 0

def clean_old_results(results_dir, max_keep=15):
    try:
        folders = [
            f for f in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, f))
        ]
        folders_with_time = [
            (f, parse_time_from_folder(f))
            for f in folders
        ]
        folders_sorted = sorted(folders_with_time, key=lambda x: x[1], reverse=True)
        to_keep = set(f[0] for f in folders_sorted[:max_keep])
        for folder_name, _ in folders_with_time[max_keep:]:
            folder_path = os.path.join(results_dir, folder_name)
            try:
                shutil.rmtree(folder_path)
                print(f"[Cleaner] Deleted: {folder_path}")
            except Exception as e:
                print(f"[Cleaner] Error deleting {folder_path}: {e}")
    except Exception as e:
        print(f"[Cleaner] Error in cleaning {results_dir}: {e}")

def main_cleaner(base_dir):
    while True:
        try:
            # 遍历所有job_x文件夹
            for job_folder in os.listdir(base_dir):
                job_path = os.path.join(base_dir, job_folder)
                if os.path.isdir(job_path):
                    results_dir = os.path.join(job_path, 'results')
                    if os.path.exists(results_dir):
                        print(f"[Cleaner] Cleaning {results_dir}")
                        clean_old_results(results_dir, max_keep=15)
        except Exception as e:
            print(f"[Cleaner] Error in main loop: {e}")
        time.sleep(600)  # 每10分钟执行一次

# 启动守护线程
base_directory = "/var/www/html/iot/yolo/2025tokyo/yolo"
clean_thread = threading.Thread(
    target=main_cleaner,
    args=(base_directory,),
    daemon=True
)
clean_thread.start()
def clean_old_images(directory, max_age_seconds=3600):
    while True:
        now = time.time()
        print(f"[Cleaner] Scanning {directory} ...")
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.isfile(file_path):
                        file_age = now - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            print(f"[Cleaner] Deleted: {file_path}")
                except Exception as e:
                    print(f"[Cleaner] Error deleting {file_path}: {e}")
        time.sleep(600)  # 每 10 分钟执行一次
clean_thread = threading.Thread(
    target=clean_old_images,
    args=('static/detect_results', 3600),  # 1 小时内保留
    daemon=True  # 守护线程，不会阻止 Flask 退出
)
clean_thread.start()

# ---- 固定模型 yolov8s.pt ----
print(f"[Startup] Loading fixed model: {FIXED_MODEL}")
fixed_model = YOLO(FIXED_MODEL)
fixed_model.eval()
print("[Startup] Fixed model loaded.")

# ---- 动态模型：最多支持 5 个 slot ----
NUM_DYNAMIC_SLOTS = 5
dynamic_models = [None] * NUM_DYNAMIC_SLOTS
model_locks = [threading.Lock() for _ in range(NUM_DYNAMIC_SLOTS)]

# ---- 图像解码辅助函数 ----
def decode_base64_image(base64_str):
    try:
        # 去除前缀：data:image/jpeg;base64,
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Base64 decode error: {str(e)}")

def img_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def base64_to_pil(img_base64):
    img_data = base64.b64decode(img_base64.split(",")[1])
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

def run_inference(model, confidence, iou, image):
    # 保存临时图片
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"upload_{timestamp}.jpg"
    temp_dir = "static/temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    image.save(temp_path)

    # 执行推理
    project_dir = os.path.join("static", "detect_results")
    res = model.predict(
        source=temp_path,
        conf=confidence / 100,
        iou=iou / 100,
        save=True,
        project=project_dir,
        name=timestamp,
        exist_ok=True
    )
    saved_path = os.path.join(res[0].save_dir, filename)
    return saved_path

@app.route("/train_status/<job_id>", methods=["GET"])
def get_train_status(job_id):
    paths = get_job_paths(job_id)
    status_path = os.path.join(paths["job_dir"], "train_status.json")

    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return jsonify(json.load(f))
    else:
        return jsonify({"status": "idle"})
    

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


@app.route("/job_list", methods=["GET"])
def list_jobs():
    job_root = "/var/www/html/iot/yolo/2025tokyo/yolo"
    if not os.path.exists(job_root):
        return jsonify([])

    job_dirs = [
        d for d in os.listdir(job_root)
        if os.path.isdir(os.path.join(job_root, d)) and d.startswith("job_")
    ]

    job_dirs_sorted = sorted(job_dirs, key=natural_key)
    return jsonify(job_dirs_sorted)


def train_job(paths,counts,use_pretrained,model_type,epochs,save_dir,weight_decay,learning_rate,optimizer_name,batch_size,experiment_name):
    status_path = os.path.join(paths["job_dir"], "train_status.json")
    try:
        # 你的训练逻辑
        # 例如：加载数据、训练模型
        class_file = os.path.join(paths["label_dir"], "class.txt")
        if not os.path.exists(class_file):
            raise FileNotFoundError(f"class.txt not found in {paths['label_dir']}")
        with open(class_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

        dataset_yaml = os.path.join(paths["job_dir"], "dataset.yaml")
        with open(dataset_yaml, "w") as f:
            f.write(f"path: {paths['job_dir']}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")

        # 生成模型权重路径
        if use_pretrained:
            weight_file = f"{model_type}.pt"  # 预训练权重
        else:
            weight_file = f"{model_type}.yaml"  # 只加载模型结构

        model = YOLO(weight_file)

        # 训练参数字典
        train_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "imgsz": 640,
            "project": save_dir,
            "name": experiment_name,
            "exist_ok": True,
            "device": 0 if torch.cuda.is_available() else "cpu",
            "workers": 0,
            "verbose": True,
            "batch": batch_size,
            "optimizer": optimizer_name,
            "lr0": learning_rate,
            "weight_decay": weight_decay,
        }

        model.train(**train_params)


        # 训练成功后，更新状态
        with open(status_path, "w") as f:
            json.dump({"status": "done", "augmentation_counts": counts}, f)

    except Exception as e:
        print(f"[!] Train failed: {e}")
        with open(status_path, "w") as f:
            json.dump({"status": "failed", "error": str(e)}, f)
    finally:
        # 如果被强制杀死，确保状态为“已取消”
        if os.path.exists(status_path):
            # 你可以在这里检测状态文件是否已写入“done”或“failed”
            # 如果没有，写入“cancelled”
            with open(status_path, "r") as f:
                status_data = json.load(f)
            if status_data.get("status") not in ["done", "failed"]:
                with open(status_path, "w") as f:
                    json.dump({"status": "cancelled"}, f)

@app.route("/train", methods=["POST"])
def start_train():
    data = request.get_json()
    job_id = data.get("job_id")
    augmentation_settings = data.get("augmentations", {})
    epochs = data.get("epochs", 30)
    optimizer_name = data.get("optimizer", "AdamW")
    learning_rate = data.get("learning_rate", 0.001)
    batch_size = data.get("batch_size", 16)
    weight_decay = data.get("weight_decay", 0.0005)
    model_type = data.get("model_type", "yolov8n")
    use_pretrained = data.get("use_pretrained", True)
    start_time_str = data.get("start_time")
    dt = datetime.fromisoformat(start_time_str.rstrip("Z"))
    dt_plus_9 = dt + timedelta(hours=9)
    # 格式化为YYYYMMDD_HHMMSS
    timestamp_str = dt_plus_9.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp_str}"

    paths = get_job_paths(job_id)
    # 生成路径
    save_dir = os.path.join(paths["job_dir"], "results")

    image_dir = paths["image_dir"]
    label_dir = paths["label_dir"]
    out_img_dir = os.path.join(image_dir, "augmented")
    out_label_dir = os.path.join(label_dir, "augmented")

    # 生成增强数据
    generate_augmented_dataset(
        image_dir,
        label_dir,
        out_img_dir,
        out_label_dir,
        augmentation_settings
    )
    counts = count_augmented_images(out_img_dir)

    # 标记为 running
    status_path = os.path.join(paths["job_dir"], "train_status.json")
    with open(status_path, "w") as f:
        json.dump({
            "status": "running",
            "augmentation_counts": counts
        }, f)


    # ✅ 使用多进程启动训练
    p = Process(
    target=train_job,
    args=(
        paths,
        counts,
        use_pretrained,
        model_type,
        epochs,
        save_dir,
        weight_decay,
        learning_rate,
        optimizer_name,
        batch_size,
        experiment_name,
        )
    )
    p.start()

    train_processes[job_id] = p

    return jsonify({
        "message": f"Training started for job {job_id}",
        "augmentation_counts": counts
    })
@app.route("/cancel_train/<job_id>", methods=["POST"])
def cancel_train(job_id):
    p = train_processes.get(job_id)
    try:
        if p and p.is_alive():
            try:
                p.terminate()
                p.join(timeout=5)
            except Exception as e:
                print(f"Error terminating process {job_id}: {e}")
        else:
            return jsonify({"error": "No active training process found for this job."}), 404
    finally:
        # 无论上面是否异常，都尝试写入取消状态
        status_path = os.path.join(base_path, job_id, "train_status.json")
        try:
            with open(status_path, "w") as f:
                json.dump({
                    "status": "cancelled",
                    "augmentation_counts": {}
                }, f)
        except Exception as e:
            print(f"Error writing status for {job_id}: {e}")

        # 从字典中移除进程对象（如果存在）
        if job_id in train_processes:
            del train_processes[job_id]

    return jsonify({"message": f"Training for {job_id} has been cancelled."})

    

@app.route("/augment_preview", methods=["POST"])
def augment_preview():
    data = request.get_json()
    img_base64 = data.get("image")
    aug_type = data.get("type")
    param = float(data.get("param", 50))  # slider 原始范围为 0~100

    img = base64_to_pil(img_base64)

    if aug_type == "blur":
        sigma = max(0.1, param / 10)  # param: 0~100 → sigma: 0.1~10
        transform = T.GaussianBlur(kernel_size=5, sigma=sigma)
        img = transform(img)

    elif aug_type == "rotate":
        angle = param  # param 直接为角度
        img = TF.rotate(img, angle)

    elif aug_type == "brightness":
        factor = 0.5 + (param - 50) / 50 * 1.5
        img = TF.adjust_brightness(img, factor)

    elif aug_type == "contrast":
        factor = 0.5 + (param - 50) / 50 * 1.5
        img = TF.adjust_contrast(img, factor)

    elif aug_type == "noise":
        tensor = TF.to_tensor(img)
        noise = torch.randn_like(tensor) * (param / 100)  # std ∈ [0, 1]
        tensor = torch.clamp(tensor + noise, 0, 1)
        img = TF.to_pil_image(tensor)

    else:
        return jsonify({"error": "Unsupported augmentation type"}), 400

    return jsonify({"image": pil_to_base64(img)})


@app.route('/datasets')
def list_datasets():
    
    datasets = []

    for job in os.listdir(base_path):
        job_path = os.path.join(base_path, job)
        image_dir = os.path.join(job_path, "images")
        label_dir = os.path.join(job_path, "labels")
        if os.path.isdir(image_dir) and os.path.isdir(label_dir):
            image_count = len([f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")])
            datasets.append({
                "job": job,
                "image_count": image_count,
            })

    return jsonify(datasets)

@app.route('/results_csv', methods=['POST'])
def get_results_csv():
    data = request.get_json()
    # 传入路径参数（示例：路径在请求体中）
    relative_path = data.get('path')
    if not relative_path:
        return jsonify({"error": "Path is required"}), 400

    # 拼接完整路径
    full_path = os.path.join(base_path, relative_path)
    if not full_path or not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404

    results = []
    with open(full_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return jsonify(results)

@app.route('/datasets/<job_id>')
def dataset_detail(job_id):
    job_path = os.path.join(base_path, job_id)
    image_dir = os.path.join(job_path, "images")
    label_dir = os.path.join(job_path, "labels")
    results_dir = os.path.join(job_path, "results")  # 你存放exp的目录

    # 检查目录是否存在
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        return jsonify({"error": "Job not found"}), 404

    # 获取图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    # 获取标签文件（排除 class.txt）
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt") and f != "class.txt"]

    # 统计所有标签文件的行数总和
    total_label_lines = 0
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r", encoding="utf-8") as lf:
            lines = lf.readlines()
            total_label_lines += len(lines)

    # 随机抽取样本图片
    sample_images_path = random.sample(image_files, min(3, len(image_files)))

    sample_images = []
    for sample_image_path in sample_images_path:
        full_path = os.path.join(image_dir, sample_image_path)
        sample_images.append({
            "filename": sample_image_path,
            "base64": img_to_base64(full_path)
        })

    # 列出results目录下所有exp_开头的文件夹或文件
    exp_list = []
    if os.path.exists(results_dir):
        for entry in os.listdir(results_dir):
            if entry.startswith("exp_"):
                exp_list.append(entry)

    return jsonify({
        "job": job_id,
        "image_count": len(image_files),
        "label_count": total_label_lines,  # 改为标签行数总和
        "sample_images": sample_images,
        "experiments": exp_list  # 返回所有exp_开头的文件夹或文件名
    })


@app.route('/yolov8', methods=['POST'])
def detect_yolov8s(fixed_model=fixed_model):
    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field"}), 400
    if "confidence" not in data or "iou" not in data:
        return jsonify({"error": "Missing 'confidence' or 'iou' field"}), 400
    iou, confidence = data["iou"], data["confidence"]
    image = decode_base64_image(data["image_base64"])
    saved_path = run_inference(fixed_model, confidence, iou, image)
    img_base64 = img_to_base64(saved_path)
    return jsonify({
        'message': 'Detection success',
        'image_path': saved_path,
        'image_base64': img_base64
    })


NUM_DYNAMIC_SLOTS = 5
dynamic_models = [None] * NUM_DYNAMIC_SLOTS
model_locks = [threading.Lock() for _ in range(NUM_DYNAMIC_SLOTS)]

# 维护 job_id -> slot 映射，slot 从 0 到 4
job_slot_map = {}
slot_lock = threading.Lock()
current_slot = 0  # 0-based

def get_next_slot():
    global current_slot
    with slot_lock:
        slot = current_slot
        current_slot = (current_slot + 1) % NUM_DYNAMIC_SLOTS
    return slot

def load_model_for_job(job_id, exp):
    # 由前端传入最新的exp，构造权重路径
    weight_path = f"/var/www/html/iot/yolo/2025tokyo/yolo/{job_id}/results/{exp}/weights/best.pt"

    slot = get_next_slot()

    try:
        new_model = YOLO(weight_path)
        with model_locks[slot]:
            dynamic_models[slot] = new_model
        job_slot_map[job_id] = slot
        print(f"Loaded model for job {job_id} (exp: {exp}) into slot {slot}")
        return slot
    except Exception as e:
        print(f"Failed to load model for job {job_id} (exp: {exp}): {e}")
        return None




@app.route("/detect/<job_id>", methods=["POST"])
def detect_dynamic(job_id):
    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    exp = data.get("exp")
    if not exp:
        return jsonify({"error": "Missing 'exp' field"}), 400

    image = decode_base64_image(data["image_base64"])
    slot = job_slot_map.get(job_id)
    if slot is None:
        # 没有分配slot，加载模型
        slot = load_model_for_job(job_id, exp)
        if slot is None:
            return jsonify({"error": "Failed to load model"}), 500

    # 尝试非阻塞获取锁
    locked = model_locks[slot].acquire(blocking=False)
    if not locked:
        return jsonify({"error": "Slot busy, please retry later"}), 429

    try:
        model = dynamic_models[slot]
        if model is None:
            return jsonify({"error": f"No model loaded in slot {slot}"}), 400

        saved_path = run_inference(model, confidence=data.get("confidence", 50), iou=data.get("iou", 50), image=image)
        img_base64 = img_to_base64(saved_path)

        return jsonify({
            'message': f'Detection success using job {job_id} (exp: {exp}) in slot {slot}',
            'image_path': saved_path,
            'image_base64': img_base64
        })
    finally:
        model_locks[slot].release()




# ---- 加载模型到动态 slot ----
@app.route("/load_model/<int:slot>", methods=["POST"])
def load_model(slot):
    if not (0 <= slot < NUM_DYNAMIC_SLOTS):
        return jsonify({"error": "Invalid slot index"}), 400

    data = request.get_json()
    weight_path = data.get("weight")
    if not weight_path or not os.path.exists(weight_path):
        return jsonify({"error": f"Model file '{weight_path}' not found"}), 400

    try:
        new_model = YOLO(weight_path)
        with model_locks[slot]:
            dynamic_models[slot] = new_model
        return jsonify({"message": f"Model loaded into slot {slot}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    app.run(host='0.0.0.0', port=28210, debug=True)

