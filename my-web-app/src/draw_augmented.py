import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_bbox(image_path, label_path, save_dir="augmented_vis"):
    img = Image.open(image_path)
    w, h = img.size
    fig, ax = plt.subplots()
    ax.imshow(img)

    with open(label_path, 'r') as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            xc *= w
            yc *= h
            bw *= w
            bh *= h
            rect = patches.Rectangle(
                (xc - bw/2, yc - bh/2), bw, bh, linewidth=2,
                edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xc - bw/2, yc - bh/2, f'{int(cls)}', color='white', backgroundcolor='red')

    plt.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 示例使用（你可以在命令行执行这个文件时启用它）
if __name__ == "__main__":
    draw_bbox(
        image_path="/var/www/html/iot/yolo/2025tokyo/yolo/job_1/image/augmented/dd442207-8475-4c29-8f31-9457af256a17_rotate1.jpg",
        label_path="/var/www/html/iot/yolo/2025tokyo/yolo/job_1/label/augmented/dd442207-8475-4c29-8f31-9457af256a17_rotate1.txt",
        save_dir="augmented_vis"
    )
