from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torchvision import transforms

from ptc_segmentor_ptc import PTCSegmentation


# Input and output settings
IMAGE_PATH = Path("images/2010_005021.jpg")
OUTPUT_PATH = Path("images/2010_005021_seg_pred.png")
CLASS_FILE = Path("configs/my_name.txt")

CLASS_NAMES = ["cabinet", "cat", "chair"]
HEX_COLORS = ["#6FA5F1", "#E9A6C9", "#F4B860"]
ALPHA = 0.75

# Normalized label centers: (x, y)
MANUAL_LABEL_POSITIONS = {
    "cabinet": (0.10, 0.30),
    "cat": (0.42, 0.62),
    "chair": (0.80, 0.20),
}


def load_font(size):
    """Load a bold font or fall back to the PIL default font."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def get_text_layout(draw, text, font, center, image_size):
    """Center text at a point while keeping it inside the image."""
    center_x, center_y = center
    width, height = image_size
    raw_box = draw.textbbox((0, 0), text, font=font)

    x = int(round(center_x - (raw_box[0] + raw_box[2]) / 2))
    y = int(round(center_y - (raw_box[1] + raw_box[3]) / 2))
    box = draw.textbbox((x, y), text, font=font)

    x += max(4 - box[0], 0) - max(box[2] - (width - 4), 0)
    y += max(4 - box[1], 0) - max(box[3] - (height - 4), 0)

    return (x, y), draw.textbbox((x, y), text, font=font)


def box_overlap_area(box_a, box_b, margin=8):
    """Compute the overlap area between two expanded text boxes."""
    left = max(box_a[0] - margin, box_b[0] - margin)
    top = max(box_a[1] - margin, box_b[1] - margin)
    right = min(box_a[2] + margin, box_b[2] + margin)
    bottom = min(box_a[3] + margin, box_b[3] + margin)

    return max(0, right - left) * max(0, bottom - top)


def add_class_labels(overlay, seg_pred, class_names, manual_positions=None):
    """Add class labels using manual positions or automatic placement."""
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)
    font = load_font(max(18, result.width // 18))
    manual_positions = manual_positions or {}
    occupied_boxes = []

    for class_id in np.unique(seg_pred):
        class_id = int(class_id)
        if not 0 <= class_id < len(class_names):
            continue

        class_name = class_names[class_id]

        if class_name in manual_positions:
            ratio_x, ratio_y = manual_positions[class_name]
            if not (0 <= ratio_x <= 1 and 0 <= ratio_y <= 1):
                raise ValueError(
                    f"The manual position of '{class_name}' must be within [0, 1]."
                )

            label_center = (
                ratio_x * result.width,
                ratio_y * result.height,
            )
            position, text_box = get_text_layout(draw, class_name, font, label_center, result.size,
            )
        else:
            binary_mask = (seg_pred == class_id).astype(np.uint8)
            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            )

            if num_labels <= 1:
                continue

            largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_region = labels == largest_id

            center_x, center_y = centroids[largest_id]
            region_width = stats[largest_id, cv2.CC_STAT_WIDTH]
            region_height = stats[largest_id, cv2.CC_STAT_HEIGHT]

            shift_x = max(12, min(40, int(region_width * 0.08)))
            shift_y = max(10, min(30, int(region_height * 0.08)))

            offsets = [(0, 0),(0, -shift_y),(0, shift_y),(-shift_x, 0),(shift_x, 0),
                (-shift_x, -shift_y),(shift_x, -shift_y),(-shift_x, shift_y),(shift_x, shift_y)]

            y_coords, x_coords = np.where(largest_region)
            best_layout = None
            best_score = None

            for rank, (offset_x, offset_y) in enumerate(offsets):
                target_x = center_x + offset_x
                target_y = center_y + offset_y

                nearest = np.argmin((x_coords - target_x) ** 2 + (y_coords - target_y) ** 2)
                label_center = (x_coords[nearest], y_coords[nearest])
                position, text_box = get_text_layout(draw, class_name, font, label_center, result.size)

                overlap = sum(
                    box_overlap_area(text_box, occupied_box)
                    for occupied_box in occupied_boxes
                )
                score = (overlap, rank)

                if best_score is None or score < best_score:
                    best_score = score
                    best_layout = (position, text_box)

                if overlap == 0:
                    break

            position, text_box = best_layout

        occupied_boxes.append(text_box)
        draw.text(position, class_name, font=font, fill=(0, 0, 0))

    return result


def main():
    if len(HEX_COLORS) < len(CLASS_NAMES):
        raise ValueError(
            "HEX_COLORS must contain at least one color for each class."
        )

    rgb_colors = np.asarray(
        [ImageColor.getrgb(color) for color in HEX_COLORS],
        dtype=np.uint8,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")

    CLASS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLASS_FILE.write_text(
        "\n".join(CLASS_NAMES),
        encoding="utf-8",
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0).cuda()

    model = ProxyCLIPSegmentation(
        clip_type="openai",
        model_type="ViT-B/16",
        vfm_model="dino",
        name_path=str(CLASS_FILE),
        ptc_enable=True,
        ptc_mu=0.3,
        ptc_min_seeds=64,
        ptc_border=1,
    )

    with torch.inference_mode():
        seg_pred = model.predict(image_tensor, data_samples=None)

    seg_pred = (seg_pred.detach().cpu().numpy().squeeze().astype(np.int32))

    if seg_pred.ndim != 2:
        raise ValueError(
            f"Expected a 2D segmentation map, but got {seg_pred.shape}."
        )

    if seg_pred.min() < 0 or seg_pred.max() >= len(CLASS_NAMES):
        raise ValueError(
            "The segmentation map contains an invalid class index."
        )

    height, width = seg_pred.shape
    image_np = np.asarray(
        image.resize((width, height), Image.Resampling.BILINEAR)
    )
    seg_color = rgb_colors[seg_pred]

    overlay = np.clip((1.0 - ALPHA) * image_np + ALPHA * seg_color,0,255).astype(np.uint8)

    result = add_class_labels(overlay, seg_pred, CLASS_NAMES, manual_positions=MANUAL_LABEL_POSITIONS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_PATH)

    active_classes = [CLASS_NAMES[class_id] for class_id in np.unique(seg_pred)]
    print(f"Predicted classes: {', '.join(active_classes)}")
    print(f"Result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()