import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, check_requirements
from utils.plots import Annotator
from utils.torch_utils import select_device

# --- Risk Assessment (3 classes) ---
def compute_risk_score(cls_name, bbox, frame_width, frame_height):
    risk_weights = {
        'pedestrian': 0.9,
        'vehicle': 0.7,
        'rockfall': 0.6
    }

    type_score = risk_weights.get(cls_name, 0.5)
    x_center = (bbox[0] + bbox[2]) / 2
    dist = abs(x_center - frame_width / 2) / (frame_width / 2)
    distance_score = 1 - dist
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_ratio = area / (frame_width * frame_height)

    total_risk = 0.6 * type_score + 0.25 * distance_score + 0.15 * area_ratio
    return round(total_risk, 3)

@torch.no_grad()
def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45, device=''):
    # --- Setup ---
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((imgsz, imgsz), s=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    save_dir = Path('runs/detect_with_risk')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Model loaded with classes: {names}")

    # --- Inference Loop ---
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=2)
        frame_h, frame_w = im0.shape[:2]

        if len(pred[0]):
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(pred[0]):
                cls_name = names[int(cls)]
                bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                risk = compute_risk_score(cls_name, bbox, frame_w, frame_h)

                label = f"{cls_name} {conf:.2f} | Risk: {risk:.2f}"
                color = (0, 0, 255) if risk > 0.7 else (0, 255, 255) if risk > 0.4 else (0, 255, 0)
                annotator.box_label(xyxy, label, color=color)

                print(f"[{cls_name}] conf={conf:.2f}, risk={risk:.2f}")

        save_path = save_dir / Path(path).name
        cv2.imwrite(str(save_path), annotator.result())
        print(f"💾 Saved: {save_path}")

    print("✅ Detection with Risk Assessment completed.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='')
    return parser.parse_args()


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
