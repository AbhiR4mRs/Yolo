# detect_final.py
import argparse
from pathlib import Path
import torch
import cv2
import yaml
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, check_yaml
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

@torch.no_grad()
def run(weights, source, data, imgsz=640, conf_thres=0.25, iou_thres=0.45, device='', save_dir='runs/detect_final'):
    # Setup
    device = select_device(device)
    # Pass data to DetectMultiBackend to load names
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # The model's `names` attribute is now correctly loaded.
    # We can remove the manual YAML loading for class names.
    print(f"✅ Loaded classes: {names}")

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Warmup (✅ fixed)
    model.warmup(imgsz=(1, 3, *imgsz))

    # Inference loop
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        preds = model(im)
        # The output of `model(im)` can be a tuple. The first element is the detection output.
        # We need to handle this to ensure `non_max_suppression` gets the correct input.
        det_preds = preds[0] if isinstance(preds, tuple) else preds
        preds = non_max_suppression(det_preds, conf_thres, iou_thres)

        for det in preds:
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2)
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            result_img = annotator.result()
            save_path = save_dir / Path(path).name
            cv2.imwrite(str(save_path), result_img)
            print(f"✅ Saved: {save_path}")

    print("🎯 Detection completed successfully!")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='path to trained model')
    parser.add_argument('--source', type=str, required=True, help='folder or image for inference')
    parser.add_argument('--data', type=str, required=True, help='YAML file with class names')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='')
    return parser.parse_args()


def main(opt):
    # Handle imgsz argument
    imgsz = opt.imgsz
    if len(imgsz) == 2:  # h, w
        imgsz = tuple(imgsz)
    elif len(imgsz) == 1:  # square
        imgsz = (imgsz[0], imgsz[0])
    opt.imgsz = imgsz
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
