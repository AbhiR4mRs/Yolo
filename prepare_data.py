# prepare_dataset_from_gen.py
import os, json, shutil, random, argparse
from glob import glob
from PIL import Image

# ---------------- CONFIG ---------------- #
CLASS_FOLDERS = {
    "vehicles": "vehicle",
    "persons": "pedestrian",
    "rocks": "rockfall",
    "motos": "vehicle",     # treat motorcycles as vehicles
    "animals": "pedestrian" # optional (small moving object)
}
FINAL_CLASSES = ["vehicle", "pedestrian", "rockfall"]

MAX_IMAGES = 2500
SPLIT_RATIO = [0.8, 0.1, 0.1]
# ----------------------------------------- #

def convert_bbox_to_yolo(img_w, img_h, bbox):
    """bbox = [xmin, ymin, xmax, ymax] → YOLO normalized [x, y, w, h]"""
    xmin, ymin, xmax, ymax = bbox
    x = (xmin + xmax) / 2.0 / img_w
    y = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x, y, w, h

def create_yolo_labels(gen_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    lbl_dir = os.path.join(out_root, "labels", "all")
    img_dir = os.path.join(out_root, "images", "all")
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    label_count = 0
    for folder in CLASS_FOLDERS.keys():
        anno_path = os.path.join(gen_root, folder, "anno")
        img_path = os.path.join(gen_root, folder, "imgs")
        if not os.path.exists(anno_path):
            continue

        cls_name = CLASS_FOLDERS[folder]
        cls_id = FINAL_CLASSES.index(cls_name)
        print(f"[+] Processing {folder} → class {cls_name} ({cls_id})")

        for ann_file in glob(os.path.join(anno_path, "*.json")):
            try:
                with open(ann_file, "r") as f:
                    data = json.load(f)
                img_name = data.get("imagePath") or os.path.basename(ann_file).replace(".json", ".jpg")
                img_file = os.path.join(img_path, img_name)
                if not os.path.exists(img_file):
                    continue

                img = Image.open(img_file)
                w, h = img.size

                # assume annotation file has 'shapes' or 'bbox'
                shapes = data.get("shapes", [])
                lines = []
                for s in shapes:
                    if "points" not in s:
                        continue
                    pts = s["points"]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                    x, y, bw, bh = convert_bbox_to_yolo(w, h, [xmin, ymin, xmax, ymax])
                    lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

                if not lines:
                    continue

                # save YOLO label
                lbl_name = os.path.basename(ann_file).replace(".json", ".txt")
                with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                    f.writelines(lines)
                # copy image
                shutil.copy(img_file, os.path.join(img_dir, img_name))
                label_count += 1
            except Exception as e:
                print("Error in", ann_file, e)
    print(f"[✓] {label_count} labeled images converted to YOLO format.")

def split_dataset(base_out):
    all_imgs = glob(os.path.join(base_out, "images", "all", "*.jpg"))
    random.shuffle(all_imgs)
    keep = all_imgs[:MAX_IMAGES]
    n = len(keep)
    n_train = int(SPLIT_RATIO[0]*n)
    n_val = int(SPLIT_RATIO[1]*n)
    splits = {
        "train": keep[:n_train],
        "val": keep[n_train:n_train+n_val],
        "test": keep[n_train+n_val:]
    }

    for split, files in splits.items():
        img_out = os.path.join(base_out, "images", split)
        lbl_out = os.path.join(base_out, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for f in files:
            name = os.path.basename(f)
            shutil.copy(f, os.path.join(img_out, name))
            lbl_f = name.replace(".jpg", ".txt")
            lbl_src = os.path.join(base_out, "labels", "all", lbl_f)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(lbl_out, lbl_f))
    print("[✓] Dataset split into train/val/test under dataset/detection/")

def main(zip_root):
    gen_root = os.path.join(zip_root, "gen")
    base_out = "dataset/detection"
    os.makedirs(base_out, exist_ok=True)

    create_yolo_labels(gen_root, base_out)
    split_dataset(base_out)
    print("[✓] Final YOLO dataset ready at dataset/detection/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unzipped", required=True, help="Path to the unzipped SynRailObs folder (containing 'gen/')")
    args = parser.parse_args()
    main(args.unzipped)
