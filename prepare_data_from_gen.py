# prepare_data_from_gen_txt.py
import os, shutil, random, argparse
from glob import glob

# ---------------- CONFIG ---------------- #
CLASS_FOLDERS = {
    "vehicles": "vehicle",
    "persons": "pedestrian",
    "rocks": "rockfall",
    "motos": "vehicle",
    "animals": "pedestrian"
}
FINAL_CLASSES = ["vehicle", "pedestrian", "rockfall"]

MAX_IMAGES = 2500
SPLIT_RATIO = [0.8, 0.1, 0.1]
# ----------------------------------------- #

def create_yolo_dataset(gen_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    lbl_all = os.path.join(out_root, "labels", "all")
    img_all = os.path.join(out_root, "images", "all")
    os.makedirs(lbl_all, exist_ok=True)
    os.makedirs(img_all, exist_ok=True)

    total = 0
    for folder, mapped_class in CLASS_FOLDERS.items():
        anno_path = os.path.join(gen_root, folder, "anno")
        img_path = os.path.join(gen_root, folder, "imgs")
        if not os.path.exists(anno_path):
            continue
        cls_id = FINAL_CLASSES.index(mapped_class)
        print(f"[+] Processing {folder} → {mapped_class} ({cls_id})")

        for label_file in glob(os.path.join(anno_path, "*.txt")):
            img_name = os.path.basename(label_file).replace(".txt", ".jpg")
            src_img = os.path.join(img_path, img_name)
            if not os.path.exists(src_img):
                # sometimes .png instead of .jpg
                img_name = img_name.replace(".jpg", ".png")
                src_img = os.path.join(img_path, img_name)
                if not os.path.exists(src_img):
                    continue

            # rewrite label file with new class ID
            new_lines = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    parts[0] = str(cls_id)
                    new_lines.append(" ".join(parts) + "\n")

            if not new_lines:
                continue

            dst_label = os.path.join(lbl_all, os.path.basename(label_file))
            dst_img = os.path.join(img_all, os.path.basename(src_img))
            with open(dst_label, "w") as f:
                f.writelines(new_lines)
            shutil.copy(src_img, dst_img)
            total += 1

    print(f"[✓] {total} labeled images converted to unified YOLO dataset.")

def split_dataset(base_out):
    all_imgs = glob(os.path.join(base_out, "images", "all", "*.jpg"))
    if len(all_imgs) == 0:
        all_imgs = glob(os.path.join(base_out, "images", "all", "*.png"))
    random.shuffle(all_imgs)
    keep = all_imgs[:MAX_IMAGES]
    n = len(keep)
    if n < 3:
        raise RuntimeError("Not enough images to split.")

    n_train = max(1, int(SPLIT_RATIO[0] * n))
    n_val = max(1, int(SPLIT_RATIO[1] * n))
    n_test = max(1, n - n_train - n_val)
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
            lbl_name = name.replace(".jpg", ".txt").replace(".png", ".txt")
            lbl_src = os.path.join(base_out, "labels", "all", lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(lbl_out, lbl_name))
    print("[✓] Split complete: train/val/test folders ready.")

def main(unzipped):
    gen_root = os.path.join(unzipped, "gen")
    base_out = "dataset/detection"
    create_yolo_dataset(gen_root, base_out)
    split_dataset(base_out)
    print("[✓] Final dataset ready in dataset/detection/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unzipped", required=True, help="Path to unzipped dataset containing 'gen/'")
    args = parser.parse_args()
    main(args.unzipped)
