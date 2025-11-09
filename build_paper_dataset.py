import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
GEN_ROOT = "/content/gen"
OUT_ROOT = "/content/drive/MyDrive/yolo/paper_dataset_2534"

FINAL_CLASSES = ["vehicle", "pedestrian", "rockfall"]

# Because each folder uses 0 internally:
ORIGINAL_CLASS_INDICES = {
    'vehicles': 0,
    'motos': 0,
    'persons': 0,
    'animals': 0,
    'rocks': 0
}

# Remap to 3 paper classes
CLASS_MAP = {
    # vehicles + motos -> vehicle
    ('vehicles', ORIGINAL_CLASS_INDICES['vehicles']): 0,
    ('motos', ORIGINAL_CLASS_INDICES['motos']): 0,
    # persons + animals -> pedestrian
    ('persons', ORIGINAL_CLASS_INDICES['persons']): 1,
    ('animals', ORIGINAL_CLASS_INDICES['animals']): 1,
    # rocks -> rockfall
    ('rocks', ORIGINAL_CLASS_INDICES['rocks']): 2,
}

# Paper counts
TARGET_COUNTS = {
    0: 848,  # vehicle
    1: 854,  # pedestrian
    2: 832   # rockfall
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def remap_and_filter(lbl_path, folder):
    """Remap labels depending on source folder."""
    new_lines = []
    try:
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                if (folder, old_id) in CLASS_MAP:
                    parts[0] = str(CLASS_MAP[(folder, old_id)])
                    new_lines.append(" ".join(parts) + "\n")
        return "".join(new_lines) if new_lines else None
    except Exception as e:
        print(f"[Error] {lbl_path}: {e}")
        return None


def main(gen_root, out_root):
    print(f"Building paper dataset from: {gen_root}")
    temp_img = Path("/content/temp_all_images"); temp_img.mkdir(exist_ok=True)
    temp_lbl = Path("/content/temp_all_labels"); temp_lbl.mkdir(exist_ok=True)

    grouped = defaultdict(list)

    # --- Phase 1: remap all labels ---
    for folder in os.listdir(gen_root):
        img_dir = Path(gen_root)/folder/"imgs"
        lbl_dir = Path(gen_root)/folder/"anno"
        if not (img_dir.exists() and lbl_dir.exists()):
            continue
        print(f"Processing {folder}...")
        for lbl in lbl_dir.glob("*.txt"):
            img_jpg = img_dir / lbl.with_suffix(".jpg").name
            img_png = img_dir / lbl.with_suffix(".png").name
            img_path = img_jpg if img_jpg.exists() else img_png if img_png.exists() else None
            if not img_path:
                continue
            new_lbl = remap_and_filter(lbl, folder)
            if not new_lbl:
                continue
            new_cls = CLASS_MAP[(folder, 0)]
            shutil.copy2(img_path, temp_img / img_path.name)
            with open(temp_lbl / lbl.name, "w") as f:
                f.write(new_lbl)
            grouped[new_cls].append(img_path.name)

    print("\n[✓] Phase 1 complete — counts by class:")
    for cid, imgs in grouped.items():
        print(f"  {FINAL_CLASSES[cid]}: {len(imgs)}")

    # --- Phase 2: sample exact counts per paper ---
    sampled = {}
    for cid, target in TARGET_COUNTS.items():
        imgs = grouped[cid]
        random.shuffle(imgs)
        if len(imgs) < target:
            print(f"[Warning] Only {len(imgs)} found for {FINAL_CLASSES[cid]}, using all.")
            sampled[cid] = imgs
        else:
            sampled[cid] = imgs[:target]

    # --- Phase 3: split into train/val/test ---
    splits = {"train": [], "val": [], "test": []}
    for cid, imgs in sampled.items():
        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        splits["train"] += imgs[:n_train]
        splits["val"] += imgs[n_train:n_train+n_val]
        splits["test"] += imgs[n_train+n_val:]

    # --- Copy final images ---
    for s in ["train", "val", "test"]:
        (Path(out_root)/"images"/s).mkdir(parents=True, exist_ok=True)
        (Path(out_root)/"labels"/s).mkdir(parents=True, exist_ok=True)
        print(f"\nCopying {s} set ({len(splits[s])} images)...")
        for img_name in splits[s]:
            img_path = temp_img / img_name
            lbl_path = temp_lbl / Path(img_name).with_suffix(".txt")
            if img_path.exists() and lbl_path.exists():
                shutil.copy2(img_path, Path(out_root)/"images"/s/img_path.name)
                shutil.copy2(lbl_path, Path(out_root)/"labels"/s/lbl_path.name)

    # --- data.yaml ---
    yaml_path = Path(out_root) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_root)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(FINAL_CLASSES)}\n")
        f.write(f"names: {FINAL_CLASSES}\n")

    total = sum(len(v) for v in sampled.values())
    print(f"\n✅ Dataset ready at {out_root}")
    print(f"Total {total} images (Train/Val/Test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})")
    for cid in sampled:
        print(f"  {FINAL_CLASSES[cid]}: {len(sampled[cid])}")

    shutil.rmtree(temp_img); shutil.rmtree(temp_lbl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", default=GEN_ROOT)
    parser.add_argument("--out_root", default=OUT_ROOT)
    args = parser.parse_args()
    main(args.gen_root, args.out_root)
