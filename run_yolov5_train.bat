@echo off
:: ---- STEP 1: clone yolov5 (skip if already cloned)
if not exist yolov5 (
    git clone https://github.com/ultralytics/yolov5.git
)

:: ---- STEP 2: install dependencies
cd yolov5
pip install -r requirements.txt

:: ---- STEP 3: copy hyperparameters file from parent
copy ..\hyp_paper.yaml .\hyp_paper.yaml

:: ---- STEP 4: run training (300 epochs, 640x640)
python train.py ^
  --img 640 ^
  --batch 16 ^
  --epochs 300 ^
  --data ..\data_train.yaml ^
  --cfg models/yolov5s.yaml ^
  --weights yolov5s.pt ^
  --hyp hyp_paper.yaml ^
  --name rail_yolov5_paper ^
  --project ..\runs

pause
