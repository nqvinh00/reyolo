# reyolo

## Prepration
- Requirements:
```
pip install -r requirements.txt
```

- To get weights file:
```
mkdir weight && cd weight
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights 
wget https://pjreddie.com/media/files/darknet53.conv.74
```

- To get coco dataset:
```
./data/get_coco_dataset.sh
```

## Detection
- Image detection:
```
# single image
python image_detect.py --images "./test_img/eagle.jpg" --cfg "cfg/yolov3.cfg" --weights "./weight/yolov3.weights"

# multiple images
python image_detect.py --images "./test_img"
```

- Video detection:
```
# video file source
python video_detect.py --video "./playback.mp4" --source "video"

# webcam source
python video_detect.py --source "webcam"
```

## Training
```
python train/train.py --cfg "./cfg/yolov3.cfg" --train_path "./data/trainvalno5k.txt" --valid_path "./data/5k.txt" --epochs 100 --cpus 4 --pretrained_weights "./weight/yolov3-tiny.pt"

# experiment dir: ./experiments
# training chart: wandb
```
