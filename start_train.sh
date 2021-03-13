rm -rf ./logs/*

nohup python -u main/train_yolo_v4_tiny.py > train.log 2>&1 &