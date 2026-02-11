# Dataset YOLOv8 - Sangramento (MVP)

## Estrutura
ai/video/yolo_bleeding/data/dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

## Formato do label (YOLO)
<class_id> <x_center> <y_center> <width> <height>
Valores normalizados (0..1).

## Classes
  0: gauze_soaked
  1: gauze_stained
  2: gauze_clean
