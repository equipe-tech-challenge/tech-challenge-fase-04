# YOLOv8 - Bleeding Detection (Modulo e Dataset)

## Visao geral
Este modulo implementa treinamento, validacao, exportacao e inferencia de um detector YOLOv8 para sangramento. O dataset segue o formato YOLO padrao e as rotinas de inferencia geram eventos e metricas agregadas para uso em pipeline.

## Estrutura de diretorios
ai/video/yolo_bleeding/
  data/
    README.md
    data.yaml
    dataset/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
  export_onnx.py
  infer_runtime.py
  infer_validar.py
  model.pt
  train.py
  validate.py

Arquivos relacionados fora do modulo:
runs/bleeding/train/weights/best.pt
yolov8n.pt
storage/results/<job_id>/bleeding_detected.mp4

## Dataset (YOLO)
### Estrutura fisica
ai/video/yolo_bleeding/data/dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

### data.yaml
Arquivo de configuracao usado por Ultralytics YOLO. Contem o caminho base do dataset e as classes:
- path: /app/ai/video/yolo_bleeding/data/dataset
- train: images/train
- val: images/val
- nc: 8
- names: lista de classes

Observacao: o campo `path` esta apontando para `/app/...` (padrao de container). Em ambiente local, ajuste se necessario.

### Formato do label (YOLO)
Cada linha em `labels/*.txt`:
<class_id> <x_center> <y_center> <width> <height>
Todos os valores sao normalizados em [0..1] pelo tamanho da imagem.

### Classes
0: Baby
1: Blood
2: Gauze
3: Hand
4: Person
5: RedGauze
6: StainedGauze
7: StainedHand

## Responsabilidades dos arquivos
- `train.py`: treina o modelo usando `yolov8n.pt` como base, com `imgsz=640`, `epochs=50`, `batch=8`, `device="cpu"`, `workers=2`, `patience=10`, e salva o melhor peso em `ai/video/yolo_bleeding/model.pt`.
- `validate.py`: valida `model.pt` com `data.yaml` usando `imgsz=640` e `device=0`.
- `export_onnx.py`: exporta `model.pt` para ONNX (`imgsz=640`, `opset=12`).
- `infer_validar.py`: CLI de inferencia para video/imagem. Usa `--source` e opcionalmente `--save` para salvar anotacoes.
- `infer_runtime.py`: inferencia para uso em pipeline, com amostragem de frames e geracao de eventos/metricas.
  - Entrada: `video_path`, `job_id`, `fps_sample=5`, `conf=0.25`
  - Saida: `events`, `metrics`, `raw`, `output_video`
  - Saida de video: `storage/results/<job_id>/bleeding_detected.mp4` (pasta base via `STORAGE_ROOT`, default `/app/storage`)
  - Thresholds de sangue: `small < 1%`, `medium < 5%`, `large >= 5%` da area do frame
  - Pooling: detecta aumento consistente do `blood_ratio` em janela de 6 amostras
  - Eventos:
    - `gauze_soaked_high`: >= 3 deteccoes de `RedGauze`
    - `blood_large_frequent`: >= 3 frames com `blood_large`
    - `blood_pooling_suspected`: >= 2 frames com pooling

## Artefatos gerados
- Treinamento: `runs/bleeding/train/weights/best.pt`
- Modelo final: `ai/video/yolo_bleeding/model.pt`
- Video anotado (infer runtime): `storage/results/<job_id>/bleeding_detected.mp4`

## Dependencias principais
- `ultralytics` (YOLOv8)
- `opencv-python` (leitura e escrita de video)
