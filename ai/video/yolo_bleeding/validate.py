from ultralytics import YOLO
from pathlib import Path


def main():
    model_path = Path("ai/video/yolo_bleeding/model.pt")
    data_yaml = Path("ai/video/yolo_bleeding/data/data.yaml")

    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Treine primeiro e copie best.pt para ai/video/yolo_bleeding/model.pt"
        )

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        device=0,  # ou "cpu"
    )

    print(metrics)


if __name__ == "__main__":
    main()
