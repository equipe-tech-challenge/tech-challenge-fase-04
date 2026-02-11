from pathlib import Path
from ultralytics import YOLO


def main():
    model_path = Path("ai/video/yolo_bleeding/model.pt")
    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado.")

    model = YOLO(str(model_path))

    model.export(format="onnx", imgsz=640, opset=12)
    print("Export concluído.")


if __name__ == "__main__":
    main()
