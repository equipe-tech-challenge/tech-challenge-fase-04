from ultralytics import YOLO
from pathlib import Path


def main():
    base_model = "yolov8n.pt"

    data_yaml = Path("ai/video/yolo_bleeding/data/data.yaml")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml}")

    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=50,
        batch=8,
        device="cpu",
        workers=2,
        project="runs/bleeding",
        name="train",
        patience=10,
        verbose=True,
    )

    best_path = Path("runs/bleeding/train/weights/best.pt")
    target_path = Path("ai/video/yolo_bleeding/model.pt")

    if best_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(best_path.read_bytes())
        print(f"Modelo salvo em: {target_path}")
    else:
        print("Treino terminou, mas best.pt não foi encontrado.")

    return results


if __name__ == "__main__":
    main()
