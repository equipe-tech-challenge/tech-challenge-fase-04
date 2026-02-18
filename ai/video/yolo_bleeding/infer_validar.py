import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Caminho do vídeo ou imagem")
    parser.add_argument("--save", action="store_true", help="Salvar resultados anotados")
    args = parser.parse_args()

    model_path = Path("ai/video/yolo_bleeding/model.pt")
    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado em ai/video/yolo_bleeding/model.pt")

    model = YOLO(str(model_path))

    results = model.predict(
        source=args.source,
        imgsz=640,
        conf=0.25,
        save=args.save,
        verbose=True
    )

    for r in results:
        if r.boxes is None:
            continue
        print(f"Detecções: {len(r.boxes)}")


if __name__ == "__main__":
    main()
