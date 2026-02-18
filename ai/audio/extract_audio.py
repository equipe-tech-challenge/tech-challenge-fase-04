import os
import subprocess
from pathlib import Path


def extract_audio_from_video(video_path: str, out_wav_path: str, sr: int = 16000) -> str:
    """
    Extrai áudio do vídeo e salva em WAV mono PCM16.
    Requer ffmpeg instalado.
    """
    video_path = str(video_path)
    out_wav_path = str(out_wav_path)

    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        "-f", "wav",
        out_wav_path
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError(
            "Falha ao extrair áudio do vídeo.\n"
            f"ffmpeg stderr:\n{proc.stderr.decode(errors='ignore')}"
        )

    if not os.path.exists(out_wav_path) or os.path.getsize(out_wav_path) < 1000:
        raise RuntimeError("Áudio extraído inválido (arquivo vazio).")

    return out_wav_path
