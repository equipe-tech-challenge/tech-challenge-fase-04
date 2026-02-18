import os

from typing import Dict, Any
from ai.audio.consultation.audio_analyzer import ConsultationAudioAnalyzer


def process_audio(wav_path: str) -> Dict[str, Any]:
    analyzer = ConsultationAudioAnalyzer(sr=16000)

    analyze = analyzer.analyze(wav_path)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    return analyze
