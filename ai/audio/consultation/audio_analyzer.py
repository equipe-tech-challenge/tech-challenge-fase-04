from typing import Dict, Any, List
import numpy as np
import librosa


class ConsultationAudioAnalyzer:
    """
    - detecta pausas / hesitação
    - instabilidade vocal (pitch)
    - energia baixa ou oscilante
    - proxies de ansiedade/trauma (NÃO diagnóstico)
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    @staticmethod
    def _safe_mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    @staticmethod
    def _safe_std(x):
        return float(np.std(x)) if len(x) else 0.0

    def analyze(self, wav_path: str) -> Dict[str, Any]:
        y, sr = librosa.load(wav_path, sr=self.sr, mono=True)

        if y is None or len(y) < sr * 2:
            return {
                "metrics": {},
                "scores": {},
                "events": [{
                    "type": "audio_too_short",
                    "score": 0.0,
                    "description": "Áudio muito curto para análise (mínimo recomendado: 2s)."
                }]
            }

        hop_length = 512
        frame_length = 2048

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_mean = self._safe_mean(rms)
        rms_std = self._safe_std(rms)

        thr = np.percentile(rms, 15) if len(rms) else 0.0
        silent_frames = rms < (thr * 1.25)

        silent_ratio = float(np.mean(silent_frames)) if len(silent_frames) else 0.0

        f0 = librosa.yin(y, fmin=70, fmax=350, sr=sr, frame_length=frame_length, hop_length=hop_length)
        f0 = np.array(f0)

        f0_valid = f0[(f0 > 70) & (f0 < 350)]
        f0_mean = self._safe_mean(f0_valid)
        f0_std = self._safe_std(f0_valid)

        f0_diff = np.abs(np.diff(f0_valid)) if len(f0_valid) > 3 else np.array([])
        f0_instability = self._safe_mean(f0_diff)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        duration_sec = len(y) / sr

        onset_rate = float(len(onsets) / max(duration_sec, 0.001))

        hesitation_score = 0.0
        hesitation_score += min(1.0, silent_ratio / 0.35)
        hesitation_score += min(1.0, max(0.0, (1.0 - (onset_rate / 3.5))))
        hesitation_score = float(np.clip(hesitation_score / 2.0, 0.0, 1.0))

        anxiety_vocal_score = 0.0
        anxiety_vocal_score += min(1.0, f0_instability / 12.0) if f0_instability > 0 else 0.0
        anxiety_vocal_score += min(1.0, rms_std / max(rms_mean, 1e-6))
        anxiety_vocal_score += min(1.0, onset_rate / 6.0)
        anxiety_vocal_score = float(np.clip(anxiety_vocal_score / 3.0, 0.0, 1.0))

        postpartum_risk_proxy_score = float(np.clip(
            0.65 * hesitation_score + 0.35 * min(1.0, max(0.0, (0.02 - rms_mean) / 0.02)),
            0.0, 1.0
        ))

        trauma_vocal_proxy_score = float(np.clip(
            0.45 * hesitation_score + 0.35 * min(1.0, f0_instability / 12.0) + 0.20 * min(1.0, rms_std / max(rms_mean, 1e-6)),
            0.0, 1.0
        ))

        events: List[Dict[str, Any]] = []

        if hesitation_score >= 0.7:
            events.append({
                "type": "high_hesitation",
                "score": round(hesitation_score, 3),
                "description": "Hesitação/pausas elevadas no discurso (indicador vocal)."
            })

        if anxiety_vocal_score >= 0.7:
            events.append({
                "type": "high_anxiety_vocal_proxy",
                "score": round(anxiety_vocal_score, 3),
                "description": "Indicadores vocais compatíveis com ansiedade (proxy)."
            })

        if postpartum_risk_proxy_score >= 0.7:
            events.append({
                "type": "postpartum_risk_proxy_high",
                "score": round(postpartum_risk_proxy_score, 3),
                "description": "Proxy de risco aumentado (pós-parto): energia baixa + hesitação."
            })

        if trauma_vocal_proxy_score >= 0.7:
            events.append({
                "type": "trauma_proxy_high",
                "score": round(trauma_vocal_proxy_score, 3),
                "description": "Indicadores vocais compatíveis com estresse/trauma (proxy)."
            })

        return {
            "metrics": {
                "duration_sec": round(duration_sec, 3),
                "rms_mean": round(rms_mean, 6),
                "rms_std": round(rms_std, 6),
                "silent_ratio": round(silent_ratio, 3),
                "pitch_mean_hz": round(f0_mean, 3),
                "pitch_std_hz": round(f0_std, 3),
                "pitch_instability": round(f0_instability, 3),
                "onset_rate": round(onset_rate, 3),
            },
            "scores": {
                "hesitation_score": round(hesitation_score, 3),
                "anxiety_vocal_score": round(anxiety_vocal_score, 3),
                "postpartum_risk_proxy_score": round(postpartum_risk_proxy_score, 3),
                "trauma_vocal_proxy_score": round(trauma_vocal_proxy_score, 3),
            },
            "events": events,
        }
