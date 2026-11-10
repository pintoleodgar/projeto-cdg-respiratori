"""Extração de features de áudio com librosa.

Para classificação de sons respiratórios, as features mais usadas na literatura:
- MFCCs (Mel-Frequency Cepstral Coefficients): captura timbre/textura
- Mel-spectrogram: representação tempo-frequência em escala perceptual
- Chroma: 12 classes de altura
- ZCR (zero-crossing rate): sinais ruidosos (wheezes/crackles)
- Spectral centroid/rolloff/bandwidth/flatness: forma do espectro
- RMS: energia
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa

from .config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, MAX_DURATION_S


def load_audio(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, offset=offset, duration=duration, mono=True)
    return y, sr


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_mel_spectrogram(
    y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 128
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    return librosa.power_to_db(S, ref=np.max)


def extract_feature_vector(
    y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC
) -> dict:
    """Vetor de features agregadas (mean/std) para um clipe — pronto para ML clássico."""
    feats: dict = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i+1}_std"] = float(mfcc[i].std())
        feats[f"mfcc{i+1}_d_mean"] = float(delta[i].mean())
        feats[f"mfcc{i+1}_d2_mean"] = float(delta2[i].mean())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = float(chroma[i].mean())

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(contrast.shape[0]):
        feats[f"contrast{i+1}_mean"] = float(contrast[i].mean())

    feats["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).mean())
    feats["rms_mean"] = float(librosa.feature.rms(y=y, hop_length=HOP_LENGTH).mean())
    feats["centroid_mean"] = float(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["rolloff_mean"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["bandwidth_mean"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["flatness_mean"] = float(librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH).mean())

    feats["duration_s"] = float(len(y) / sr)
    return feats


def extract_from_file(
    path: str | Path, max_duration: float | None = MAX_DURATION_S
) -> dict:
    """Features de um ficheiro, cortado aos primeiros `max_duration` segundos.

    O corte é o mesmo no treino e na inferência (ver config.MAX_DURATION_S).
    `max_duration=None` reproduz o comportamento antigo (áudio inteiro).
    """
    y, sr = load_audio(path, duration=max_duration)
    return extract_feature_vector(y, sr)


def extract_segment(
    y_full: np.ndarray, sr: int, start_s: float, end_s: float, min_samples: int = 1024
) -> dict | None:
    """Extrai features de um trecho de áudio já carregado.
    Retorna None se o trecho for muito curto pra computar features."""
    i0 = int(start_s * sr)
    i1 = int(end_s * sr)
    seg = y_full[i0:i1]
    if len(seg) < min_samples:
        return None
    return extract_feature_vector(seg, sr)
