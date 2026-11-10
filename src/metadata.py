"""Constrói DataFrames de metadata do ICBHI 2017.

Fontes:
- patient_diagnosis.csv: <patient_id>,<diagnosis>
- demographic_info.txt:  patient_id age sex adult_BMI child_weight child_height
- nomes dos .wav:        <pid>_<recidx>_<local>_<modo>_<equipamento>.wav
- .txt por gravação:     start end crackles wheezes (por ciclo respiratório)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import AUDIO_DIR, DIAGNOSIS_CSV, DEMOGRAPHIC_TXT


TARGET_CLASSES = ["Healthy", "COPD", "URTI"]
DISEASE_CLASSES = ["COPD", "URTI"]


CHEST_LOCATION_MAP = {
    "Tc": "Trachea",
    "Al": "Anterior left",
    "Ar": "Anterior right",
    "Pl": "Posterior left",
    "Pr": "Posterior right",
    "Ll": "Lateral left",
    "Lr": "Lateral right",
}

ACQUISITION_MODE_MAP = {
    "sc": "single channel",
    "mc": "multichannel",
}

EQUIPMENT_MAP = {
    "AKGC417L": "AKG C417L Microphone",
    "Meditron": "WelchAllyn Meditron",
    "LittC2SE": "Littmann Classic II SE",
    "Litt3200": "Littmann 3200 Electronic",
}

# Fonte única da lista de equipamentos suportados (app, CLI, testes).
EQUIPMENTS = sorted(EQUIPMENT_MAP.values())
DEFAULT_EQUIPMENT = "WelchAllyn Meditron"


def load_diagnosis(path: Path = DIAGNOSIS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["patient_id", "diagnosis"])
    df["patient_id"] = df["patient_id"].astype(int)
    return df


def load_demographics(path: Path = DEMOGRAPHIC_TXT) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"],
        na_values=["NA"],
    )
    df["patient_id"] = df["patient_id"].astype(int)
    return df


def parse_filename(stem: str) -> dict:
    """`101_1b1_Al_sc_Meditron` -> dict com campos parseados."""
    parts = stem.split("_")
    if len(parts) != 5:
        return {
            "patient_id": None,
            "recording_idx": None,
            "chest_location_code": None,
            "chest_location": None,
            "acquisition_mode_code": None,
            "acquisition_mode": None,
            "equipment_code": None,
            "equipment": None,
        }
    pid, recidx, loc, mode, equip = parts
    return {
        "patient_id": int(pid),
        "recording_idx": recidx,
        "chest_location_code": loc,
        "chest_location": CHEST_LOCATION_MAP.get(loc, loc),
        "acquisition_mode_code": mode,
        "acquisition_mode": ACQUISITION_MODE_MAP.get(mode, mode),
        "equipment_code": equip,
        "equipment": EQUIPMENT_MAP.get(equip, equip),
    }


def list_recordings(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    rows = []
    for wav in sorted(audio_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        info = parse_filename(wav.stem)
        info["filename"] = wav.name
        info["wav_path"] = str(wav)
        info["txt_path"] = str(txt) if txt.exists() else None
        rows.append(info)
    return pd.DataFrame(rows)


def load_cycles(txt_path: str | Path) -> pd.DataFrame:
    """Ciclos respiratórios anotados: start, end, crackles, wheezes."""
    return pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        names=["start", "end", "crackles", "wheezes"],
    )


def build_recordings_dataframe(
    audio_dir: Path = AUDIO_DIR,
    diagnosis_csv: Path = DIAGNOSIS_CSV,
    demographic_txt: Path = DEMOGRAPHIC_TXT,
) -> pd.DataFrame:
    """DataFrame com uma linha por gravação (920 linhas), enriquecido com
    diagnóstico do paciente e dados demográficos."""
    rec = list_recordings(audio_dir)
    diag = load_diagnosis(diagnosis_csv)
    demo = load_demographics(demographic_txt)
    df = rec.merge(diag, on="patient_id", how="left")
    df = df.merge(demo, on="patient_id", how="left")
    return df


def filter_target_classes(
    df: pd.DataFrame, classes: list[str] = TARGET_CLASSES
) -> pd.DataFrame:
    """Mantém apenas gravações cujo diagnóstico está em `classes`."""
    return df[df["diagnosis"].isin(classes)].copy().reset_index(drop=True)


def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as colunas-alvo para as duas tarefas de modelagem:
    - `label_multi`: Healthy | COPD | URTI    (3 classes)
    - `label_binary`: Healthy | Disease       (Disease = COPD ∪ URTI)
    """
    df = df.copy()
    df["label_multi"] = df["diagnosis"]
    df["label_binary"] = df["diagnosis"].apply(
        lambda d: "Healthy" if d == "Healthy" else "Disease"
    )
    return df


def build_cycles_dataframe(recordings_df: pd.DataFrame) -> pd.DataFrame:
    """Expande para uma linha por ciclo respiratório (~6.9k linhas)."""
    frames = []
    for _, row in recordings_df.iterrows():
        if not row["txt_path"]:
            continue
        cycles = load_cycles(row["txt_path"])
        cycles["filename"] = row["filename"]
        cycles["patient_id"] = row["patient_id"]
        cycles["diagnosis"] = row.get("diagnosis")
        cycles["duration"] = cycles["end"] - cycles["start"]
        frames.append(cycles)
    out = pd.concat(frames, ignore_index=True)
    out["cycle_idx"] = out.groupby("filename").cumcount()
    return out
