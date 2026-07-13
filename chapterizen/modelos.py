"""Modelos de datos: dataclasses y modelos pydantic usados en todo el
pipeline. Movido sin cambios desde chapterizen.py (monolito original,
v0.0.7)."""
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field

import numpy as np


@dataclass
class ParsedAnime:
    """Resultado normalizado del parseo de un nombre de archivo de anime."""
    titulo:    str           # título limpio, listo para consultar Jikan
    temporada: Optional[int] # None si no se detectó
    episodio:  Optional[int] # None si no se detectó
    fuente:    str           # "aniparse" | "anitopy" | "aniparse+anitopy" | "fallback"


@dataclass
class TemaAudio:
    """Audio de un tema OP/ED precargado en memoria, listo para matching."""
    nombre:   str
    audio:    "np.ndarray"
    hz:       int
    frames:   int          # len(audio) // _HOP_LENGTH — precalculado
    features: "np.ndarray" # MFCC + chroma precalculados — evita recalcular en cada ventana


@dataclass
class CandidatoFFT:
    """Resultado de la fase FFT para un tema candidato."""
    tema:      "TemaAudio" # referencia al tema original — sin copiar arrays
    inicio:    float       # segundos en la ventana
    fin:       float       # segundos en la ventana
    score_fft: float


class AnimeDetectado(BaseModel):
    titulo:     str
    anilist_id: Optional[int] = None
    episodio:   Optional[int] = None
    similitud:  float

class ResultadoCoincidencia(BaseModel):
    nombre_tema: str
    inicio:      float
    fin:         float
    puntuacion:  float

class ParametrosTrabajo(BaseModel):
    video:             str
    carpeta_salida:    str
    crear_subcarpeta:  bool
    usar_exacto:       bool
    submuestreo:       int   = Field(default=32,   ge=1)
    porcion_theme:     float = Field(default=0.90, ge=0.5, le=1.0)
    puntuacion_minima: float = Field(default=0.25, ge=0.05, le=1.0)

    search_override: str = ""
    slug:            str = ""
    titulo_usado:    str = ""
    episodio:        int = 0

    model_config = {"arbitrary_types_allowed": True}

class PickRequest(BaseModel):
    kind:      str
    titulo:    str
    subtitulo: str
    columnas:  List[Tuple[str, int]]
    filas:     List[List[str]]
    payload:   List[dict]

    model_config = {"arbitrary_types_allowed": True}
