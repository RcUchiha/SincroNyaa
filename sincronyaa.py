__author__  = "CiferrC"
__license__ = "MIT"
__version__ = "0.2.0"

import sys
import os
import subprocess
import tempfile
import bisect
from pathlib import Path
import json

import numpy as np
import librosa
import pysubs2
from scipy.signal import fftconvolve

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
    QTextEdit, QFrame, QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────

SAMPLE_RATE       = 16000
HOP_LENGTH        = 512
WINDOW_SEC        = 30       # duración de cada ventana de correlación
STEP_SEC          = 10       # paso entre ventanas
SEARCH_MARGIN_SEC = 300      # margen de búsqueda en el video nuevo (±segundos)
CONFIDENCE_THRESH = 3.0      # mínima confianza para considerar una ventana fiable
OFFSET_JUMP_SEC   = 1.0      # diferencia de offset que marca un nuevo segmento
GAP_SEC           = 20.0     # hueco temporal que marca un nuevo segmento (OP/ED)
KEYFRAME_SNAP_MS  = 170     # umbral para considerar un tiempo anclado a un keyframe


# ─────────────────────────────────────────────
#  FFMPEG / FFPROBE
# ─────────────────────────────────────────────

def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_audio(video_path: str, wav_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-8:]) or "(sin salida de stderr)"
        raise RuntimeError(
            f"ffmpeg fallo al extraer audio de {video_path} "
            f"(codigo {result.returncode}):\n{stderr_tail}"
        )


def get_video_fps(video_path: str) -> tuple[int, int]:
    """
    Devuelve el framerate del video como fracción entera (num, den).
    Ej: 23.976fps -> (24000, 1001), 24fps -> (24, 1)
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        data = json.loads(result.stdout)
        rate = data["streams"][0]["r_frame_rate"]
        num, den = (int(x) for x in rate.split("/"))
        return num, den
    except Exception:
        return 24000, 1001


def extract_keyframes(video_path: str) -> list[int]:
    """
    Extrae los timestamps de los keyframes (I-frames) en milisegundos.

    Aegisub convierte pts_time a ms con floor(pts_time * 1000), sin redondeo.
    Usamos exactamente esa formula para que los valores coincidan con los
    keyframes que Aegisub muestra en el timeline.

    Se pide best_effort_timestamp_time como fallback por si pts_time no
    esta disponible en algun stream.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v",
        "-show_frames",
        "-show_entries", "frame=pts_time,best_effort_timestamp_time,pict_type",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    keyframes_ms = []
    try:
        data = json.loads(result.stdout)
        for frame in data.get("frames", []):
            if frame.get("pict_type") != "I":
                continue
            t_str = frame.get("pts_time") or frame.get("best_effort_timestamp_time")
            if t_str is None:
                continue
            # floor() igual que Aegisub, sin round()
            ms = int(float(t_str) * 1000)
            keyframes_ms.append(ms)
    except Exception:
        pass
    return sorted(keyframes_ms)


def parse_keyframe_frames(kf_path: str) -> list[int]:
    """
    Lee un archivo de keyframes y devuelve los numeros de frame como lista de enteros.
    Soporta dos formatos:

    1. Aegisub / wwxd / scxvid / vstools:
       Cabecera "# keyframe format v1", lineas "FRAME I -1".

    2. XviD 2pass stats:
       Cabecera "# XviD 2pass stat file", lineas "i/p/b ...".
       El numero de frame es la posicion ordinal de cada linea de frame.
    """
    lines = []
    with open(kf_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    is_xvid = any("xvid" in l.lower() and "stat" in l.lower() for l in lines[:3])

    frames = []
    if is_xvid:
        frame_idx = 0
        for line in lines:
            if line.startswith("#"): continue
            ft = line[0].lower() if line else ""
            if ft in ("i", "p", "b"):
                if ft == "i":
                    frames.append(frame_idx)
                frame_idx += 1
    else:
        for line in lines:
            if line.startswith("#"): continue
            if line.lower().startswith("fps"): continue
            parts = line.split()
            try:
                frames.append(int(parts[0]))
            except (ValueError, IndexError):
                pass

    return sorted(frames)


# Script que se ejecuta en un subproceso para extraer timestamps con VapourSynth.
# Se mantiene como string para escribirlo a un archivo temporal cuando se necesite.
_VS_TIMESTAMPS_SCRIPT = r'''
import sys, json

try:
    import vapoursynth as vs
except Exception as e:
    print(json.dumps({"error": f"import vapoursynth fallo: {e}"}))
    sys.exit(1)

video_path = sys.argv[1]
core = vs.core

clip = None
last_error = None
for source_attr in ("lsmas", "ffms2"):
    try:
        ns = getattr(core, source_attr)
    except AttributeError:
        continue
    try:
        if source_attr == "lsmas":
            clip = ns.LWLibavSource(video_path)
        else:
            clip = ns.Source(video_path)
        break
    except Exception as e:
        last_error = e
        continue

if clip is None:
    print(json.dumps({"error": f"no se pudo abrir el video: {last_error}"}))
    sys.exit(1)

timestamps = []
fps_num, fps_den = clip.fps_num, clip.fps_den

# IMPORTANTE: siempre intentar leer _AbsoluteTime por frame. El fps_num/fps_den
# reportado por clip puede no corresponder a CFR real cuando lsmas/ffms2
# aplica IVTC o decimation interna — en ese caso los timestamps reales son
# variables aunque el fps reportado parezca constante.
try:
    use_props = False
    # Probar con el frame 0 si tiene _AbsoluteTime
    try:
        f0 = clip.get_frame(0)
        if "_AbsoluteTime" in f0.props:
            use_props = True
        del f0
    except Exception:
        pass

    if use_props:
        for n in range(clip.num_frames):
            f = clip.get_frame(n)
            timestamps.append(int(float(f.props["_AbsoluteTime"]) * 1000))
            del f
    elif fps_num > 0 and fps_den > 0:
        # Fallback CFR puro si _AbsoluteTime no esta disponible
        for n in range(clip.num_frames):
            timestamps.append(n * 1000 * fps_den // fps_num)
    else:
        # Ultimo fallback: aproximacion a 23.976
        for n in range(clip.num_frames):
            timestamps.append(int(n * 1001000 / 24000))
except Exception as e:
    print(json.dumps({"error": f"error iterando frames: {e}"}))
    sys.exit(1)

print(json.dumps({"timestamps": timestamps}))
'''


def build_frame_timestamps(video_path: str) -> list[int]:
    """
    Extrae el timestamp en ms de cada frame del video en orden de presentacion,
    usando VapourSynth con el mismo source filter (lsmas/ffms2) que usan
    wwxd, scxvid y vstools al generar los archivos de keyframes.

    VapourSynth no es estable cuando se llama desde un QThread secundario
    (puede terminar el proceso sin lanzar excepcion). Para evitar esto se
    ejecuta en un subproceso Python aislado y los timestamps se reciben
    via stdout en formato JSON. Si el subproceso falla por cualquier razon,
    se lanza una excepcion con el mensaje correspondiente.
    """
    # Escribir el script auxiliar a un archivo temporal
    fd, script_path = tempfile.mkstemp(suffix="_vs_timestamps.py", text=True)
    try:
        os.write(fd, _VS_TIMESTAMPS_SCRIPT.encode("utf-8"))
        os.close(fd)

        proc = subprocess.run(
            [sys.executable, script_path, video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if proc.returncode != 0:
        raise RuntimeError(
            f"Subproceso de VapourSynth fallo (codigo {proc.returncode}). "
            f"stderr: {proc.stderr.strip() or '(vacio)'}"
        )

    try:
        data = json.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo parsear la salida del subproceso: {e}. "
            f"stdout: {proc.stdout[:300]!r}"
        )

    if "error" in data:
        raise RuntimeError(data["error"])

    return data.get("timestamps", [])


def keyframe_frames_to_ms(frame_numbers: list[int], frame_timestamps: list[int]) -> list[int]:
    """
    Convierte numeros de frame a ms usando la tabla de timestamps reales del video.
    Frames fuera de rango se ignoran.
    """
    n = len(frame_timestamps)
    return sorted(frame_timestamps[f] for f in frame_numbers if 0 <= f < n)



def nearest_keyframe(t_ms: int, keyframes_ms: list[int]) -> int | None:
    """
    Devuelve el keyframe más cercano a t_ms, o None si la lista está vacía.
    Usa búsqueda binaria para eficiencia.
    """
    if not keyframes_ms:
        return None
    idx = bisect.bisect_left(keyframes_ms, t_ms)
    candidates = []
    if idx < len(keyframes_ms):
        candidates.append(keyframes_ms[idx])
    if idx > 0:
        candidates.append(keyframes_ms[idx - 1])
    return min(candidates, key=lambda k: abs(k - t_ms))


def _ms_to_ass_time(ms: int) -> str:
    """Convierte ms a formato ASS H:MM:SS.CC."""
    if ms < 0:
        ms = 0
    h  = ms // 3600000
    ms = ms %  3600000
    m  = ms // 60000
    ms = ms %  60000
    s  = ms // 1000
    cs = (ms %  1000) // 10
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _prev_frame_ms(kf_ms: int, frame_timestamps_new: list[int], fallback_frame_ms: int = 42) -> int:
    """
    Devuelve el timestamp del frame inmediatamente anterior al frame del keyframe.
    Si frame_timestamps_new no esta disponible (lista vacia), aproxima restando
    fallback_frame_ms (duracion de un frame al fps real del video nuevo si se
    pudo determinar, o ~42ms/23.976fps por defecto).
    """
    if not frame_timestamps_new:
        return max(0, kf_ms - fallback_frame_ms)
    idx = bisect.bisect_left(frame_timestamps_new, kf_ms)
    if idx > 0:
        return frame_timestamps_new[idx - 1]
    return max(0, kf_ms - fallback_frame_ms)


def _ass_round_for_end(kf_ms: int, frame_timestamps_new: list[int], fallback_frame_ms: int = 42) -> int:
    """
    Para snap de END: queremos que en Aegisub la linea termine en el frame
    inmediatamente anterior al frame del keyframe. ASS guarda timestamps en
    centesimas (precision 10ms).

    Aegisub considera "frame final" al ultimo frame cuyo timestamp es <= end.
    Para que el frame anterior al keyframe (con timestamp T_prev) sea el frame
    final, el end debe ser >= T_prev. Usamos ceil a centesimas para garantizar
    eso despues del truncado del ASS.
    """
    prev_ms = _prev_frame_ms(kf_ms, frame_timestamps_new, fallback_frame_ms)
    # ceil a centesimas: redondea hacia arriba al multiplo de 10 mas cercano
    return -(-prev_ms // 10) * 10


def _ass_round_for_start(kf_ms: int) -> int:
    """
    Para snap de START: queremos que en Aegisub la linea empiece en el frame
    del keyframe. Empiricamente, Aegisub muestra el frame del keyframe cuando
    el start tiene valor floor(kf_ms / 10) * 10 (truncado a centesima inferior).
    """
    return (kf_ms // 10) * 10


def snap_to_keyframes(
    subs: pysubs2.SSAFile,
    keyframes_old_ms: list[int],
    keyframes_new_ms: list[int],
    frame_timestamps_new: list[int] | None = None,
    threshold_ms: int = KEYFRAME_SNAP_MS,
    fallback_frame_ms: int = 42,
) -> tuple[int, int, list[str]]:
    """
    Para cada línea ya desplazada por el offset de audio:
    - Comprueba si el start y/o end ORIGINAL (pre-offset, guardado en line._orig_start/end)
      estaba dentro de `threshold_ms` de un keyframe del video antiguo.
    - Si es así, hace snap del tiempo correspondiente en el video nuevo al keyframe
      más cercano en keyframes_new_ms, siempre que también esté dentro del umbral.

    Para `start`: snap al timestamp del keyframe (truncado a centesimas).
    Para `end`: snap al timestamp del FRAME ANTERIOR al keyframe (ceil a centesimas),
      siguiendo la convencion de timing en fansub donde el end es exclusivo —
      la linea debe terminar antes de que comience la escena nueva.

    PARES PEGADOS: si el END de una línea y el START de otra estaban anclados al
    mismo keyframe en el video antiguo (convención común en fansub para evitar
    parpadeo en cambios de escena), ambos tiempos se hacen IDÉNTICOS en el video
    nuevo usando el valor del START (timestamp del frame del keyframe). Así se
    mantiene la línea sin vacío intermedio.

    Si frame_timestamps_new no se pasa, el frame anterior se aproxima restando
    fallback_frame_ms (por defecto ~42ms, un frame a 23.976fps).

    Retorna (n_starts_snapped, n_ends_snapped, detalles_por_linea) para el log.
    """
    if frame_timestamps_new is None:
        frame_timestamps_new = []
    snapped_starts = 0
    snapped_ends   = 0
    details: list[str] = []

    # ── Detectar pares pegados en el subtítulo original ──────────────────
    # Un par "pegado" es: línea A.end y línea B.start están ambos dentro del
    # umbral del mismo keyframe del video antiguo. Para esos pares, ambos
    # tiempos en el video nuevo deben quedar iguales (snap a start), sin
    # vacío intermedio.
    # Primero, mapear cada línea con su keyframe_old del end (si aplica) y del start
    line_end_kf:   dict[int, int] = {}
    line_start_kf: dict[int, int] = {}
    for i, line in enumerate(subs, 1):
        ne = nearest_keyframe(line._orig_end, keyframes_old_ms)
        if ne is not None and abs(line._orig_end - ne) <= threshold_ms:
            line_end_kf[i] = ne
        ns = nearest_keyframe(line._orig_start, keyframes_old_ms)
        if ns is not None and abs(line._orig_start - ns) <= threshold_ms:
            line_start_kf[i] = ns

    # Buscar pares: line A.end y line B.start con mismo keyframe_old
    glued_ends:   set[int] = set()   # líneas cuyo end debe pegarse al start de otra
    glued_starts: set[int] = set()   # líneas cuyo start es el "ancla" del pegado
    for i, kf_end in line_end_kf.items():
        for j, kf_start in line_start_kf.items():
            if i == j:
                continue
            if kf_end == kf_start:
                glued_ends.add(i)
                glued_starts.add(j)

    # Líneas a debuggear (números de línea 1-indexed). Vaciar para desactivar.
    DEBUG_LINES: set[int] = set()

    for i, line in enumerate(subs, 1):
        orig_start_ms = line._orig_start
        orig_end_ms   = line._orig_end

        start_snapped = False
        end_snapped   = False
        new_start_before = line.start
        new_end_before   = line.end

        dbg = i in DEBUG_LINES
        if dbg:
            details.append(
                f"      [DEBUG L{i}] orig_start={orig_start_ms} orig_end={orig_end_ms} "
                f"| post_offset start={line.start} end={line.end}"
            )

        # ── snap de start ────────────────────────────────────────────────
        near_old = nearest_keyframe(orig_start_ms, keyframes_old_ms)
        if dbg:
            d_old = abs(orig_start_ms - near_old) if near_old is not None else "N/A"
            details.append(
                f"      [DEBUG L{i}] start: near_old_kf={near_old} dist_old={d_old}"
            )
        if near_old is not None and abs(orig_start_ms - near_old) <= threshold_ms:
            near_new = nearest_keyframe(line.start, keyframes_new_ms)
            if dbg:
                d_new = abs(line.start - near_new) if near_new is not None else "N/A"
                details.append(
                    f"      [DEBUG L{i}] start: near_new_kf={near_new} dist_new={d_new} threshold={threshold_ms}"
                )
            if near_new is not None and abs(line.start - near_new) <= threshold_ms:
                line.start = _ass_round_for_start(near_new)
                snapped_starts += 1
                start_snapped = True
                if dbg:
                    details.append(
                        f"      [DEBUG L{i}] start ASIGNADO a {line.start} (kf bruto={near_new})"
                    )

        # ── snap de end ──────────────────────────────────────────────────
        near_old = nearest_keyframe(orig_end_ms, keyframes_old_ms)
        if dbg:
            d_old = abs(orig_end_ms - near_old) if near_old is not None else "N/A"
            details.append(
                f"      [DEBUG L{i}] end: near_old_kf={near_old} dist_old={d_old}"
            )
        if near_old is not None and abs(orig_end_ms - near_old) <= threshold_ms:
            near_new = nearest_keyframe(line.end, keyframes_new_ms)
            if dbg:
                d_new = abs(line.end - near_new) if near_new is not None else "N/A"
                details.append(
                    f"      [DEBUG L{i}] end: near_new_kf={near_new} dist_new={d_new} threshold={threshold_ms}"
                )
            if near_new is not None and abs(line.end - near_new) <= threshold_ms:
                if i in glued_ends:
                    # Este end debe quedar pegado al start de otra línea: usar
                    # el mismo valor que tendría un start (timestamp del keyframe).
                    line.end = _ass_round_for_start(near_new)
                else:
                    line.end = _ass_round_for_end(near_new, frame_timestamps_new, fallback_frame_ms)
                snapped_ends += 1
                end_snapped = True
                if dbg:
                    details.append(
                        f"      [DEBUG L{i}] end ASIGNADO a {line.end} (kf bruto={near_new}, glued={i in glued_ends})"
                    )

        if start_snapped or end_snapped:
            tag = []
            if start_snapped:
                tag.append(f"start {_ms_to_ass_time(new_start_before)} → {_ms_to_ass_time(line.start)}")
            if end_snapped:
                tag.append(f"end {_ms_to_ass_time(new_end_before)} → {_ms_to_ass_time(line.end)}")
            details.append(f"      Línea {i}: " + " · ".join(tag))

    return snapped_starts, snapped_ends, details


# ─────────────────────────────────────────────
#  EXTRACCIÓN DE CARACTERÍSTICAS
# ─────────────────────────────────────────────

def compute_features(audio_path: str) -> tuple[np.ndarray, int, int]:
    """
    Vector de características por frame combinando:
    - Mel-spectrogram (energía espectral)  — peso 0.3
    - Onset strength (ritmo/cortes)        — peso 0.7

    El onset recibe mayor peso porque es el más discriminativo para
    detectar cortes de contenido independientemente de la tonalidad.
    El chroma se elimina: añade ruido en contenido hablado y no mejora
    la detección de offset para subtítulos.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel     = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=64)
    mel_db  = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.mean(mel_db, axis=0)

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    def norm(v: np.ndarray) -> np.ndarray:
        std = np.std(v)
        return (v - np.mean(v)) / (std if std > 0 else 1.0)

    min_len = min(len(mel_feat), len(onset))
    combined = (
        norm(mel_feat[:min_len]) * 0.3 +
        norm(onset[:min_len])    * 0.7
    )
    return combined, HOP_LENGTH, sr


# ─────────────────────────────────────────────
#  CORRELACIÓN POR VENTANAS
# ─────────────────────────────────────────────

def find_offsets_by_windows(
    feat_old: np.ndarray,
    feat_new: np.ndarray,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    search_margin_sec: float = SEARCH_MARGIN_SEC,
) -> list[tuple[float, float, float]]:
    """
    Calcula el offset local en ventanas deslizantes sobre feat_old.

    Para cada ventana, busca su mejor coincidencia en feat_new dentro de un
    margen temporal (no en todo el audio) usando correlación cruzada FFT.

    Retorna lista de (t_old_sec, offset_sec, confianza).
    La confianza es el pico de correlación normalizado por la desviación
    estándar de la señal de correlación: valores altos → ventana fiable.
    """
    frame_dur    = hop_length / sr
    win_frames   = int(window_sec / frame_dur)
    step_frames  = int(step_sec   / frame_dur)
    margin_frames = int(search_margin_sec / frame_dur)

    results = []

    for start in range(0, len(feat_old) - win_frames, step_frames):
        window_old = feat_old[start : start + win_frames]
        t_old      = start * frame_dur

        search_start = max(0, start - margin_frames)
        search_end   = min(len(feat_new), start + win_frames + margin_frames)
        region_new   = feat_new[search_start:search_end]

        if len(region_new) < win_frames:
            continue

        a = window_old - np.mean(window_old)
        b = region_new  - np.mean(region_new)
        corr = fftconvolve(b, a[::-1], mode="full")

        best_idx   = int(np.argmax(corr))
        confidence = float(corr[best_idx] / (np.std(corr) + 1e-8))

        # lag relativo a search_start, convertido a segundos
        lag_frames = best_idx - (win_frames - 1)
        offset_sec = (search_start + lag_frames - start) * frame_dur

        results.append((t_old, offset_sec, confidence))

    return results


# ─────────────────────────────────────────────
#  CLUSTERING DE OFFSETS → SEGMENTOS
# ─────────────────────────────────────────────

def cluster_offsets(
    window_results: list[tuple[float, float, float]],
    confidence_threshold: float = CONFIDENCE_THRESH,
    offset_jump_sec: float = OFFSET_JUMP_SEC,
    gap_sec: float = GAP_SEC,
) -> list[tuple[float, float, float]]:
    """
    Agrupa las ventanas fiables por offset similar y continuidad temporal.

    Un nuevo segmento comienza cuando:
    - El offset salta más de `offset_jump_sec` respecto al segmento actual, O
    - Hay un hueco temporal > `gap_sec` entre ventanas consecutivas
      (señal de que hay OP/ED entre medias con baja confianza).

    Dentro de cada segmento el offset final es el promedio ponderado por
    confianza de todas las ventanas que lo componen.

    Retorna lista de (t_start_sec, t_end_sec, offset_sec).
    """
    reliable = [
        (t, off, conf)
        for t, off, conf in window_results
        if conf >= confidence_threshold
    ]
    if not reliable:
        # Si no hay ninguna ventana fiable, usar la de mayor confianza disponible
        if window_results:
            best = max(window_results, key=lambda x: x[2])
            return [(best[0], best[0], best[1])]
        return []

    segments: list[tuple[float, float, float]] = []
    seg_t_start                = reliable[0][0]
    seg_off_accum              = reliable[0][1] * reliable[0][2]
    seg_conf_accum             = reliable[0][2]
    prev_t                     = reliable[0][0]

    for t, off, conf in reliable[1:]:
        offset_jump = abs(off - (seg_off_accum / seg_conf_accum)) > offset_jump_sec
        time_gap    = (t - prev_t) > gap_sec

        if offset_jump or time_gap:
            segments.append((
                seg_t_start,
                prev_t,
                seg_off_accum / seg_conf_accum,
            ))
            seg_t_start    = t
            seg_off_accum  = off * conf
            seg_conf_accum = conf
        else:
            seg_off_accum  += off * conf
            seg_conf_accum += conf

        prev_t = t

    segments.append((
        seg_t_start,
        prev_t,
        seg_off_accum / seg_conf_accum,
    ))

    return segments


# ─────────────────────────────────────────────
#  APLICAR OFFSETS POR SEGMENTO
# ─────────────────────────────────────────────

def apply_segmented(
    sub_path: str,
    segments: list[tuple[float, float, float]],
    fallback_offset: float,
) -> pysubs2.SSAFile:
    """
    Aplica el offset correcto a cada línea de subtítulo según en qué segmento
    cae su tiempo de inicio (medido sobre el video antiguo).

    Guarda los tiempos originales en line._orig_start / line._orig_end
    para que el paso posterior de snap a keyframes pueda consultarlos.

    El margen de ±5 s en la búsqueda de segmento cubre pequeñas
    imprecisiones en los bordes calculados por el clustering.
    Si ningún segmento cubre la línea se usa fallback_offset.

    Devuelve el objeto SSAFile modificado (sin guardar a disco).
    """
    subs = pysubs2.load(sub_path)

    def get_offset(t_sec: float) -> float:
        for t_start, t_end, off in segments:
            if t_start - 5.0 <= t_sec <= t_end + 5.0:
                return off
        return fallback_offset

    for line in subs:
        start_sec   = line.start / 1000.0
        duration_ms = line.end - line.start
        off = get_offset(start_sec)
        # Guardar tiempos originales (pre-offset) para el paso de snap posterior
        line._orig_start = line.start
        line._orig_end   = line.end
        new_start_ms = max(0, int((start_sec + off) * 1000))
        line.start = new_start_ms
        line.end   = new_start_ms + duration_ms

    return subs  # no guardar todavía; el caller aplica snap y luego guarda


# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────

class SyncWorker(QThread):
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, old_video: str, new_video: str, sub_path: str, output_path: str,
                 kf_old_path: str = "", kf_new_path: str = ""):
        super().__init__()
        self.old_video   = old_video
        self.new_video   = new_video
        self.sub_path    = sub_path
        self.output_path = output_path
        self.kf_old_path = kf_old_path
        self.kf_new_path = kf_new_path
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True

    def _resolver_keyframes(
        self,
        video_path: str,
        kf_path: str,
        etiqueta: str,
        progreso_carga: int,
        progreso_lectura: int,
    ) -> tuple[list[int], list[int]]:
        """
        Resuelve los keyframes (ms) de un video: si `kf_path` viene cargado,
        parsea el archivo y convierte sus numeros de frame a ms usando
        timestamps reales de VapourSynth, cayendo a deteccion automatica por
        ffprobe si VapourSynth falla. Si no hay `kf_path`, extrae los
        keyframes automaticamente por ffprobe directamente.

        `etiqueta` ("antiguo"/"nuevo") solo se usa para los mensajes de log.
        `progreso_carga`/`progreso_lectura` son los valores de progreso a
        emitir en cada paso, iguales a los que usaba el bloque original.

        Devuelve (keyframes_ms, frame_timestamps_ms); frame_timestamps_ms
        queda vacio salvo que se haya usado VapourSynth con exito.
        """
        if kf_path:
            self.log.emit(f"🔑 Cargando keyframes del video {etiqueta} (archivo)...")
            self.progress.emit(progreso_carga)
            kf_frames = parse_keyframe_frames(kf_path)
            self.log.emit(f"🔑 Leyendo timestamps del video {etiqueta} (VapourSynth)...")
            self.progress.emit(progreso_lectura)
            try:
                ts = build_frame_timestamps(video_path)
                kf_ms = keyframe_frames_to_ms(kf_frames, ts)
                self.log.emit(f"   ↳ {len(kf_ms)} keyframes cargados (timestamps reales)")
                return kf_ms, ts
            except Exception as e:
                self.log.emit(f"   ⚠ VapourSynth fallo: {e}")
                self.log.emit("   ↳ Cayendo a deteccion automatica con ffprobe.")
                return extract_keyframes(video_path), []
        else:
            self.log.emit(f"🔑 Extrayendo keyframes del video {etiqueta}...")
            self.progress.emit(progreso_lectura)
            return extract_keyframes(video_path), []

    def run(self):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                old_wav = os.path.join(tmp, "old.wav")
                new_wav = os.path.join(tmp, "new.wav")

                # ── Extracción de audio ──────────────────────────────────
                self.log.emit("🎬 Extrayendo audio del video original...")
                self.progress.emit(5)
                extract_audio(self.old_video, old_wav)
                if self._cancelled: return

                self.log.emit("🎬 Extrayendo audio del video nuevo...")
                self.progress.emit(12)
                extract_audio(self.new_video, new_wav)
                if self._cancelled: return

                # ── Características ──────────────────────────────────────
                self.log.emit("🔍 Analizando características de audio (mel + onset)...")
                self.progress.emit(22)
                feat_old, hop, sr = compute_features(old_wav)
                feat_new, _,   _  = compute_features(new_wav)
                if self._cancelled: return

                # ── Correlación por ventanas ─────────────────────────────
                self.log.emit("📐 Calculando offsets por ventanas deslizantes...")
                self.log.emit(f"   ↳ Ventana: {WINDOW_SEC}s · Paso: {STEP_SEC}s")
                self.progress.emit(35)
                window_results = find_offsets_by_windows(feat_old, feat_new, hop, sr)
                if self._cancelled: return

                total_windows    = len(window_results)
                reliable_windows = sum(1 for _, _, c in window_results if c >= CONFIDENCE_THRESH)
                self.log.emit(
                    f"   ↳ Ventanas analizadas: {total_windows} "
                    f"· Fiables: {reliable_windows}"
                )
                self.progress.emit(65)

                # ── Clustering → segmentos ───────────────────────────────
                self.log.emit("🗂️ Agrupando segmentos por offset...")
                segments = cluster_offsets(window_results)
                if self._cancelled: return

                if not segments:
                    self.error.emit(
                        "No se encontraron segmentos fiables.\n"
                        "Comprueba que los videos comparten contenido."
                    )
                    return

                self.log.emit(f"   ↳ Segmentos detectados: {len(segments)}")
                for i, (t0, t1, off) in enumerate(segments, 1):
                    self.log.emit(
                        f"      Segmento {i}: {t0:.1f}s – {t1:.1f}s  →  offset {off:+.3f}s"
                    )
                self.progress.emit(75)

                # ── Aplicar offset por segmento ──────────────────────────
                fallback = segments[0][2]
                self.log.emit("✏️ Aplicando sincronización a los subtítulos...")
                self.progress.emit(78)
                subs = apply_segmented(self.sub_path, segments, fallback)
                if self._cancelled: return

                # ── Keyframes: cargar o extraer ──────────────────────────
                kf_old, _ts_old = self._resolver_keyframes(
                    self.old_video, self.kf_old_path, "antiguo", 80, 83
                )
                kf_new, ts_new = self._resolver_keyframes(
                    self.new_video, self.kf_new_path, "nuevo", 87, 90
                )

                if kf_old and kf_new:
                    self.log.emit(
                        f"   ↳ Keyframes: {len(kf_old)} (antiguo) / {len(kf_new)} (nuevo)"
                    )
                    self.log.emit(f"   ↳ Umbral de snap: ±{KEYFRAME_SNAP_MS}ms")
                    fallback_frame_ms = 42
                    if not ts_new:
                        fps_num, fps_den = get_video_fps(self.new_video)
                        if fps_num > 0:
                            fallback_frame_ms = round(1000 * fps_den / fps_num)
                    n_starts, n_ends, snap_details = snap_to_keyframes(
                        subs, kf_old, kf_new,
                        frame_timestamps_new=ts_new,
                        fallback_frame_ms=fallback_frame_ms,
                    )
                    self.log.emit(
                        f"   ↳ Snap aplicado — starts: {n_starts} · ends: {n_ends}"
                    )
                    for d in snap_details:
                        self.log.emit(d)
                else:
                    self.log.emit("   ↳ Sin keyframes disponibles, snap omitido.")

                self.progress.emit(96)
                subs.save(self.output_path)
                if self._cancelled: return

                self.progress.emit(100)
                self.log.emit(f"✅ ¡Listo! Subtítulo guardado en:\n   {self.output_path}")
                self.finished.emit(self.output_path)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"{type(e).__name__}: {e}\n\n{tb}")


# ─────────────────────────────────────────────
#  DIRECTORIOS PERSISTENTES POR CAMPO
# ─────────────────────────────────────────────

_DIRS_PATH = Path.home() / ".sincronizador_dirs.json"


def _load_dirs() -> dict:
    try:
        if _DIRS_PATH.exists():
            return json.loads(_DIRS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_dirs(dirs: dict):
    try:
        _DIRS_PATH.write_text(
            json.dumps(dirs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass


def get_dir(key: str) -> str:
    return _load_dirs().get(key, str(Path.home()))


def set_dir(key: str, path: str):
    dirs = _load_dirs()
    dirs[key] = str(Path(path).parent)
    _save_dirs(dirs)


# ─────────────────────────────────────────────
#  ESTILO (Catppuccin Mocha)
# ─────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #cba6f7;
    padding: 8px 0px;
}

QLabel#section {
    font-size: 11px;
    color: #a6adc8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QLabel#hint {
    font-size: 11px;
    color: #6c7086;
    font-style: italic;
}

QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
}

QLineEdit:focus {
    border: 1px solid #cba6f7;
}

QPushButton#browse {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    color: #cdd6f4;
    min-width: 72px;
}

QPushButton#browse:hover {
    background-color: #45475a;
    border-color: #cba6f7;
}

QPushButton#sync {
    background-color: #cba6f7;
    border: none;
    border-radius: 8px;
    padding: 10px 30px;
    color: #1e1e2e;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#sync:hover {
    background-color: #d4b8f8;
}

QPushButton#sync:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#cancel {
    background-color: transparent;
    border: 1px solid #f38ba8;
    border-radius: 8px;
    padding: 10px 20px;
    color: #f38ba8;
    font-size: 13px;
}

QPushButton#cancel:hover {
    background-color: #2a1e24;
}

QPushButton#cancel:disabled {
    border-color: #45475a;
    color: #6c7086;
}

QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #cba6f7);
    border-radius: 5px;
}

QTextEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 8px;
    color: #a6e3a1;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}

QFrame#separator {
    background-color: #313244;
    max-height: 1px;
}
"""


# ─────────────────────────────────────────────
#  WIDGETS
# ─────────────────────────────────────────────

class DropLineEdit(QLineEdit):
    """QLineEdit con soporte de drag & drop para archivos."""

    _NORMAL_STYLE  = ""
    _HOVER_STYLE   = "border: 1px solid #cba6f7; background-color: #3d3d5c;"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._HOVER_STYLE)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._NORMAL_STYLE)   # restaura sin pisar el STYLE global

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._NORMAL_STYLE)
        urls = event.mimeData().urls()
        if urls:
            self.setText(urls[0].toLocalFile())
            event.acceptProposedAction()


class FileRow(QWidget):
    def __init__(self, label: str, key: str, save_mode: bool = False,
                 filters: str = "", optional: bool = False):
        super().__init__()
        self.save_mode = save_mode
        self.filters   = filters
        self.key       = key
        self.optional  = optional

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setObjectName("section" if not optional else "hint")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.entry = DropLineEdit()
        self.entry.setPlaceholderText(
            "Opcional — dejar vacío para detectar automáticamente"
            if optional else
            ("Selecciona un archivo o arrastra aquí..."
             if not save_mode else
             "Selecciona dónde guardar...")
        )
        row.addWidget(self.entry)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        btn = QPushButton("Buscar" if not save_mode else "Guardar como")
        btn.setObjectName("browse")
        btn.clicked.connect(self._browse)
        btn_row.addWidget(btn)

        if optional:
            btn_clear = QPushButton("✕")
            btn_clear.setObjectName("browse")
            btn_clear.setFixedWidth(28)
            btn_clear.setToolTip("Limpiar")
            btn_clear.clicked.connect(lambda: self.entry.clear())
            btn_row.addWidget(btn_clear)

        row.addLayout(btn_row)

        layout.addLayout(row)

    def _browse(self):
        start_dir = get_dir(self.key)
        if self.save_mode:
            path, _ = QFileDialog.getSaveFileName(self, directory=start_dir, filter=self.filters)
        else:
            path, _ = QFileDialog.getOpenFileName(self, directory=start_dir, filter=self.filters)
        if path:
            self.entry.setText(path)
            set_dir(self.key, path)

    def set(self, path: str):
        self.entry.setText(path)

    def get(self) -> str:
        return self.entry.text().strip()


# ─────────────────────────────────────────────
#  VENTANA PRINCIPAL
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SincroNyaa")
        self.setMinimumSize(700, 660)
        self.worker: SyncWorker | None = None
        self._build_ui()
        self.setStyleSheet(STYLE)
        self._check_ffmpeg()

    # ── Verificación de dependencias ─────────────────────────────────────

    def _check_ffmpeg(self):
        if not check_ffmpeg():
            QMessageBox.critical(
                self,
                "ffmpeg no encontrado",
                "SincroNyaa necesita ffmpeg para funcionar.\n\n"
                "Instálalo y asegúrate de que esté en el PATH del sistema."
            )

    # ── Construcción de UI ───────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(14)

        # Título
        header = QHBoxLayout()
        title = QLabel("🎬 SincroNyaa")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        sep = QFrame(); sep.setObjectName("separator")
        root.addWidget(sep)

        # File rows
        self.row_old = FileRow(
            "Video antiguo  (al que estaban sincronizados los subs)",
            key="old_video",
            filters="Video (*.mp4 *.mkv *.avi *.mov *.ts);;Todos (*)",
        )
        self.row_new = FileRow(
            "Video nuevo  (destino de la sincronización)",
            key="new_video",
            filters="Video (*.mp4 *.mkv *.avi *.mov *.ts);;Todos (*)",
        )
        self.row_subs = FileRow(
            "Subtítulo original",
            key="sub_path",
            filters="Subtítulos (*.ass *.ssa *.srt);;Todos (*)",
        )
        self.row_output = FileRow(
            "Subtítulo sincronizado  (salida)",
            key="output_path",
            save_mode=True,
            filters="ASS (*.ass);;SRT (*.srt);;Todos (*)",
        )

        self.row_kf_old = FileRow(
            "Keyframes del video antiguo  (opcional — cargar para más precisión)",
            key="kf_old",
            filters="Keyframes (*.txt *.log);;Todos (*)",
            optional=True,
        )
        self.row_kf_new = FileRow(
            "Keyframes del video nuevo  (opcional — cargar para más precisión)",
            key="kf_new",
            filters="Keyframes (*.txt *.log);;Todos (*)",
            optional=True,
        )

        # Auto-sugerir output al elegir subtítulo de entrada
        self.row_subs.entry.textChanged.connect(self._suggest_output)

        for row in (self.row_old, self.row_new, self.row_subs, self.row_output,
                    self.row_kf_old, self.row_kf_new):
            root.addWidget(row)

        sep2 = QFrame(); sep2.setObjectName("separator")
        root.addWidget(sep2)

        # Botones
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setObjectName("cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(self.btn_cancel)

        self.btn_sync = QPushButton("⚡  Sincronizar")
        self.btn_sync.setObjectName("sync")
        self.btn_sync.clicked.connect(self._start)
        btn_row.addWidget(self.btn_sync)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # Progreso
        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        # Log
        log_lbl = QLabel("Log de proceso")
        log_lbl.setObjectName("section")
        root.addWidget(log_lbl)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        root.addWidget(self.log_box)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _suggest_output(self, sub_path: str):
        """Rellena el campo de salida automáticamente si está vacío."""
        if not sub_path or self.row_output.get():
            return
        p = Path(sub_path)
        suggestion = str(p.with_name(p.stem + "_synced" + p.suffix))
        self.row_output.set(suggestion)

    # ── Control del worker ───────────────────────────────────────────────

    def _start(self):
        old  = self.row_old.get()
        new  = self.row_new.get()
        subs = self.row_subs.get()
        out  = self.row_output.get()

        if not all([old, new, subs, out]):
            QMessageBox.warning(
                self, "Faltan archivos",
                "Debes seleccionar todos los archivos antes de continuar."
            )
            return

        if not Path(old).is_file():
            QMessageBox.warning(
                self, "Archivo inválido",
                f"El video antiguo no es un archivo válido:\n{old}"
            )
            return
        if not Path(new).is_file():
            QMessageBox.warning(
                self, "Archivo inválido",
                f"El video nuevo no es un archivo válido:\n{new}"
            )
            return
        if not Path(subs).is_file():
            QMessageBox.warning(
                self, "Archivo inválido",
                f"El subtítulo original no es un archivo válido:\n{subs}"
            )
            return
        if not Path(out).parent.is_dir():
            QMessageBox.warning(
                self, "Ruta de salida inválida",
                f"El directorio de salida no existe:\n{Path(out).parent}"
            )
            return

        self.log_box.clear()
        self.progress.setValue(0)
        self.btn_sync.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        kf_old = self.row_kf_old.get()
        kf_new = self.row_kf_new.get()
        self.worker = SyncWorker(old, new, subs, out, kf_old, kf_new)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._append_log("⏹ Sincronización cancelada por el usuario.")
            self.progress.setValue(0)
        self.btn_sync.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _append_log(self, msg: str):
        self.log_box.append(msg)

    def _on_done(self, path: str):
        self.btn_sync.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.information(
            self, "¡Listo!",
            f"Subtítulo sincronizado guardado en:\n{path}"
        )

    def _on_error(self, msg: str):
        self.btn_sync.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self._append_log(f"❌ Error: {msg}")
        QMessageBox.critical(self, "Error", msg)


# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()