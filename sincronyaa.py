__author__  = "CiferrC"
__license__ = "MIT"
__version__ = "0.0.1"

import sys
import os
import subprocess
import tempfile
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


# ─────────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────────

def extract_audio(video_path: str, wav_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def compute_chroma_and_onset(audio_path: str):
    """
    Retorna un vector de características combinando:
    - Mel-spectrogram (energía espectral)
    - Chroma (información tonal)
    - Onset strength (ritmo/cortes)
    Esto da mayor robustez que solo usar mel-spectrogram.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    hop = 512

    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.mean(mel_db, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    chroma_feat = np.mean(chroma, axis=0)

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # Normalizar cada canal por separado
    def norm(v):
        std = np.std(v)
        return (v - np.mean(v)) / (std if std > 0 else 1.0)

    # Combinar: damos más peso a onset (muy discriminativo en cortes)
    min_len = min(len(mel_feat), len(chroma_feat), len(onset))
    combined = (
        norm(mel_feat[:min_len]) * 0.4 +
        norm(chroma_feat[:min_len]) * 0.2 +
        norm(onset[:min_len]) * 0.4
    )
    return combined, hop, sr


def find_offset_precise(feat_old, feat_new, hop_length=512, sr=16000):
    """
    Calcula el offset usando correlación cruzada FFT sobre el vector combinado.
    Devuelve el offset en segundos y los N mejores candidatos para validación.
    """
    a = feat_old - np.mean(feat_old)
    b = feat_new - np.mean(feat_new)

    corr = fftconvolve(b, a[::-1], mode="full")
    lags = np.arange(-(len(a) - 1), len(b))

    frame_duration = hop_length / sr

    # Top-5 candidatos (para votación posterior con escenas)
    top_k = 5
    peaks = np.argsort(corr)[::-1][:top_k * 10]
    # Suprimir picos muy cercanos (non-maximum suppression simple)
    candidates = []
    for idx in peaks:
        lag = lags[idx]
        if all(abs(lag - c[0]) > 50 for c in candidates):
            candidates.append((lag, corr[idx]))
        if len(candidates) == top_k:
            break

    offsets = [(lag * frame_duration, score) for lag, score in candidates]
    best_offset = offsets[0][0] if offsets else 0.0
    return best_offset, offsets


# ─────────────────────────────────────────────
#  ESCENAS
# ─────────────────────────────────────────────

def detect_scenes(video_path: str, threshold=0.35):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    times = []
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            try:
                t = float(line.split("pts_time:")[1].split()[0])
                times.append(t)
            except Exception:
                pass
    return times


def filter_scenes(scenes, min_gap=1.0):
    if not scenes:
        return scenes
    filtered = [scenes[0]]
    for s in scenes[1:]:
        if s - filtered[-1] > min_gap:
            filtered.append(s)
    return filtered


def vote_offset_with_scenes(candidates, scenes_old, scenes_new):
    """
    Vota entre los offsets candidatos usando las escenas detectadas.
    El candidato que mejor alinea las escenas del video viejo con las del nuevo gana.
    """
    if not scenes_old or not scenes_new:
        return candidates[0][0]

    best_offset = candidates[0][0]
    best_score = -np.inf

    for offset, _ in candidates:
        score = 0
        for s_old in scenes_old:
            target = s_old + offset
            dist = min(abs(s_new - target) for s_new in scenes_new)
            score -= dist  # menor distancia = mejor
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def match_scenes(scenes_old, scenes_new, offset):
    pairs = []
    for s_old in scenes_old:
        target = s_old + offset
        closest = min(scenes_new, key=lambda s: abs(s - target))
        pairs.append((s_old, closest))
    return pairs


# ─────────────────────────────────────────────
#  APLICAR
# ─────────────────────────────────────────────

def apply_scene_based(sub_path, scene_pairs, offset, output_path):
    subs = pysubs2.load(sub_path)

    SCENE_PULL_THRESHOLD = 0.6
    SCENE_PULL_STRENGTH  = 0.5

    for line in subs:
        start = line.start / 1000.0
        end   = line.end   / 1000.0
        duration = end - start

        new_start = start + offset

        if scene_pairs:
            closest_old, closest_new = min(scene_pairs, key=lambda p: abs(p[0] - start))
            dist = abs(start - closest_old)
            if dist < SCENE_PULL_THRESHOLD:
                delta = closest_new - (closest_old + offset)
                new_start += delta * SCENE_PULL_STRENGTH

        new_end = new_start + duration
        line.start = max(0, int(new_start * 1000))
        line.end   = max(0, int(new_end   * 1000))

    subs.save(output_path)


# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────

class SyncWorker(QThread):
    progress   = pyqtSignal(int)
    log        = pyqtSignal(str)
    finished   = pyqtSignal(str)   # output path
    error      = pyqtSignal(str)

    def __init__(self, old_video, new_video, sub_path, output_path):
        super().__init__()
        self.old_video   = old_video
        self.new_video   = new_video
        self.sub_path    = sub_path
        self.output_path = output_path

    def run(self):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                old_wav = os.path.join(tmp, "old.wav")
                new_wav = os.path.join(tmp, "new.wav")

                self.log.emit("🎬 Extrayendo audio del video original...")
                self.progress.emit(5)
                extract_audio(self.old_video, old_wav)

                self.log.emit("🎬 Extrayendo audio del video nuevo...")
                self.progress.emit(15)
                extract_audio(self.new_video, new_wav)

                self.log.emit("🔍 Analizando características de audio (mel + chroma + onset)...")
                self.progress.emit(30)
                feat_old, hop, sr = compute_chroma_and_onset(old_wav)
                feat_new, _,   _  = compute_chroma_and_onset(new_wav)

                self.log.emit("📐 Calculando offset por correlación cruzada FFT...")
                self.progress.emit(45)
                raw_offset, candidates = find_offset_precise(feat_old, feat_new, hop, sr)
                self.log.emit(f"   ↳ Offset inicial (audio): {raw_offset:+.3f}s")
                self.log.emit(f"   ↳ Top candidatos: {[f'{o:+.3f}s' for o, _ in candidates]}")

                self.log.emit("🎞️ Detectando cortes de escena en video original...")
                self.progress.emit(60)
                scenes_old = filter_scenes(detect_scenes(self.old_video))

                self.log.emit("🎞️ Detectando cortes de escena en video nuevo...")
                self.progress.emit(72)
                scenes_new = filter_scenes(detect_scenes(self.new_video))
                self.log.emit(f"   ↳ Escenas encontradas: {len(scenes_old)} (orig) / {len(scenes_new)} (nuevo)")

                self.log.emit("🗳️ Votando offset final con escenas...")
                self.progress.emit(80)
                final_offset = vote_offset_with_scenes(candidates, scenes_old, scenes_new)
                self.log.emit(f"   ↳ Offset final: {final_offset:+.3f}s")

                self.log.emit("🔗 Emparejando escenas...")
                self.progress.emit(87)
                scene_pairs = match_scenes(scenes_old, scenes_new, final_offset)

                self.log.emit("✏️ Aplicando sincronización a los subtítulos...")
                self.progress.emit(94)
                apply_scene_based(self.sub_path, scene_pairs, final_offset, self.output_path)

                self.progress.emit(100)
                self.log.emit(f"✅ ¡Listo! Subtítulo guardado en:\n   {self.output_path}")
                self.finished.emit(self.output_path)

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
#  DIRECTORIOS PERSISTENTES POR CAMPO
# ─────────────────────────────────────────────

_DIRS_PATH = Path.home() / ".auto_subtitle_sync_dirs.json"


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
    """Devuelve el último directorio guardado para este campo."""
    return _load_dirs().get(key, str(Path.home()))


def set_dir(key: str, path: str):
    """Guarda el directorio del archivo elegido para este campo."""
    dirs = _load_dirs()
    dirs[key] = str(Path(path).parent)
    _save_dirs(dirs)


# ─────────────────────────────────────────────
#  GUI
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


class DropLineEdit(QLineEdit):
    """QLineEdit con soporte de drag & drop para archivos."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 1px solid #cba6f7; background-color: #3d3d5c;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)
            event.acceptProposedAction()


class FileRow(QWidget):
    def __init__(self, label: str, key: str, save_mode=False, filters=""):
        super().__init__()
        self.save_mode = save_mode
        self.filters   = filters
        self.key       = key          # identificador único para recordar su directorio

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setObjectName("section")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.entry = DropLineEdit()
        self.entry.setPlaceholderText(
            "Selecciona un archivo o arrastra aquí..."
            if not save_mode else
            "Selecciona dónde y cómo guardar..."
        )
        row.addWidget(self.entry)

        btn = QPushButton("Buscar" if not save_mode else "Guardar")
        btn.setObjectName("browse")
        btn.clicked.connect(self._browse)
        row.addWidget(btn)

        layout.addLayout(row)

    def _browse(self):
        start_dir = get_dir(self.key)   # abre en el último directorio de este campo
        if self.save_mode:
            path, _ = QFileDialog.getSaveFileName(self, directory=start_dir, filter=self.filters)
        else:
            path, _ = QFileDialog.getOpenFileName(self, directory=start_dir, filter=self.filters)
        if path:
            self.entry.setText(path)
            set_dir(self.key, path)     # guarda el directorio para la próxima vez

    def set(self, path: str):
        self.entry.setText(path)

    def get(self):
        return self.entry.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SincroNyaa")
        self.setMinimumSize(680, 620)
        self.worker = None
        self._build_ui()
        self.setStyleSheet(STYLE)

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
        self.row_old    = FileRow("Video anterior",
                                  key="old_video",
                                  filters="Video (*.mp4 *.mkv *.avi *.mov *.ts);;Todos (*)")
        self.row_new    = FileRow("Video nuevo (al que quieres sincronizar)",
                                  key="new_video",
                                  filters="Video (*.mp4 *.mkv *.avi *.mov *.ts);;Todos (*)")
        self.row_subs   = FileRow("Subtítulo original",
                                  key="sub_path",
                                  filters="Subtítulos (*.ass *.ssa *.srt);;Todos (*)")
        self.row_output = FileRow("Subtítulo sincronizado",
                                  key="output_path",
                                  save_mode=True,
                                  filters="ASS (*.ass);;SRT (*.srt);;Todos (*)")

        for row in (self.row_old, self.row_new, self.row_subs, self.row_output):
            root.addWidget(row)

        sep2 = QFrame(); sep2.setObjectName("separator")
        root.addWidget(sep2)

        # Botón
        self.btn_sync = QPushButton("⚡  Sincronizar")
        self.btn_sync.setObjectName("sync")
        self.btn_sync.clicked.connect(self._start)
        root.addWidget(self.btn_sync, alignment=Qt.AlignmentFlag.AlignHCenter)

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
        self.log_box.setMinimumHeight(140)
        root.addWidget(self.log_box)

    def _start(self):
        old   = self.row_old.get()
        new   = self.row_new.get()
        subs  = self.row_subs.get()
        out   = self.row_output.get()

        if not all([old, new, subs, out]):
            QMessageBox.warning(self, "Faltan archivos",
                                "Debes seleccionar todos los archivos antes de continuar.")
            return

        self.log_box.clear()
        self.progress.setValue(0)
        self.btn_sync.setEnabled(False)

        self.worker = SyncWorker(old, new, subs, out)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _append_log(self, msg: str):
        self.log_box.append(msg)

    def _on_done(self, path: str):
        self.btn_sync.setEnabled(True)
        QMessageBox.information(self, "¡Listo!",
                                f"Subtítulo sincronizado guardado en:\n{path}")

    def _on_error(self, msg: str):
        self.btn_sync.setEnabled(True)
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
