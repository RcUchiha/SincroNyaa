# SincroNyaa

¿Tienes subtítulos que van desincronizados porque encontraste una versión diferente del video? SincroNyaa lo resuelve automáticamente. Carga el video anterior, el nuevo y el subtítulo original, y SincroNyaa calculará el desfase y generará un subtítulo sincronizado listo para usar. Sin configuración manual, sin ajustar tiempos a mano.

---

<img width="688" height="735" alt="image" src="https://github.com/user-attachments/assets/2efb96c9-6373-450e-9741-40a6ced2f45e" />

---

## Instalación

### Requisitos del sistema

- Python 3.11 o superior (el piso real lo fija `scipy`, que desde la 1.17.1 requiere `>=3.11` — confirmado contra PyPI, no asumido)
- [ffmpeg](https://ffmpeg.org/download.html) instalado y disponible en el PATH del sistema

### Dependencias de Python

Las dependencias están declaradas en [`pyproject.toml`](pyproject.toml):

```bash
pip install .
```

### Dependencia opcional: VapourSynth (más precisión en el snap a keyframes)

SincroNyaa puede usar [VapourSynth](https://www.vapoursynth.com/) para leer los timestamps reales de cada frame del video, en vez de asumir un framerate constante. Esto importa sobre todo en videos con fps variable (por ejemplo, contenido con IVTC/decimation aplicado), donde asumir un fps fijo puede desviar ligeramente el punto exacto de anclaje al hacer snap a keyframes.

**Si no la instalás, no pasa nada**: SincroNyaa cae automáticamente a detección de keyframes por `ffprobe` (sin timestamps reales de frame). Es un fallback silencioso esperado, no un bug — el programa funciona igual, solo con un poco menos de precisión en ese paso puntual.

Si la querés instalar:

```bash
pip install .[vapoursynth]
```

Esto instala el paquete `vapoursynth` de PyPI (que desde la versión R74 de VapourSynth incluye el core, no solo un binding). Hace falta un paso extra después, fuera de pip:

```bash
vapoursynth config
```

Además, este extra **no instala los plugins de source filter** que SincroNyaa necesita (`lsmas`/L-SMASH-Works y `ffms2`) — esos no se distribuyen por pip. Se instalan aparte con [VSRepo](https://www.vapoursynth.com/doc/installation.html) (`pip install vsrepo`, luego `vsrepo install lsmas ffms2`) o manualmente. Ver la [documentación oficial de instalación de VapourSynth](https://www.vapoursynth.com/doc/installation.html) para el detalle completo.

### Ejecutar

```bash
python sincronyaa.py
```

---

## ¿Cómo funciona?

SincroNyaa combina análisis de audio y detección visual de escenas para calcular el desfase con la mayor precisión posible.

### 1. Extracción de audio

Usando `ffmpeg`, extrae el audio de ambos videos en formato WAV mono a 16 kHz, que es suficiente para el análisis y reduce el tiempo de procesamiento.

### 2. Análisis de características de audio

En lugar de comparar el audio directamente, SincroNyaa extrae tres tipos de características y las combina en un único vector:

- **Mel-spectrogram** — representa la energía espectral del audio, útil para identificar la textura sonora general.
- **Chroma** — captura la información tonal, robusta frente a cambios de codificación o calidad de audio.
- **Onset strength** — detecta la fuerza de los ataques sonoros (cortes, golpes, inicio de diálogos), muy discriminativo para encontrar puntos de anclaje.

Cada canal se normaliza por separado antes de combinarse, dándole más peso a onset y mel-spectrogram (40% cada uno) y menos a chroma (20%).

### 3. Correlación cruzada FFT

Se aplica correlación cruzada sobre el vector combinado usando transformada de Fourier rápida (`fftconvolve`), lo que permite comparar eficientemente señales largas. En lugar de quedarse con el único mejor resultado, se calculan los **5 mejores offsets candidatos**, suprimiendo picos demasiado cercanos entre sí para maximizar la diversidad de candidatos.

### 4. Detección de cortes de escena

`ffmpeg` detecta los cortes visuales en ambos videos usando un umbral de diferencia de fotogramas. Los cortes demasiado cercanos entre sí se filtran para evitar ruido.

### 5. Votación del offset final

Cada offset candidato se evalúa contra los cortes de escena detectados: se elige el que mejor alinea los cortes del video anterior con los del nuevo. Esto añade una capa de validación visual sobre el análisis puramente auditivo, aumentando la robustez ante videos con calidades de audio muy distintas.

### 6. Aplicación al subtítulo

Con el offset final determinado, se recorre cada línea del subtítulo y se le aplica el desplazamiento. Adicionalmente, las líneas cercanas a un corte de escena reciben un ajuste suave que las "atrae" hacia el corte más próximo, mejorando la precisión en los momentos donde el timing importa más.

---

## Licencia

MIT © [CiferrC](https://github.com/CiferrC)
