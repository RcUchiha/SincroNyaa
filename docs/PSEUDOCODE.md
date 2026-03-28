# SincroNyaa — Seudocódigo

---

## 1. Seudocódigo clásico (estilo algoritmo)

```
PROGRAMA SincroNyaa

ENTRADA:
  video_anterior  → archivo de video original
  video_nuevo     → archivo de video nuevo
  subtitulo       → archivo .ass / .srt original
  salida          → ruta donde guardar el subtítulo sincronizado

─────────────────────────────────────────
FASE 1: EXTRACCIÓN DE AUDIO
─────────────────────────────────────────
audio_ant ← ffmpeg(video_anterior, mono, 16kHz)
audio_nue ← ffmpeg(video_nuevo,    mono, 16kHz)

─────────────────────────────────────────
FASE 2: ANÁLISIS DE CARACTERÍSTICAS
─────────────────────────────────────────
FUNCIÓN compute_features(audio):
  mel    ← mel_spectrogram(audio)       // energía espectral
  chroma ← chroma_stft(audio)           // información tonal
  onset  ← onset_strength(audio)        // fuerza de ataques sonoros

  mel    ← normalizar(mel)
  chroma ← normalizar(chroma)
  onset  ← normalizar(onset)

  // Pesos: onset y mel dominan (40% c/u), chroma apoya (20%)
  RETORNAR mel * 0.4 + chroma * 0.2 + onset * 0.4

feat_ant ← compute_features(audio_ant)
feat_nue ← compute_features(audio_nue)

─────────────────────────────────────────
FASE 3: CORRELACIÓN CRUZADA FFT
─────────────────────────────────────────
FUNCIÓN find_offset(feat_ant, feat_nue):
  corr ← fftconvolve(feat_nue, invertir(feat_ant))  // correlación cruzada

  candidatos ← []
  PARA CADA pico en top_picos(corr):
    SI distancia(pico, candidatos existentes) > 50 frames:
      agregar pico a candidatos
    SI len(candidatos) == 5: PARAR

  // Convertir lags a segundos (hop=512, sr=16000)
  RETORNAR [(lag * 512/16000, score) PARA CADA candidato]

candidatos ← find_offset(feat_ant, feat_nue)

─────────────────────────────────────────
FASE 4: DETECCIÓN DE ESCENAS
─────────────────────────────────────────
FUNCIÓN detect_scenes(video, umbral=0.35):
  cortes ← ffmpeg select=gt(scene, umbral)
  RETORNAR filtrar_cortes_cercanos(cortes, min_gap=1.0s)

escenas_ant ← detect_scenes(video_anterior)
escenas_nue ← detect_scenes(video_nuevo)

─────────────────────────────────────────
FASE 5: VOTACIÓN DEL OFFSET FINAL
─────────────────────────────────────────
FUNCIÓN vote_offset(candidatos, escenas_ant, escenas_nue):
  mejor_offset ← candidatos[0]
  mejor_score  ← -∞

  PARA CADA offset EN candidatos:
    score ← 0
    PARA CADA corte EN escenas_ant:
      target   ← corte + offset
      dist     ← min(|corte_nue - target| PARA corte_nue EN escenas_nue)
      score    ← score - dist    // menor distancia = mejor alineación
    SI score > mejor_score:
      mejor_score  ← score
      mejor_offset ← offset

  RETORNAR mejor_offset

offset_final ← vote_offset(candidatos, escenas_ant, escenas_nue)

─────────────────────────────────────────
FASE 6: APLICACIÓN AL SUBTÍTULO
─────────────────────────────────────────
FUNCIÓN apply(subtitulo, escenas_ant, escenas_nue, offset):
  pares ← emparejar_escenas(escenas_ant, escenas_nue, offset)

  PARA CADA linea EN subtitulo:
    new_start ← linea.start + offset

    // Ajuste suave por escena cercana
    (corte_ant, corte_nue) ← par_mas_cercano(pares, linea.start)
    dist ← |linea.start - corte_ant|
    SI dist < 0.6s:
      delta     ← corte_nue - (corte_ant + offset)
      new_start ← new_start + delta * 0.5   // atracción parcial al corte

    linea.start ← new_start
    linea.end   ← new_start + duracion_original

  GUARDAR subtitulo en salida

apply(subtitulo, escenas_ant, escenas_nue, offset_final)
FIN
```

---

## 2. Lenguaje natural paso a paso

**Paso 1 — Extraer audio**
Se usa ffmpeg para extraer el audio de ambos videos como WAV mono a 16 kHz. Mono y 16 kHz son suficientes para el análisis y reducen el tiempo de procesamiento.

**Paso 2 — Calcular características de audio**
Para cada audio se calculan tres descriptores:
- *Mel-spectrogram*: cómo se distribuye la energía a lo largo del tiempo en distintas bandas de frecuencia.
- *Chroma*: qué notas o tonos predominan en cada instante. Es robusto ante diferencias de calidad de audio entre versiones.
- *Onset strength*: qué tan fuerte es cada "ataque" sonoro (inicio de diálogo, efecto de sonido, golpe). Es el descriptor más útil para detectar puntos de anclaje temporales.

Los tres se normalizan y se combinan en un único vector ponderado, dando más peso a los descriptores más discriminativos.

**Paso 3 — Correlación cruzada FFT**
Se compara el vector del audio anterior contra el del nuevo usando correlación cruzada por FFT. Esto produce una curva de similitud donde los picos indican posibles offsets. En lugar de quedarse solo con el mejor pico, se recogen los 5 mejores candidatos asegurando que estén suficientemente separados entre sí, para evitar que todos apunten al mismo lugar.

**Paso 4 — Detectar cortes de escena**
ffmpeg analiza ambos videos fotograma a fotograma y detecta los momentos donde la imagen cambia bruscamente. Se filtran los cortes demasiado seguidos para quedarse solo con los significativos.

**Paso 5 — Votar el offset final**
Cada uno de los 5 candidatos se prueba contra las escenas detectadas: se calcula qué tan bien alinea los cortes del video anterior con los del nuevo. El candidato que produce la mejor alineación visual gana. Esto actúa como segunda opinión sobre el análisis de audio.

**Paso 6 — Aplicar el offset al subtítulo**
Cada línea del subtítulo se desplaza por el offset final. Adicionalmente, si una línea está cerca de un corte de escena, recibe un pequeño ajuste extra que la "atrae" hacia ese corte, mejorando la precisión justo en los momentos de cambio de escena donde el timing es más crítico.

---

## 3. Diagrama de flujo en texto

```
┌─────────────────────────────────────────────────────┐
│                    SINCRONYAA                       │
│              Inicio del proceso                     │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │   Extraer audio       │
           │  ffmpeg → WAV mono    │
           │  16kHz, ambos videos  │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Calcular features    │
           │  ┌─────────────────┐  │
           │  │ mel-spectrogram │  │  → energía espectral
           │  │ chroma          │  │  → información tonal
           │  │ onset strength  │  │  → ataques sonoros
           │  └────────┬────────┘  │
           │     normalizar        │
           │     ponderar y        │
           │     combinar          │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Correlación cruzada  │
           │       FFT             │
           │                       │
           │  feat_ant ──┐         │
           │  feat_nue ──┼→ corr   │
           │             │         │
           │  top 5 picos          │  → 5 offsets candidatos
           │  (no máx supresión)   │    separados entre sí
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Detectar escenas     │
           │  ffmpeg scene filter  │
           │  umbral = 0.35        │
           │  filtrar gap < 1.0s   │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │   Votar offset final  │
           │                       │
           │  Para cada candidato: │
           │    medir distancia    │
           │    a escenas nuevas   │
           │                       │
           │  ┌──────────────────┐ │
           │  │ mejor alineación │ │  → offset_final
           │  └──────────────────┘ │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Aplicar al subtítulo │
           │                       │
           │  Para cada línea:     │
           │    start += offset    │
           │                       │
           │    ¿cerca de escena?  │
           │    ┌────────────────┐ │
           │    │ Sí → ajuste    │ │  → atracción suave
           │    │     suave      │ │     al corte (50%)
           │    │ No → sin extra │ │
           │    └────────────────┘ │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │   Guardar subtítulo   │
           │   sincronizado (.ass  │
           │   o .srt)             │
           └───────────────────────┘
```
