- Hacer que en el subtítulo sincronizado se actualicen los metadatos del .ass cambiando la fuente del video anterior a la del video nuevo.

En la sección `[Aegisub Project Garbage]`, lo que se actualizaría serían las partes de `Audio File:` y `Video File:`. Y no sé si haga falta cambiar algo más de la sección.

- Usar scenedetect para reemplazar la detección con ffmpeg.

Actualmente la función `detect_scenes(...)` depende de strings tipo `"pts_time:"`

  Con scenedetect:
```
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list()

    times = [scene[0].get_seconds() for scene in scenes]
    return times
```  

- ¿Serviría usar matplotlib para el offset de audio?

- ¿Serviría usar rapidfuzz para mejorar matching de escenas?

Actualmente se hace: `closest = min(scenes_new, key=lambda s: abs(s - target))`

  Mejor se puede hacer algo tipo:

```
from rapidfuzz import process

# conceptual: matching por distancia mínima global
```

O mejor aún:
👉 Hungarian algorithm (scipy)

```
from scipy.optimize import linear_sum_assignment
```
