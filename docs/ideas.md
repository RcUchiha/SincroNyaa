- [Implementado] Hacer que en el subtítulo sincronizado se actualicen los metadatos del .ass cambiando la fuente del video anterior a la del video nuevo.

`SyncWorker.run()` actualiza `subs.aegisub_project["Video File"]` y `subs.aegisub_project["Audio File"]` al `self.new_video`, justo antes de `subs.save(...)` — se agregan aunque el .ass original no tuviera la sección `[Aegisub Project Garbage]`. No se toca ningún otro campo de esa sección (Video Position, Active Line, Zoom, etc.).

- Herramienta de diagnóstico opcional (matplotlib) para visualizar la sincronización de un episodio.

Graficar la salida de `find_offsets_by_windows()` (offset por ventana a lo largo del tiempo) superpuesta con los segmentos que arma `cluster_offsets()`. Útil para depurar a ojo casos donde la sincronización de un episodio sale rara, sobre todo en las zonas de OP/ED, donde el offset puede saltar o la confianza de las ventanas caer. No sería parte del flujo normal del programa — una utilidad aparte (por ejemplo, un script en `scripts/` que reciba los mismos dos videos y muestre el gráfico, sin tocar subtítulos).
