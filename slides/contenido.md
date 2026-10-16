# ¿Cómo enseñarle a una IA a escuchar?

Iteración enfocada en 35 minutos de exposición y 10 minutos de preguntas. El PPTX usa layouts y placeholders estándar, sin tema gráfico fijado.

## 1. ¿Cómo enseñarle a una IA a escuchar? (Apertura)

Exposición: 35 min · Preguntas: 10 min
RAG sobre video: transcripción, búsqueda semántica y respuestas con minuto citado.

## 2. 0–3 · Una pregunta y un minuto exacto (0–3 · Gancho)

- Demo en vivo: preguntar sobre el video de Saturno.
- Ver la respuesta y los fragmentos que la sustentan.
- Elegir una cita y saltar al momento del video.
- Pregunta: ¿qué acaban de descubrir en Saturno?

## 3. 3–7 · ¿Qué cambia cuando añadimos RAG? (3–7 · El problema)

**Columna izquierda**
- SIN RAG
- El modelo responde desde sus parámetros y conocimientos previos.
- Una noticia reciente puede ser desconocida o dar pie a una invención.
- No hay vínculo verificable con el video.

**Columna derecha**
- CON RAG
- Buscamos fragmentos de la transcripción antes de generar.
- El modelo recibe evidencia concreta y cita los fragmentos [n].
- Si el video no lo menciona, la respuesta debe decirlo.

## 4. 7–10 · Arquitectura: dos recorridos (7–10 · Arquitectura)

**Columna izquierda**
- INGESTA · preparar el video
- MP4 → Whisper → segmentos con marcas de tiempo.
- Segmentos → chunks → embeddings → Qdrant.
- La transcripción queda en caché para reutilizarla.

**Columna derecha**
- CONSULTA · responder la pregunta
- Pregunta → embedding → búsqueda top-k en Qdrant.
- Qwen recibe pregunta y fragmentos recuperados.
- Respuesta con citas → reproductor en el minuto citado.
- Mapa interactivo Archify: .archify/architecture-rag-conferencia-20261013-223433/arquitectura-rag.html

## 5. 10–14 · Whisper convierte audio en evidencia temporal (10–14 · Whisper)

- Entrada: video MP4. Salida: segmentos {start, end, text}.
- Las marcas de tiempo permiten regresar al segundo relevante.
- faster-whisper en Docker (CPU) o MLX nativo en Apple Silicon.
- En video.mp4 (10:16): 123,9 s faster-whisper; 17,9 s MLX con modelo en caché.
- La primera ejecución de MLX tardó 81,0 s e incluyó descarga del modelo.

## 6. 14–18 · Chunking: contexto sin perder el tiempo (14–18 · Chunking)

**Columna izquierda**
- SI SON MUY CORTOS
- Cada vector contiene poco contexto.
- Una idea puede quedar partida entre resultados.

**Columna derecha**
- SI SON MUY LARGOS
- Se mezclan temas y baja la precisión del fragmento recuperado.
- El prompt incluye texto irrelevante.
- Configuración de la demo: máximo 700 caracteres y solapamiento de 1 segmento.
- Cada chunk conserva el inicio y final para citar el minuto.

## 7. 18–22 · Embeddings: significado como vector (18–22 · Embeddings)

- bge-m3 convierte cada fragmento y cada pregunta en un vector de 1024 dimensiones.
- Textos semánticamente parecidos quedan próximos en el espacio vectorial.
- La pregunta y los chunks deben usar el mismo modelo de embeddings.
- Un embedding representa texto; no es una respuesta ni una base de datos.

## 8. 22–26 · Búsqueda: NumPy explica, Qdrant escala (22–26 · Búsqueda)

**Columna izquierda**
- NUMPY · a mano
- Normalizamos vectores y calculamos producto punto.
- Ordenamos por similitud coseno y tomamos los mejores k.
- Sirve para ver qué operación hace la búsqueda.

**Columna derecha**
- QDRANT · servicio vectorial
- Guarda vectores y metadatos: texto, video y marcas de tiempo.
- Recupera los 4 fragmentos más similares en esta demo.
- La puntuación ordena resultados; no es una probabilidad de verdad.

## 9. 26–28 · Generación: qué recibe el modelo (26–28 · Prompt)

- Instrucción: responde solo con la información de los fragmentos.
- Cita los fragmentos [n] y reconoce cuando falta evidencia.
- Prompt = instrucción + fragmentos recuperados + pregunta; modelo: Qwen3.8 27B.

## 10. 28–30 · Una respuesta que vuelve al video (28–30 · Citas)

- Los fragmentos conservan texto, video, puntuación y tiempo.
- La cita [n] lleva al momento correspondiente del reproductor.
- Medimos tiempo total, tokens de salida y tokens/s promedio (incluye espera inicial).

## 11. 30–32 · ¿Cuándo falla el RAG? (30–32 · Límites)

**Columna izquierda**
- RECUPERACIÓN
- Nombres propios o cifras pueden no recuperarse bien.
- Un chunk mal cortado puede perder contexto.
- Similitud alta no garantiza una respuesta.

**Columna derecha**
- GENERACIÓN
- La pregunta puede estar fuera del video.
- El modelo aún puede inferir de más.
- Inspeccionamos los fragmentos antes de confiar.

## 12. 32–33 · Mejoras para explorar (32–33 · Mejoras)

- Híbrida: significado + palabras exactas.
- Reranking: reordena candidatos; HyDE ayuda a formular la búsqueda.
- Son mejoras para probar, no evidencia nueva.

## 13. 33–35 · Local, nube y frameworks (33–35 · Ecosistema)

**Columna izquierda**
- ESTA DEMO
- Ollama local para embeddings y chat.
- Qdrant y Gradio en Docker.
- Proveedores intercambiables por base URL, modelo y clave en .env.

**Columna derecha**
- OTRAS OPCIONES
- OpenAI-compatible permite cambiar a un proveedor en la nube.
- LangChain y LlamaIndex ofrecen componentes y abstracciones para RAG.
- Aquí escribimos el pipeline para hacer visibles sus etapas.

## 14. 35–45 · Cierre y preguntas (35–45 · Preguntas)

- Cierre (35–36): una respuesta útil recupera evidencia y conserva su origen.
- Repo: transcripción → chunks → embeddings → búsqueda → generación.
- Después: retos en docs/retos.md.
- Preguntas del público (36–45).
