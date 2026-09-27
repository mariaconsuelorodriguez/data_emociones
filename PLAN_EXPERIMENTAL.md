# PLAN EXPERIMENTAL PROPUESTO
## Modelos unimodales de Deep Learning para reconocimiento de emociones (FER, EEG, EMG, PPG)

**Proyecto:** Modelo Multimodal para la identificación de emociones basado en la correlación entre FER, señales biofeedback (EEG, EMG, PPG) y expresión explícita — estudio de caso educativo.
**Etapa actual:** Diseño y validación de ramas unimodales (sin fusión).
**Estado:** Borrador para revisión — **pendiente de aprobación antes de generar código.**

---

## 0. Alcance y restricciones del entorno de trabajo

Antes de entrar al análisis, dos advertencias metodológicas que condicionan todo lo demás (regla científica del punto 21: no inventar resultados):

1. **(Nota: esta afirmación de la primera versión del plan quedó desactualizada — ver secciones 0bis/0ter/0quater.)** En efecto, la mayoría de los datasets (AffectNet, CK+, DEAP, DREAMER, MAHNOB-HCI, AMIGOS, WESAD, RAF-ML) siguen sin estar presentes y requieren descarga con EULA. Pero **SEED (EEG), Yale, RAF-DB (etiquetas) y PME4 (manifiesto)** ya tienen contenido real verificado en este repositorio, como se detalla más abajo. El código seguirá siendo agnóstico a la ubicación de los datos (rutas configurables vía `configs/`) para los datasets que aún falten.
2. **"PME4" y "PPGE"**: PME4 ya no es un dataset sin verificar — su manifiesto real confirma su estructura (ver 0quater). PPGE sigue sin identificar (sección 7). No voy a inventar características de lo que sigue sin evidencia real.

---

## 0bis. Inventario real verificado en la rama `main` de este repositorio

Tras la primera versión de este plan se confirmó que la rama `main` de `data_emociones` **sí contiene datos** (no era un repositorio vacío; el plan original se escribió antes de hacer `git fetch` de `main`). Se verificó el contenido real descargando los objetos Git LFS e inspeccionando los archivos (no solo los nombres). Resultado:

| Ruta en el repo | Contenido verificado | Estado |
|---|---|---|
| `Yale Face Database/` | 165 imágenes reales (15 sujetos × 11), íntegras | ✅ Utilizable (con las reservas de la sección 1.1) |
| `data/` (duplicado de Yale) | Árbol Git idéntico a `Yale Face Database/` | 🗑️ **Eliminado** (commit `e049a6d` en `main`) por ser un duplicado exacto |
| `EEG/desktop.ini`, `EEG/SEED_IV/desktop.ini` | Metadatos de sincronización de OneDrive/Windows, sin contenido de dataset | 🗑️ **Eliminados** (commit `e049a6d` en `main`) |
| `EEG/SEED_IV/` | Tras quitar el `desktop.ini`, la carpeta queda vacía | ❌ SEED-IV no llegó a subirse realmente; pendiente si se quiere usar |
| `EEG/SEED/*.npz` | Cargados con NumPy: `DatasetCaricatoNoImage.npz` → `arr_0` shape **(50910, 5, 62)** float32 (ventanas × 5 bandas de frecuencia × 62 canales, coherente con features de Differential Entropy por banda); `LabelsNoImage.npz` → 3 clases balanceadas (16800 / 16560 / 17550 → positivo/neutral/negativo); `SubjectsNoImage.npz` → **15 sujetos** (id 0–14), exactamente 3394 muestras por sujeto | ✅ **Es SEED real y consistente** con la literatura (15 sujetos, 3 clases, features DE 5-bandas). Es el dataset EEG con evidencia más sólida disponible hoy en el repo |
| `PPG_Dataset.csv` (raíz, vía Git LFS) | 2576 filas × 2000 muestras de señal PPG cruda + columna `Label` con valores **"Normal" (1282)** y **"MI" (1294)** — MI = *Myocardial Infarction* (infarto de miocardio) | ⚠️ **No es un dataset de emociones**: es un dataset de diagnóstico cardíaco. Ver análisis de reuso en la sección 1.4 |
| `PME4/`, `RAF-ML/` | Aparecen como **gitlinks rotos** (`160000 commit`, sin `.gitmodules`) — referencias a un commit de otro repositorio git que quedó pegado accidentalmente, sin URL asociada. Git no puede resolver ni descargar su contenido desde aquí | ❌ Sin contenido real en el repo. Fuente identificada (ver 0ter) pero bloqueada por red en este entorno |

### 0ter. Fuente de PME4 y RAF-ML — bloqueada por política de red de este entorno

Confirmaste que PME4 y RAF-ML están en esta carpeta de SharePoint/OneDrive institucional:
`https://unadvirtualedu-my.sharepoint.com/:f:/g/personal/mariac_rodriguez_unad_edu_co/...`

Se intentó el acceso (vía `curl` directo y vía la herramienta de fetch web) y **el dominio `unadvirtualedu-my.sharepoint.com` está bloqueado por la política de egress de este entorno en la nube** ("Access to unadvirtualedu-my.sharepoint.com is blocked by the network egress proxy"). No es un problema de permisos del enlace ni de credenciales: este sandbox no tiene salida de red hacia dominios SharePoint/OneDrive en absoluto.

**Alternativa propuesta:** descarga tú los archivos de esa carpeta a tu equipo y súbelos al repositorio (o a esta rama) mediante `git add`/`git push` normal — evitando repetir el error anterior de arrastrar una carpeta que internamente era otro repositorio git (lo que produjo los gitlinks rotos). Si prefieres, puedo dejar preparado un script simple de "ingesta" (mover archivos a `data/PME4/` y `data/RAF-ML/` con Git LFS activado para los binarios) para que tú ejecutes la subida desde tu máquina, ya que este entorno no puede alcanzar SharePoint directamente.

### 0quater. Actualización: subiste contenido real de RAF-DB y PME4 directamente a esta rama

Mientras se redactaba este plan, subiste (vía la interfaz web de GitHub, commits "Add files via upload") tres archivos directamente a la rama `claude/unimodal-emotion-recognition-models-oacupp`: `train_labels.csv`, `test_labels.csv` y `PME4/PME4_dataset_configs.csv`. Se verificaron con el mismo rigor que el resto:

| Archivo | Verificación real | Conclusión |
|---|---|---|
| `train_labels.csv` (12.271 filas) + `test_labels.csv` (3.068 filas) | Columnas `image,label`; nombres de archivo tipo `train_00001_aligned.jpg` / `test_0001_aligned.jpg` (convención de alineado facial típica de RAF); distribución de clases train = [1290, 281, 717, 4772, 1982, 705, 2524] para labels 1–7 | Esta distribución y estos tamaños **coinciden exactamente con el benchmark estándar de 7 clases de RAF-DB** (12.271 train + 3.068 test, con "Happy"=4 como clase mayoritaria). **Esto es RAF-DB, no RAF-ML** — son etiquetas de clase única (single-label), no distribuciones multi-etiqueta. Actualizo la sección 1.1 y la matriz para reflejar que lo que hay disponible es RAF-DB. **Importante:** solo están las etiquetas (CSV); las imágenes referenciadas (`train_00001_aligned.jpg`, etc.) no están en el repositorio todavía |
| `PME4/PME4_dataset_configs.csv` (3.829 filas) | Manifiesto por ensayo con columnas: `subject` (11 sujetos, 1–11), `trial`, `emotion`/`emotion_num`, tiempos de habla, y **rutas** a `audio_wav`, 2 variantes de MFCC de audio, `raw_eeg_filepath` (5kHz), `raw_emg_filepath` (5kHz), `processed_eeg_filepath` (1kHz), `processed_emg_filepath`, y `face_vgg16_features_filepath`. 7 emociones casi perfectamente balanceadas (anger 548, disgust 545, fear 547, happy 546, neutral 548, sad 547, surprise 548) | **Confirma la descripción del prompt original**: PME4 es multimodal (audio + EEG + EMG + rasgos faciales) con 7 clases y sí existe con esa estructura. Pero **es solo el índice/manifiesto**: los archivos `.npy`/`.wav` reales referenzados por esas rutas (p.ej. `s01/t001/s01_t001_raw_eeg_5kHz.npy`) **no están en el repositorio** (verificado: 0 archivos `.npy` o `.wav` en todo el árbol de git). Sin esos archivos no se puede entrenar todavía sobre PME4, solo planear la ingesta. Nota de calidad de datos: se detectó al menos una fila (`subject=1, trial=11`) cuya columna `face_vgg16_features_filepath` apunta a `s08/t350/...` — una posible inconsistencia sujeto/ensayo en el manifiesto que conviene que verifiques en la fuente original |

**Impacto en el plan:** ya no se trata a RAF-ML como "pendiente total" — se trata como **RAF-DB con etiquetas disponibles pero imágenes faltantes**, y RAF-ML como dataset separado que sigue sin evidencia de contenido real (el gitlink roto en `main` no aporta nada; si tienes RAF-ML de verdad, deberá subirse aparte). PME4 pasa de "sin verificar" a "estructura y taxonomía confirmadas, contenido de señal pendiente de subir".

### 0quinquies. Segunda actualización: subiste las imágenes reales y los .zip de señal de PME4 a `main`

Ajustaste las carpetas `PME4` y `RAF-ML` en `main` (commit "Corrigiendo carpetas PME4 y RAF-ML para que se vean normales"). Verificación real, incluyendo descarga y extracción efectiva de los archivos LFS (no solo nombres):

**PME4 — ahora completo y verificado de extremo a extremo.** `PME4/` contiene 11 archivos `sXX.zip` (~510 MB cada uno, ~5.6 GB en total) vía Git LFS. Se descargó `s01.zip` (510.324.081 bytes reales) y se abrió sin extraerlo por completo: contiene 2.808 archivos organizados como `s01/tNNN/...`, con exactamente 350 ensayos × {`raw_eeg_5kHz.npy`, `raw_emg_5kHz.npy`, 2 variantes de MFCC de audio, `processed_eeg`/`processed_emg`, el `.wav`}, tal como describe el manifiesto. Se cargó un archivo real con NumPy: `s01_t001_raw_emg_5kHz.npy` → shape **(6, 25000)** float64 (6 canales EMG, 25.000 muestras = 5 s a 5kHz), `s01_t001_raw_eeg_5kHz.npy` → shape **(8, 25000)**. Añadí `load_array_from_zip`/`load_raw_emg_from_zip`/`load_raw_eeg_from_zip` en `preprocessing/emg/pme4_dataset.py` para leer un `.npy` directamente desde el `.zip` sin extraer el archivo completo (~500 MB) a disco, y lo validé contra el archivo real. **PME4 pasa de "pendiente" a "listo para entrenar"** en la rama EMG.

**"RAF-ML" — resultó ser RAF-DB duplicado, no un dataset nuevo.** La carpeta traía: (a) las mismas imágenes y etiquetas de RAF-DB (12.271 train + 3.068 test, en subcarpetas por clase 1–7, distribución de clases idéntica a la ya verificada) y (b) una copia bit a bit de los 11 `.zip` de PME4 y de su manifiesto (mismos hashes SHA-256 exactos que `PME4/`). No apareció en ningún lado un dataset RAF-ML real (distribución de etiquetas). Con tu confirmación, **renombré `RAF-ML/` a `RAF-DB/`** y eliminé los archivos de PME4 duplicados dentro de esa carpeta; RAF-ML queda sin contenido, pendiente de que subas el dataset auténtico si lo tienes. Reescribí `preprocessing/fer/raf_db_dataset.py` para leer directamente la estructura real `RAF-DB/DATASET/{train,test}/<1-7>/*.jpg` (antes asumía imágenes sueltas + CSV) y lo validé cargando una imagen real (100×100 px) con el conteo de clases exacto. **RAF-DB pasa de "etiquetas sin imágenes" a "listo para entrenar"**.

**Nota sobre el histórico de `main`:** tu ajuste se subió como un *force-push* (reemplazó el historial anterior de `main` en vez de continuarlo), lo que revirtió sin querer la limpieza que habíamos hecho antes (el duplicado `data/` y los `desktop.ini` de OneDrive habían vuelto a aparecer). Los volví a eliminar en un commit normal sobre tu nuevo `main` — no hace falta que hagas nada al respecto, solo lo dejo documentado para que sepas por qué aparece dos veces en el historial.

**Resultado real del piloto EEG/SEED (LOSO completo, 15 sujetos, modelo baseline):** accuracy media 43.0% ± 10.9%, balanced accuracy 42.9% ± 11.0%, F1 macro 34.9% ± 13.8% (azar ≈ 33.3% en 3 clases). Resultado real, no simulado — ver `results/eeg/seed/baseline/loso_summary.json`. Es un resultado modesto pero por encima del azar y consistente con lo reportado en la literatura para reconocimiento de emociones EEG *subject-independent* (LOSO), que suele ser sustancialmente más difícil que la validación dentro del mismo sujeto.

---

## 1. Análisis de datasets por modalidad

### 1.1 FER

| Dataset | Sujetos/imágenes | Etiquetas originales | Formato | Acceso | Compatibilidad objetivo emocional | Riesgo de leakage |
|---|---|---|---|---|---|---|
| **FER2013** | 35.887 imágenes, sin ID de sujeto (recolectadas de Google Image Search) | 7 discretas: angry, disgust, fear, happy, sad, surprise, neutral | Grises 48×48, CSV con píxeles | Público (Kaggle) | Alta, pero con ruido de etiquetado documentado en literatura (~10-15%) y fuerte desbalance (disgust ≈ 1.5% del total) | No hay sujeto → no se puede hacer LOSO. Usar el split oficial (train/PublicTest/PrivateTest) y verificar duplicados casi-idénticos entre splits (perceptual hashing) |
| **AffectNet** | ~1M recolectadas, ~450K anotadas manualmente | 8 discretas (7 básicas + contempt) + valencia/activación continuas | JPEG variable + landmarks | **Restringido**: requiere solicitud formal a los autores (Mohammad Mahoor lab) y EULA | Alta (única con anotación dimensional PAD parcial a nivel FER) | Sin ID de sujeto consistente (web scraping) → riesgo de imágenes casi-duplicadas entre train/val; deduplicación recomendada |
| **RAF-DB** ✅ **etiquetas reales verificadas en esta rama** | 12.271 train + 3.068 test (confirmado por `train_labels.csv`/`test_labels.csv`); sin ID de sujeto (in-the-wild) | 7 básicas, single-label (confirmado: distribución de clases coincide con el benchmark estándar) | Nombres `*_aligned.jpg` (rostros ya alineados) — **pero las imágenes no están subidas aún, solo las etiquetas** | Normalmente restringido (solicitud a los autores), aunque las etiquetas ya están en el repo | Alta — es un dataset single-label estándar, comparable con FER2013/CK+ | Sin ID de sujeto → mismo tratamiento que AffectNet (dedup por hash una vez estén las imágenes) |
| **RAF-ML** | Desconocido — el gitlink roto en `main` no aporta contenido, y no se ha subido nada identificable como RAF-ML por separado | Se esperaría distribución sobre 6 emociones (label distribution learning), pero no hay evidencia real en el repo todavía | — | **Restringido**: requiere solicitud a los autores | No verificable aún | No verificable aún |
| **CK+** | 123 sujetos, 593 secuencias (327 con etiqueta de emoción) | 7 básicas + contempt (8 clases en el subset etiquetado), asignadas al frame de apex de la secuencia | Secuencias de imágenes (neutral→apex) | Moderado: formulario a CMU | Alta, pero es un dataset **posado/controlado en laboratorio** — no generaliza directamente a expresión espontánea | ID de sujeto disponible → **LOSO viable**. Riesgo real: frames de la misma secuencia repartidos entre train/test (leakage temporal) — deben agruparse por secuencia y por sujeto |
| **Yale Face Database** | 15 sujetos, 165 imágenes (11 por sujeto) | Categorías informales de variación (p.ej. "happy", "sad", "sleepy", "surprised", "wink") mezcladas con variaciones de iluminación/pose | GIF/PGM | Público | **Baja**: diseñado para reconocimiento facial bajo iluminación variable, no es un benchmark validado de emoción | Extremo: 15 sujetos es insuficiente para entrenar y validar una CNN profunda con generalización creíble. Cualquier LOSO aquí tendría intervalos de confianza inmanejables |

**Decisión propuesta para Yale:** no tratarlo como pipeline de DL convencional. Usarlo solo como (a) estudio cualitativo de transferencia (extracción de embeddings con un encoder ya entrenado en otro dataset FER y proyección/visualización, p.ej. t-SNE) o (b) demostración de overfitting controlado, documentando explícitamente la limitación en vez de reportar una métrica de "accuracy" que sería estadísticamente engañosa. Esto respeta la advertencia del punto 3 del prompt original.

### 1.2 EEG

| Dataset | Sujetos | Señales | Etiquetas | Fs | Duración | Acceso | Notas |
|---|---|---|---|---|---|---|---|
| **DEAP** | 32 | EEG 32 canales + periféricas (EOG, EMG zigomático/trapecio, GSR, respiración, **PPG/pletismografía**, temperatura); video facial frontal disponible para 22/32 sujetos | Valencia/activación/dominancia/liking, continuas 1–9 (autoreporte) | 512 Hz (versión preprocesada a 128 Hz) | 40 videos musicales de 1 min/sujeto | Restringido (EULA, Queen Mary University London) | **DEAP es multimodal**: la misma sesión aporta datos potencialmente a las cuatro ramas (EEG, EMG, PPG, y parcialmente FER vía el video facial). Esto no genera leakage dentro de un modelo unimodal, pero es crítico para la futura fusión: el split por sujeto debe ser **idéntico y consistente entre las cuatro ramas basadas en DEAP** |
| **SEED** ✅ **datos reales verificados en `main`** | 15 (confirmado: `SubjectsNoImage.npz`, ids 0–14, 3394 muestras/sujeto exactas) | EEG 62 canales, ya reducido a features **Differential Entropy por banda**: `DatasetCaricatoNoImage.npz` shape (50910, 5, 62) = ventanas × 5 bandas × 62 canales | 3 discretas: positive/neutral/negative (confirmado: `LabelsNoImage.npz`, 16800/16560/17550 — balanceadas), asignadas por el clip de estímulo (no autoreporte) | No aplica directamente (los datos ya vienen como features DE, no señal cruda); el preprocesamiento original habría usado ~200 Hz | Clips de películas, 3 sesiones por sujeto en el diseño original (no se puede confirmar desde los `.npz` si las 3 sesiones están mezcladas en las 3394 muestras/sujeto — pendiente de revisar metadatos adicionales si existen) | Restringido (BCMI lab, SJTU) — pero **ya está en este repo** | Paradigma de etiquetado distinto a DEAP (etiqueta por estímulo, no por percepción individual) → no comparable directamente sin justificación. Al ser ya features (no señal cruda), el pipeline de esta rama empieza en "extracción/aprendizaje de características → Deep Learning", no en filtrado crudo |
| **DREAMER** | 23 | EEG 14 canales (Emotiv EPOC) + **ECG** (no EMG) | Valencia/activación/dominancia continuas 1–5 (autoreporte) | EEG 128 Hz, ECG 256 Hz | 18 clips de película | Moderado (acuerdo de datos) | Confirmo lo que indica la propia consigna: DREAMER trae ECG, no EMG — coherente con que el prompt no lo liste en la sección EMG |

### 1.3 EMG

| Dataset | Sujetos | ¿EMG facial real disponible? | Etiquetas | Acceso | Notas |
|---|---|---|---|---|---|
| **DEAP** | 32 | **Sí**: 2 canales (zigomático mayor, trapecio) | Igual que EEG (PAD continuo) | Restringido | Mismo caveat de consistencia de splits entre ramas |
| **MAHNOB-HCI** | 27 | ⚠️ **Por verificar**: la documentación pública estándar de MAHNOB-HCI que conozco incluye EEG (32 ch), ECG, GSR, respiración, temperatura y **video facial/corporal**, pero no recuerdo con certeza canales de electrodos EMG dedicados — la expresión facial ahí se capta por video, no por EMG | Autoreporte + etiquetado externo de arousal/valencia | Restringido | **No lo incluyo en la matriz experimental de EMG hasta confirmar** que el release contiene señal EMG cruda. Alternativa si no la tiene: excluirlo de esta rama (podría eventualmente aportar a FER vía video) |
| **AMIGOS** | 40 | ⚠️ **Por verificar**: el release estándar que conozco es EEG (14 ch Emotiv), ECG, GSR + video facial, sin canal EMG dedicado | Valencia/activación/dominancia continuas + anotación externa | Restringido | Mismo caveat que MAHNOB-HCI |
| **PME4** ✅ **manifiesto real verificado en esta rama** | 11 sujetos (confirmado: ids 1–11 en `PME4_dataset_configs.csv`) | **Sí**: el manifiesto confirma `raw_emg_filepath` (5kHz) y `processed_emg_filepath` por cada uno de los 3.829 ensayos | 7 discretas, balanceadas (anger/disgust/fear/happy/neutral/sad/surprise, ~547 c/u) | Fuente identificada (SharePoint institucional), bloqueada por red en este entorno (ver 0ter) | Estructura y taxonomía confirmadas; **falta subir los archivos `.npy` de señal real** (el CSV es solo el índice/manifiesto, 0 archivos de señal presentes en el repo) |

**Implicación actualizada:** ahora hay **dos** datasets con evidencia real de canal EMG: DEAP (EULA pendiente) y **PME4 (manifiesto confirmado, contenido de señal pendiente de subida)**. PME4 es además multimodal por diseño (audio + EEG + EMG + features faciales por el mismo sujeto/ensayo), lo que lo hace muy valioso para la futura fusión si se puede completar la subida de los archivos de señal. MAHNOB-HCI/AMIGOS siguen como "pendientes de verificación de canal EMG" hasta que confirmes.

### 1.4 PPG

| Dataset | Sujetos | Señal | Etiquetas | Fs | Acceso | Notas |
|---|---|---|---|---|---|---|
| **DEAP** | 32 | Canal de pletismografía (BVP/PPG) | PAD continuo | 512→128 Hz | Restringido | Mismo caveat de consistencia de splits |
| **WESAD** | 15 | PPG/BVP vía Empatica E4 (muñeca) + ECG/EMG/EDA/temp/resp vía RespiBAN (pecho) | **Condiciones**, no emociones básicas: baseline / estrés (TSST) / diversión (amusement) / meditación; + autoreporte PANAS/SAM | E4 BVP 64 Hz, RespiBAN 700 Hz | Moderado (formulario Bosch) | El ground truth es **por condición experimental**, no una taxonomía de emociones discretas — mapeable de forma aproximada a alto/bajo arousal y valencia negativa/positiva, pero debe documentarse como un mapeo, no una equivalencia directa |
| **PPGE** | — | Desconocido | Desconocido | — | Desconocido | Igual que PME4: no puedo verificar este dataset. Pendiente de que aportes la fuente |
| **`PPG_Dataset.csv` (ya presente en `main`)** | Sin metadato de sujeto en el CSV (2576 filas, no se puede saber cuántos individuos distintos hay sin un ID) | 2000 muestras de señal PPG cruda por fila (fs y duración no documentadas en el repo — no hay archivo README asociado) | **"Normal" (1282) / "MI"** — infarto de miocardio, no una emoción | Desconocida (no hay metadato) | Ya está en el repo (posiblemente un dataset público tipo Kaggle "PPG for MI detection", pero no puedo confirmar la fuente exacta sin metadato) | Ver análisis de reutilización abajo |

### Análisis de reutilización de `PPG_Dataset.csv` (punto 3 de tu confirmación)

**No puede usarse como ground truth de emoción.** "Normal" vs "MI" es un constructo clínico (presencia de infarto), no una etiqueta de valencia/activación/emoción — usarlo como si fuera una clase emocional introduciría una variable de confusión no válida (un paciente con infarto tiene alteraciones cardiovasculares que no tienen relación causal con un estado emocional inducido experimentalmente). Esto sería mezclar constructos, algo que el punto 13 y el punto 21 del plan piden evitar explícitamente.

**Usos legítimos que sí propongo:**
1. **Pre-entrenamiento del encoder PPG (transfer learning de bajo nivel):** usar las 2576 señales para entrenar de forma no supervisada / auto-supervisada (autoencoder, o un encoder contrastivo) un extractor de morfología de pulso genérico, y luego transferir/ajustar (fine-tuning) ese encoder sobre el dataset real de emoción (DEAP-PPG o WESAD) que sí tenga etiqueta afectiva. Esto es razonable porque la morfología del pulso PPG es un dato de dominio compartido, aunque la tarea (infarto vs. emoción) sea distinta.
2. **Banco de pruebas para el pipeline de preprocesamiento:** validar la detección de picos, el cálculo de HR/PRV-HRV y las características de morfología de onda (secciones 6 y 9 del plan) sobre señal PPG real, antes de aplicarlas al dataset de emoción — como prueba de humo del código, no como fuente de resultados de investigación.
3. **Nunca:** como fila adicional de "dataset de emoción PPG" en la matriz experimental (sección 6), ni mezclado con DEAP/WESAD en el mismo entrenamiento.

**Limitación a documentar:** no hay README ni metadato de frecuencia de muestreo junto al CSV en el repo, así que la frecuencia de muestreo real (necesaria para convertir "muestras" a HR real en bpm) queda como supuesto a verificar antes de usarlo, incluso para los usos 1 y 2.

---

## 2. Riesgos de leakage identificados (resumen transversal)

1. **Compartición de sujetos entre modalidades (DEAP):** si en el futuro se funde EEG+EMG+PPG(+FER parcial) de DEAP, el sujeto debe quedar en el mismo fold en las cuatro ramas. Se define esto ahora aunque la fusión no se implemente todavía, para no tener que re-particionar después.
2. **Normalización/estandarización:** todo escalado (z-score, min-max, filtrado adaptativo) se ajusta **solo con el fold de entrenamiento** y se aplica después a validación/test.
3. **Augmentación antes del split:** prohibido; el split ocurre primero, la augmentación solo sobre el conjunto de entrenamiento resultante.
4. **Selección de características (EMG/PPG) usando todo el dataset:** prohibido; cualquier selección o PCA se ajusta dentro de cada fold de entrenamiento.
5. **Datasets sin ID de sujeto (FER2013, AffectNet, RAF-DB/RAF-ML):** riesgo de imágenes casi-duplicadas repartidas entre splits. Mitigación: hashing perceptual (pHash) + verificación de similitud antes de fijar el split final.
6. **CK+:** frames de una misma secuencia no deben quedar repartidos entre train y test; agrupar por secuencia y por sujeto.
7. **SEED:** sesiones repetidas del mismo sujeto no deben quedar repartidas entre train y test.
8. **Expresión explícita (PrEmo/SAM):** se reserva exclusivamente como variable de contraste/ground truth para la futura fase de validación cruzada de constructos, nunca como característica de entrada a los modelos unimodales que se entrenan ahora.

---

## 3. Estrategia de validación por dataset

| Dataset | Estrategia | Justificación |
|---|---|---|
| FER2013 | Split oficial train/PublicTest/PrivateTest + dedup | No hay sujeto; el split ya es un estándar de comparación en literatura |
| AffectNet | Split estratificado por clase (80/10/10) + dedup | Sin ID de sujeto; volumen grande permite split simple sin comprometer poder estadístico |
| RAF-DB/RAF-ML | Split oficial provisto por los autores | Mantiene comparabilidad con literatura previa |
| CK+ | **LOSO** agrupado por sujeto y por secuencia (o k-fold agrupado, p.ej. 10-fold, si LOSO resulta computacionalmente costoso dado 123 sujetos) | Objetivo: generalización a sujetos no vistos; dataset lo permite por tener ID |
| Yale | Ninguna métrica de generalización formal — solo análisis cualitativo/transferencia | 15 sujetos es insuficiente para una estimación de generalización con intervalos de confianza razonables |
| DEAP, DREAMER | **LOSO** (32 y 23 sujetos respectivamente) | Tamaño manejable, evalúa generalización entre sujetos, es el estándar en literatura de EEG afectivo |
| SEED | **LOSO**, agrupando las 3 sesiones de cada sujeto | Evita leakage de sesión; también se reportará opcionalmente "cross-session, within-subject" como análisis secundario (no como validación principal) |
| MAHNOB-HCI, AMIGOS | LOSO **condicionado a confirmar disponibilidad de canal EMG** | Ver sección 7 |
| WESAD | **LOSO** (15 sujetos) | Estándar en literatura de estrés/afecto con wearables |
| PME4, PPGE | Pendiente | Ver sección 7 |

---

## 4. Homogeneización de taxonomías emocionales

| Dataset | Taxonomía original | Tipo | ¿Comparable con básicas 6-7? | ¿Comparable con PAD? |
|---|---|---|---|---|
| FER2013 | 7 discretas | Categórica | Sí (referencia) | No |
| AffectNet | 8 discretas + V/A continuo | Categórica + dimensional | Sí (7 básicas + contempt aparte) | Sí (única FER con V/A) |
| RAF-DB | 7 básicas (+11 compuestas) | Categórica | Sí | No |
| RAF-ML | Distribución sobre 6 emociones | **Label distribution learning** | Parcial (requiere argmax o KL, no comparación directa de accuracy) | No |
| CK+ | 7 básicas + contempt | Categórica (posada) | Sí, con reserva por ser posado vs. espontáneo | No |
| Yale | Categorías informales no estandarizadas | No validada | No | No |
| DEAP | Valencia/Activación/Dominancia/Liking 1–9 | Dimensional continua | No sin binarizar (umbral a justificar, típicamente en 5) | Sí (referencia) |
| SEED | Positivo/Neutral/Negativo (por estímulo) | Categórica (3 clases) | No (taxonomía distinta: 3 vs 6-7) | Parcial: mapea solo a signo de valencia, no a arousal |
| DREAMER | V/A/D 1–5 | Dimensional continua | No sin binarizar | Sí |
| MAHNOB-HCI | V/A 1–9 + etiquetas de emoción por palabra clave | Mixta | Parcial | Sí |
| AMIGOS | V/A/D continuo + anotación externa | Dimensional | No sin binarizar | Sí |
| WESAD | Condición experimental (baseline/estrés/diversión/meditación) + PANAS/SAM | Categórica por condición, no por emoción básica | No directamente | Aproximado (mapeo, no equivalencia) |

**Regla de decisión (conservadora, según punto 13 del prompt):**
- Cada experimento se entrena y reporta **primero en su propia taxonomía original**.
- Solo se plantea una comparación secundaria (no un entrenamiento conjunto) entre datasets dimensionales (DEAP, DREAMER, AMIGOS, AffectNet-dim, MAHNOB) sobre valencia/activación **binarizadas**, documentando explícitamente el umbral usado y sus limitaciones.
- SEED y WESAD no se fuerzan a esa comparación binaria: SEED carece de dimensión de activación, y WESAD etiqueta condiciones, no percepción emocional directa. Se documentan como no comparables sin mapeo adicional, que quedaría fuera del alcance de esta etapa a menos que tú lo autorices explícitamente.
- No se mezclan datasets categóricos y dimensionales en un mismo entrenamiento.

---

## 5. Arquitecturas propuestas por modalidad (Baseline / Avanzada / Propuesta)

| Modalidad | Baseline (A) | Avanzada (B) | Propuesta (C, candidata a encoder unimodal) |
|---|---|---|---|
| FER (imagen estática: FER2013, AffectNet, RAF-DB/ML) | CNN pequeña desde cero (4-6 capas conv) | ResNet18/EfficientNet-B0 preentrenado + fine-tuning | El mejor de B, con cabeza dual: clasificación + proyección a embedding (128-256 d) para fusión futura. Para RAF-ML: misma arquitectura pero con cabeza de salida softmax + pérdida KL-divergence en vez de cross-entropy |
| FER (secuencia: CK+) | CNN frame-a-frame (solo frame de apex) | CNN (features por frame) + BiLSTM sobre la secuencia | CNN + Transformer temporal ligero, comparado contra BiLSTM |
| FER (Yale) | Transfer learning (encoder congelado de un modelo FER entrenado en otro dataset) + clasificador lineal | — (no se justifica una arquitectura "avanzada" separada dado el tamaño) | No aplica una "propuesta" con pretensión de generalización; se documenta como estudio de caso |
| EEG | CNN 1D sobre señal temporal filtrada por banda, o CNN 2D sobre PSD/DE por banda×canal | CNN + BiLSTM sobre secuencia de ventanas | Transformer temporal o modelo basado en grafo (CI-Graph/GNN) sobre topología de electrodos, comparado empíricamente contra B antes de adoptarlo como definitivo |
| EMG | MLP sobre características manuales (RMS, MAV, WL, ZC, SSC, amplitud/duración de contracción, MNF, MDF) | CNN 1D sobre señal cruda segmentada | CNN+LSTM o Transformer, comparado contra Graph Transformer solo si el número de canales/dataset lo justifica (con 1-2 canales EMG como en DEAP, un modelo de grafo tiene poco sentido — se evaluará esto explícitamente antes de implementarlo) |
| PPG | CNN 1D sobre señal filtrada | CNN + BiLSTM | Transformer temporal, comparado contra B |

En todos los casos, la arquitectura "C" expone dos salidas (clasificación + embedding), como pide el punto 14, y la elección de "candidata a rama multimodal" se hará después de comparar A/B/C, no asumiendo que C gana por defecto (punto 8).

---

## 6. Matriz experimental (ajustada tras el análisis de disponibilidad)

| Modalidad | Dataset | Baseline | Avanzado | Propuesto | Validación | Estado |
|---|---|---|---|---|---|---|
| FER | FER2013 | CNN | ResNet18/EfficientNet-B0 | Mejor de B + embedding | Split oficial + dedup | ✅ Listo para implementar |
| FER | AffectNet | CNN | EfficientNet | Mejor de B + embedding | Split estratificado + dedup | ⚠️ Requiere que gestiones el acceso (EULA) antes de descargar |
| FER | RAF-DB | CNN | ResNet | Mejor de B + embedding | Split oficial (`RAF-DB/DATASET/{train,test}/<1-7>`, 12.271/3.068 verificado) | ✅ **Imágenes y etiquetas reales en el repo, listo para entrenar** (`preprocessing/fer/raf_db_dataset.py` ya implementado y probado) |
| FER | RAF-ML | CNN (softmax+KL) | ResNet (softmax+KL) | Mejor de B + embedding | Split oficial | ❌ **Pendiente**: lo que había con este nombre resultó ser RAF-DB duplicado (ver 0quinquies); sin evidencia real de RAF-ML en ningún lado |
| FER | CK+ | CNN (frame apex) | CNN+BiLSTM (secuencia) | Transformer temporal | LOSO agrupado por sujeto/secuencia | ⚠️ Requiere formulario CMU |
| FER | Yale | Transfer learning | — | Estudio de caso, no benchmark | Sin LOSO formal | ⚠️ Uso restringido a análisis cualitativo (ver 1.1) |
| EEG | DEAP | CNN 1D/2D | CNN+BiLSTM | Transformer/Graph (a decidir tras baseline) | LOSO | ⚠️ Requiere EULA |
| EEG | **SEED** | CNN sobre features DE (50910,5,62) | CNN+BiLSTM | A decidir tras baseline | LOSO (15 sujetos, ids confirmados) | ✅ **Datos reales ya en `main`, verificados (shapes y balance de clases confirmados). Listo para implementar sin depender de descargas externas** |
| EEG | DREAMER | CNN | CNN+BiLSTM | A decidir | LOSO | ⚠️ Requiere acuerdo de datos |
| EMG | DEAP | MLP (features) | CNN 1D | CNN+LSTM/Transformer | LOSO | ⚠️ Requiere EULA. Dataset primario confirmado para esta rama |
| EMG | MAHNOB-HCI | — | — | — | — | ❌ **Pendiente**: confirmar si el release incluye canal EMG crudo antes de comprometer diseño |
| EMG | AMIGOS | — | — | — | — | ❌ **Pendiente**: mismo caveat |
| EMG | PME4 | MLP (features) | CNN 1D | CNN+LSTM/Transformer | LOSO (11 sujetos) | ✅ **Señal real verificada y cargable** (`s01.zip` descargado, `raw_emg_5kHz.npy` shape (6,25000) confirmado con NumPy); `preprocessing/emg/pme4_dataset.py` ya lee directo desde los `.zip`, listo para entrenar |
| PPG | DEAP | CNN 1D | CNN+BiLSTM | Transformer | LOSO | ⚠️ Requiere EULA |
| PPG | WESAD | CNN 1D | CNN+BiLSTM | Transformer | LOSO | ⚠️ Requiere formulario Bosch; etiqueta = condición, documentar mapeo a V/A |
| PPG | PPGE | — | — | — | — | ❌ **Pendiente**: dataset no verificable, necesito fuente |
| PPG | `PPG_Dataset.csv` (ya en `main`) | — | — | — | No aplica como rama de emoción | ❌ **Excluido de la matriz de emoción** (etiqueta clínica Normal/MI, no afectiva). Reservado solo para pre-entrenamiento de encoder y pruebas de pipeline (ver 1.4) |

**Nota importante:** ningún experimento se "fuerza". Las filas marcadas ❌ no entran a la matriz de ejecución hasta que confirmes fuente/acceso/estructura, tal como pide el punto 21 (no simular resultados, no forzar experimentación sin base).

---

## 7. Datasets que requieren tu confirmación antes de seguir

1. ~~**PME4**~~ — **resuelto**: señal real verificada y cargable (ver 0quinquies).
2. **RAF-ML** — sigue sin ningún contenido verificable; lo que tenía este nombre era RAF-DB duplicado (ver 0quinquies) y ya se limpió. Si tienes el dataset RAF-ML real, súbelo aparte.
3. ~~**RAF-DB (imágenes)**~~ — **resuelto**: imágenes y etiquetas reales verificadas, listo para entrenar (ver 0quinquies).
4. **PPGE** — sigue sin identificar; no está en el repo ni tengo fuente. Si es distinto de `PPG_Dataset.csv` (que ya descarté como no-emocional), necesito su origen.
5. **MAHNOB-HCI y AMIGOS para EMG** — necesito que confirmes si tienes acceso a una versión de estos datasets que incluya canal EMG facial crudo (electrodos), distinto del video facial. Si no lo tienen, propongo: (a) excluirlos de la rama EMG, o (b) reasignarlos como fuente adicional de video para la rama FER en una fase posterior (fuera del alcance actual).
6. **Acceso EULA pendiente** en AffectNet, CK+, DEAP, DREAMER, MAHNOB-HCI, AMIGOS, WESAD — asumo que como investigador doctoral ya gestionarás o tienes en trámite estos accesos; el código se construirá para leer desde una ruta local configurable, nunca para descargar automáticamente contenido restringido. **SEED, RAF-DB y PME4 ya no están en esta lista**: los tres tienen contenido real, verificado y listo para entrenar en el repo.

---

## 8. Métricas y análisis estadístico (resumen, se detalla en la implementación)

- Clasificación: Accuracy, Balanced Accuracy, Precision/Recall/F1 (macro y weighted), matriz de confusión, ROC-AUC/PR-AUC cuando aplique.
- Dimensional (V/A/D): MAE, RMSE, correlación de Pearson y Spearman, CCC.
- Comparación entre modelos (A vs B vs C): por sujeto cuando haya LOSO, con prueba de normalidad (Shapiro-Wilk) antes de decidir entre ANOVA de medidas repetidas o su alternativa no paramétrica (Friedman + post-hoc de Nemenyi/Wilcoxon con corrección), justificando la elección en cada caso — nunca aplicando ANOVA por defecto.

---

## 9. Qué NO se hace en esta etapa

- No se implementa fusión multimodal ni cross-attention entre modalidades.
- No se descarga ni se simula ningún dataset.
- No se generan resultados numéricos de ejemplo/placeholder.
- No se fuerza el uso de PME4, PPGE, MAHNOB-EMG ni AMIGOS-EMG hasta verificación.
- No se trata a Yale como benchmark de generalización.
- No se usa `PPG_Dataset.csv` (Normal/MI) como fuente de etiqueta emocional.

---

## 11. Resultados reales — EEG/SEED, piloto A/B/C completo (LOSO, 15 sujetos)

Los tres modelos definidos en la sección 5 se entrenaron y evaluaron de extremo a extremo sobre los datos reales de SEED, con LOSO completo (15 folds, sujetos disjuntos, normalización ajustada solo en train — auditoría de leakage pasó en los 15 folds). Resultados reales, no simulados (`results/eeg/seed/{baseline,cnn_bilstm,transformer}/loso_summary.json`):

| Modelo | Accuracy | Balanced Accuracy | F1 macro | Tiempo total (CPU) |
|---|---|---|---|---|
| A — CNN baseline | 43.0% ± 10.9% | 42.9% ± 11.0% | 34.9% ± 13.8% | ~19 min |
| B — CNN + BiLSTM | 44.5% ± 11.8% | 44.5% ± 11.8% | 35.1% ± 16.4% | ~52 min |
| C — Transformer | **54.3% ± 13.7%** | **54.0% ± 13.6%** | **47.8% ± 16.8%** | ~5 min |

(Azar ≈ 33.3% en un problema de 3 clases.)

**Comparación estadística** (`evaluation/compare_eeg_models.py`, resultado en `results/eeg/seed/model_comparison.json`), siguiendo la regla de la sección 8 de no aplicar ANOVA por defecto:

- Shapiro-Wilk sobre la accuracy por sujeto de cada modelo: p > 0.05 en los tres → no se rechaza normalidad → se usa **ANOVA de un factor** para la comparación conjunta: F=3.50, **p=0.039** (diferencia estadísticamente significativa entre A, B y C al nivel 0.05).
- Comparaciones pareadas (t de Student pareada, misma justificación de normalidad):
  - A vs B: p=0.621 — **sin diferencia significativa** (el modelo "avanzado" no mejora de forma confiable sobre el baseline en este piloto).
  - A vs C: p=0.0195 — **diferencia significativa**, Transformer supera al baseline.
  - B vs C: p=0.079 — diferencia marginal (no significativa al 0.05, pero cercana), Transformer por encima de CNN+BiLSTM.

**Interpretación (sin sobre-interpretar un único piloto):** en este dataset y con estas features (DE por banda ya extraídas, no señal cruda), el Transformer resultó ser el candidato más sólido tanto en desempeño como en costo computacional (más rápido de entrenar que el CNN+BiLSTM). El CNN+BiLSTM no justificó su complejidad adicional frente al baseline en este piloto — resultado real y honesto, no el que "se esperaría" si se asumiera que más complejidad siempre gana (regla de la sección 8). Esto es un solo dataset y una sola corrida por modelo (sin repetición con distintas semillas todavía), así que se reporta como evidencia inicial, no como conclusión definitiva sobre las tres arquitecturas en general.

Los embeddings (128-d) de cada modelo y cada fold quedaron guardados en `embeddings/eeg/seed/<modelo>/subject_XX.npz` (no versionados en git, ver `.gitignore`; se regeneran con `training/train_eeg.py`), listos para la futura etapa de fusión multimodal.

---

## 12. Próximos pasos (esperando tu confirmación)

Si apruebas este plan, la siguiente fase generará:

1. Estructura de proyecto (`preprocessing/`, `models/`, `training/`, `evaluation/`, `experiments/`, `embeddings/`, `checkpoints/`, `results/`, `configs/`, `notebooks/`) con configs YAML por experimento.
2. Módulos de preprocesamiento por modalidad (interfaces + lógica). **Para EEG/SEED esto ya puede ejecutarse de verdad hoy** (datos reales verificados). **RAF-DB (FER) y PME4 (EMG/multimodal)** quedarían listos en cuanto subas, respectivamente, las imágenes `*_aligned.jpg` y los `.npy`/`.wav` de señal — el código de preprocesamiento puede escribirse ya contra su estructura real (ya conocida), aunque no pueda ejecutarse end-to-end hasta que lleguen esos archivos.
3. Definición de arquitecturas A/B/C por modalidad en PyTorch, empezando por SEED (EEG) como caso piloto totalmente ejecutable hoy.
4. Lógica de split LOSO/estratificado con auditoría de leakage automatizada (SEED: LOSO sobre 15 sujetos confirmados; PME4: LOSO sobre 11 sujetos confirmados; RAF-DB: split oficial ya definido por los CSV).
5. Scripts de entrenamiento/evaluación parametrizados por config.

**Preguntas abiertas que necesito que resuelvas antes de continuar** (ver sección 7): que subas las imágenes de RAF-DB, los archivos de señal de PME4 y el dataset RAF-ML real (los tres bloqueados por red desde este entorno hacia SharePoint), fuente de PPGE, y confirmación de canal EMG en MAHNOB-HCI/AMIGOS.

¿Apruebas este plan experimental tal como está, con las exclusiones/reservas señaladas (Yale como estudio de caso, RAF-ML pendiente de contenido real, `PPG_Dataset.csv` excluido como ground truth emocional, MAHNOB-EMG/AMIGOS-EMG/PPGE pendientes), y confirmas que empecemos a generar el código real primero para **EEG/SEED** (ejecutable hoy) y dejemos el andamiaje de **RAF-DB (FER)** y **PME4 (EMG)** listo para cuando subas los archivos de señal/imagen que faltan?
