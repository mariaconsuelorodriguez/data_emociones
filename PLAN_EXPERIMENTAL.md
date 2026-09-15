# PLAN EXPERIMENTAL PROPUESTO
## Modelos unimodales de Deep Learning para reconocimiento de emociones (FER, EEG, EMG, PPG)

**Proyecto:** Modelo Multimodal para la identificación de emociones basado en la correlación entre FER, señales biofeedback (EEG, EMG, PPG) y expresión explícita — estudio de caso educativo.
**Etapa actual:** Diseño y validación de ramas unimodales (sin fusión).
**Estado:** Borrador para revisión — **pendiente de aprobación antes de generar código.**

---

## 0. Alcance y restricciones del entorno de trabajo

Antes de entrar al análisis, dos advertencias metodológicas que condicionan todo lo demás (regla científica del punto 21: no inventar resultados):

1. **Ninguno de los datasets está presente en este repositorio ni en este entorno de ejecución.** Todos requieren descarga desde el proveedor original, y varios (AffectNet, RAF-DB/RAF-ML, CK+, DEAP, SEED, DREAMER, MAHNOB-HCI, AMIGOS, WESAD) exigen firma de un EULA / acuerdo de uso académico antes de obtener acceso. El código que se genere en la siguiente fase debe ser **agnóstico a la ubicación de los datos** (rutas configurables vía `configs/`), y la ejecución real del entrenamiento deberá hacerse en tu entorno local/HPC una vez tengas los datasets descargados legalmente. Este entorno no descargará ni simulará datos.
2. **"PME4" y "PPGE"** no corresponden a datasets que pueda verificar con la información disponible (no son nombres estándar reconocibles en la literatura de EEG/EMG/PPG-emoción que yo pueda confirmar con certeza). Los trato como **pendientes de verificación** (sección 7): necesito que confirmes fuente, licencia, estructura de archivos y diccionario de etiquetas antes de incluirlos en la matriz experimental con la misma seriedad que el resto. No voy a inventar sus características.

---

## 1. Análisis de datasets por modalidad

### 1.1 FER

| Dataset | Sujetos/imágenes | Etiquetas originales | Formato | Acceso | Compatibilidad objetivo emocional | Riesgo de leakage |
|---|---|---|---|---|---|---|
| **FER2013** | 35.887 imágenes, sin ID de sujeto (recolectadas de Google Image Search) | 7 discretas: angry, disgust, fear, happy, sad, surprise, neutral | Grises 48×48, CSV con píxeles | Público (Kaggle) | Alta, pero con ruido de etiquetado documentado en literatura (~10-15%) y fuerte desbalance (disgust ≈ 1.5% del total) | No hay sujeto → no se puede hacer LOSO. Usar el split oficial (train/PublicTest/PrivateTest) y verificar duplicados casi-idénticos entre splits (perceptual hashing) |
| **AffectNet** | ~1M recolectadas, ~450K anotadas manualmente | 8 discretas (7 básicas + contempt) + valencia/activación continuas | JPEG variable + landmarks | **Restringido**: requiere solicitud formal a los autores (Mohammad Mahoor lab) y EULA | Alta (única con anotación dimensional PAD parcial a nivel FER) | Sin ID de sujeto consistente (web scraping) → riesgo de imágenes casi-duplicadas entre train/val; deduplicación recomendada |
| **RAF-DB / RAF-ML** | ~30.000 (RAF-DB) — RAF-ML es un subconjunto distinto con anotación de **distribución de etiquetas** (multi-label / label distribution learning), no clase única | RAF-DB: 7 básicas (+ subset de 11 compuestas); RAF-ML: distribución sobre 6 emociones | JPEG in-the-wild | **Restringido**: requiere solicitud a los autores | Alta, pero **RAF-ML no es un problema de clasificación single-label** — requiere función de pérdida distribucional (KL-divergence / Jensen-Shannon), no cross-entropy estándar | Sin ID de sujeto → mismo tratamiento que AffectNet |
| **CK+** | 123 sujetos, 593 secuencias (327 con etiqueta de emoción) | 7 básicas + contempt (8 clases en el subset etiquetado), asignadas al frame de apex de la secuencia | Secuencias de imágenes (neutral→apex) | Moderado: formulario a CMU | Alta, pero es un dataset **posado/controlado en laboratorio** — no generaliza directamente a expresión espontánea | ID de sujeto disponible → **LOSO viable**. Riesgo real: frames de la misma secuencia repartidos entre train/test (leakage temporal) — deben agruparse por secuencia y por sujeto |
| **Yale Face Database** | 15 sujetos, 165 imágenes (11 por sujeto) | Categorías informales de variación (p.ej. "happy", "sad", "sleepy", "surprised", "wink") mezcladas con variaciones de iluminación/pose | GIF/PGM | Público | **Baja**: diseñado para reconocimiento facial bajo iluminación variable, no es un benchmark validado de emoción | Extremo: 15 sujetos es insuficiente para entrenar y validar una CNN profunda con generalización creíble. Cualquier LOSO aquí tendría intervalos de confianza inmanejables |

**Decisión propuesta para Yale:** no tratarlo como pipeline de DL convencional. Usarlo solo como (a) estudio cualitativo de transferencia (extracción de embeddings con un encoder ya entrenado en otro dataset FER y proyección/visualización, p.ej. t-SNE) o (b) demostración de overfitting controlado, documentando explícitamente la limitación en vez de reportar una métrica de "accuracy" que sería estadísticamente engañosa. Esto respeta la advertencia del punto 3 del prompt original.

### 1.2 EEG

| Dataset | Sujetos | Señales | Etiquetas | Fs | Duración | Acceso | Notas |
|---|---|---|---|---|---|---|---|
| **DEAP** | 32 | EEG 32 canales + periféricas (EOG, EMG zigomático/trapecio, GSR, respiración, **PPG/pletismografía**, temperatura); video facial frontal disponible para 22/32 sujetos | Valencia/activación/dominancia/liking, continuas 1–9 (autoreporte) | 512 Hz (versión preprocesada a 128 Hz) | 40 videos musicales de 1 min/sujeto | Restringido (EULA, Queen Mary University London) | **DEAP es multimodal**: la misma sesión aporta datos potencialmente a las cuatro ramas (EEG, EMG, PPG, y parcialmente FER vía el video facial). Esto no genera leakage dentro de un modelo unimodal, pero es crítico para la futura fusión: el split por sujeto debe ser **idéntico y consistente entre las cuatro ramas basadas en DEAP** |
| **SEED** | 15 | EEG 62 canales | 3 discretas: positive/neutral/negative, asignadas por el clip de estímulo (no autoreporte) | 1000 Hz (preprocesado 200 Hz) | Clips de películas, 3 sesiones por sujeto (repetidas en días distintos) | Restringido (BCMI lab, SJTU) | Paradigma de etiquetado distinto a DEAP (etiqueta por estímulo, no por percepción individual) → no comparable directamente sin justificación. Estructura de sesiones repetidas: LOSO debe agrupar **todas las sesiones de un sujeto** en el mismo lado del split |
| **DREAMER** | 23 | EEG 14 canales (Emotiv EPOC) + **ECG** (no EMG) | Valencia/activación/dominancia continuas 1–5 (autoreporte) | EEG 128 Hz, ECG 256 Hz | 18 clips de película | Moderado (acuerdo de datos) | Confirmo lo que indica la propia consigna: DREAMER trae ECG, no EMG — coherente con que el prompt no lo liste en la sección EMG |

### 1.3 EMG

| Dataset | Sujetos | ¿EMG facial real disponible? | Etiquetas | Acceso | Notas |
|---|---|---|---|---|---|
| **DEAP** | 32 | **Sí**: 2 canales (zigomático mayor, trapecio) | Igual que EEG (PAD continuo) | Restringido | Mismo caveat de consistencia de splits entre ramas |
| **MAHNOB-HCI** | 27 | ⚠️ **Por verificar**: la documentación pública estándar de MAHNOB-HCI que conozco incluye EEG (32 ch), ECG, GSR, respiración, temperatura y **video facial/corporal**, pero no recuerdo con certeza canales de electrodos EMG dedicados — la expresión facial ahí se capta por video, no por EMG | Autoreporte + etiquetado externo de arousal/valencia | Restringido | **No lo incluyo en la matriz experimental de EMG hasta confirmar** que el release contiene señal EMG cruda. Alternativa si no la tiene: excluirlo de esta rama (podría eventualmente aportar a FER vía video) |
| **AMIGOS** | 40 | ⚠️ **Por verificar**: el release estándar que conozco es EEG (14 ch Emotiv), ECG, GSR + video facial, sin canal EMG dedicado | Valencia/activación/dominancia continuas + anotación externa | Restringido | Mismo caveat que MAHNOB-HCI |
| **PME4** | — | Desconocido | 7 clases (según la consigna) | Desconocido | No puedo verificar este dataset con la información que tengo. Necesito la referencia/paper o el repositorio de origen antes de incluirlo con parámetros reales |

**Implicación:** de los 4 datasets propuestos para EMG, solo **DEAP** puedo confirmar con seguridad que contiene señal EMG cruda utilizable. Propongo iniciar la rama EMG con DEAP como dataset primario, y tratar MAHNOB-HCI/AMIGOS/PME4 como "pendientes de verificación de disponibilidad de canal" antes de comprometer arquitectura y presupuesto de tiempo en ellos.

### 1.4 PPG

| Dataset | Sujetos | Señal | Etiquetas | Fs | Acceso | Notas |
|---|---|---|---|---|---|---|
| **DEAP** | 32 | Canal de pletismografía (BVP/PPG) | PAD continuo | 512→128 Hz | Restringido | Mismo caveat de consistencia de splits |
| **WESAD** | 15 | PPG/BVP vía Empatica E4 (muñeca) + ECG/EMG/EDA/temp/resp vía RespiBAN (pecho) | **Condiciones**, no emociones básicas: baseline / estrés (TSST) / diversión (amusement) / meditación; + autoreporte PANAS/SAM | E4 BVP 64 Hz, RespiBAN 700 Hz | Moderado (formulario Bosch) | El ground truth es **por condición experimental**, no una taxonomía de emociones discretas — mapeable de forma aproximada a alto/bajo arousal y valencia negativa/positiva, pero debe documentarse como un mapeo, no una equivalencia directa |
| **PPGE** | — | Desconocido | Desconocido | — | Desconocido | Igual que PME4: no puedo verificar este dataset. Pendiente de que aportes la fuente |

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
| FER | RAF-DB | CNN | ResNet | Mejor de B + embedding | Split oficial | ⚠️ Requiere acceso |
| FER | RAF-ML | CNN (softmax+KL) | ResNet (softmax+KL) | Mejor de B + embedding | Split oficial | ⚠️ Requiere acceso; pérdida distinta (LDL) |
| FER | CK+ | CNN (frame apex) | CNN+BiLSTM (secuencia) | Transformer temporal | LOSO agrupado por sujeto/secuencia | ⚠️ Requiere formulario CMU |
| FER | Yale | Transfer learning | — | Estudio de caso, no benchmark | Sin LOSO formal | ⚠️ Uso restringido a análisis cualitativo (ver 1.1) |
| EEG | DEAP | CNN 1D/2D | CNN+BiLSTM | Transformer/Graph (a decidir tras baseline) | LOSO | ⚠️ Requiere EULA |
| EEG | SEED | CNN | CNN+BiLSTM | A decidir | LOSO agrupado por sesión | ⚠️ Requiere acceso BCMI |
| EEG | DREAMER | CNN | CNN+BiLSTM | A decidir | LOSO | ⚠️ Requiere acuerdo de datos |
| EMG | DEAP | MLP (features) | CNN 1D | CNN+LSTM/Transformer | LOSO | ⚠️ Requiere EULA. Dataset primario confirmado para esta rama |
| EMG | MAHNOB-HCI | — | — | — | — | ❌ **Pendiente**: confirmar si el release incluye canal EMG crudo antes de comprometer diseño |
| EMG | AMIGOS | — | — | — | — | ❌ **Pendiente**: mismo caveat |
| EMG | PME4 | — | — | — | — | ❌ **Pendiente**: dataset no verificable con la información actual; necesito referencia/fuente |
| PPG | DEAP | CNN 1D | CNN+BiLSTM | Transformer | LOSO | ⚠️ Requiere EULA |
| PPG | WESAD | CNN 1D | CNN+BiLSTM | Transformer | LOSO | ⚠️ Requiere formulario Bosch; etiqueta = condición, documentar mapeo a V/A |
| PPG | PPGE | — | — | — | — | ❌ **Pendiente**: dataset no verificable, necesito fuente |

**Nota importante:** ningún experimento se "fuerza". Las filas marcadas ❌ no entran a la matriz de ejecución hasta que confirmes fuente/acceso/estructura, tal como pide el punto 21 (no simular resultados, no forzar experimentación sin base).

---

## 7. Datasets que requieren tu confirmación antes de seguir

1. **PME4** — necesito el paper/fuente original o el repositorio de descarga. Sin eso no puedo definir sujetos, fs, canales ni formato real.
2. **PPGE** — mismo caso.
3. **MAHNOB-HCI y AMIGOS para EMG** — necesito que confirmes si tienes acceso a una versión de estos datasets que incluya canal EMG facial crudo (electrodos), distinto del video facial. Si no lo tienen, propongo: (a) excluirlos de la rama EMG, o (b) reasignarlos como fuente adicional de video para la rama FER en una fase posterior (fuera del alcance actual).
4. **Acceso EULA pendiente** en AffectNet, RAF-DB/RAF-ML, CK+, DEAP, SEED, DREAMER, MAHNOB-HCI, AMIGOS, WESAD — asumo que como investigador doctoral ya gestionarás o tienes en trámite estos accesos; el código se construirá para leer desde una ruta local configurable, nunca para descargar automáticamente contenido restringido.

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

---

## 10. Próximos pasos (esperando tu confirmación)

Si apruebas este plan, la siguiente fase generará, **sin ejecutar entrenamiento aún** (porque no hay datos disponibles en este entorno):

1. Estructura de proyecto (`data/`, `preprocessing/`, `models/`, `training/`, `evaluation/`, `experiments/`, `embeddings/`, `checkpoints/`, `results/`, `configs/`, `notebooks/`) con configs YAML por experimento.
2. Módulos de preprocesamiento por modalidad (interfaces + lógica, sobre datos de ejemplo sintéticos solo para pruebas unitarias de forma/shape, nunca como sustituto de resultados reales).
3. Definición de arquitecturas A/B/C por modalidad en PyTorch.
4. Lógica de split LOSO/estratificado con auditoría de leakage automatizada.
5. Scripts de entrenamiento/evaluación parametrizados por config.

**Preguntas abiertas que necesito que resuelvas antes de continuar** (ver sección 7): fuente de PME4 y PPGE, y confirmación de canal EMG en MAHNOB-HCI/AMIGOS.

¿Apruebas este plan experimental tal como está, con las exclusiones/reservas señaladas (Yale como estudio de caso, RAF-ML con pérdida LDL, PME4/PPGE/MAHNOB-EMG/AMIGOS-EMG pendientes), o quieres ajustar algo antes de que genere la estructura de código?
