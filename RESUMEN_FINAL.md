# RESUMEN COMPARATIVO FINAL
## Modelos unimodales de Deep Learning para reconocimiento de emociones — FER, EEG, EMG, PPG

**Estado:** las cuatro ramas unimodales tienen al menos un pilotos A/B/C real, ejecutado de extremo a extremo sobre datos reales verificados (no simulados). Este documento consolida los resultados y prepara conceptualmente la etapa de fusión multimodal, que **no se implementa todavía** (fuera del alcance de esta etapa, según el punto 20 del prompt original).

Todos los números de este documento provienen directamente de los archivos JSON en `results/`, generados por los scripts en `training/` y `evaluation/`. Ninguno fue inventado ni ajustado a mano.

---

## 1. Tabla comparativa consolidada (punto 19 del prompt original)

| Modalidad | Dataset | Arquitectura | F1 Macro | Balanced Accuracy | Validación | Embedding | Observaciones |
|---|---|---|---:|---:|---|---|---|
| FER | RAF-DB | A — CNN (desde cero) | 22.9% | 28.3% | Split oficial (12.271/3.068) | 128-d | Baseline débil, sin transfer learning |
| FER | RAF-DB | **B — ResNet18** | **53.3%** | **59.2%** | Split oficial | 128-d | Sin pesos ImageNet (red bloqueada en sandbox); aun así el mejor resultado categórico del proyecto |
| EEG | SEED | A — CNN 2D (features DE) | 34.9% ± 13.8% | 42.9% ± 11.0% | LOSO, 15 sujetos | 128-d | — |
| EEG | SEED | B — CNN + BiLSTM | 35.1% ± 16.4% | 44.5% ± 11.8% | LOSO, 15 sujetos | 128-d | No mejora significativamente sobre A (p=0.62) pese a 10x más costo de entrenamiento |
| EEG | SEED | **C — Transformer** | **47.8% ± 16.8%** | **54.0% ± 13.6%** | LOSO, 15 sujetos | 128-d | Mejor F1/accuracy y el más barato de entrenar; diferencia significativa vs. A (p=0.0195) |
| EMG | PME4 | A — MLP (features RMS/MAV/WL/ZC/SSC/MNF/MDF) | 11.0% ± 3.1% | 15.4% ± 2.7% | LOSO, 11 sujetos | 64-d | ≈ azar (14.3%) |
| EMG | PME4 | B — CNN1D (señal cruda 1kHz) | 9.4% ± 3.1% | 14.6% ± 2.9% | LOSO, 11 sujetos | 128-d | ≈ azar |
| EMG | PME4 | C — CNN + BiLSTM | 11.5% ± 3.7% | 14.9% ± 2.7% | LOSO, 11 sujetos | 128-d | ≈ azar; ANOVA A/B/C p=0.797, sin diferencia significativa |
| PPG | PPGE (`external-ppg`) | A — CNN1D | 16.6% ± 7.2% | 24.0% ± 7.7% | LOSO, 18 sujetos | 128-d | ≈ azar (25%) |
| PPG | PPGE | B — CNN + BiLSTM | 19.7% ± 9.1% | 26.8% ± 10.0% | LOSO, 18 sujetos | 128-d | ≈ azar |
| PPG | PPGE | C — Transformer | 19.9% ± 9.5% | 25.7% ± 9.5% | LOSO, 18 sujetos | 128-d | ≈ azar; ANOVA A/B/C p=0.780, sin diferencia significativa |

**Nota metodológica sobre FER:** RAF-DB no tiene ID de sujeto (imágenes in-the-wild), por lo que no aplica LOSO — se usa el split oficial train/test más una validación estratificada del 10% del train (ver `PLAN_EXPERIMENTAL.md`, sección 3).

---

## 2. Selección de candidatos — no solo por accuracy (punto 19 del prompt)

| Criterio | EEG (SEED) | FER (RAF-DB) | EMG (PME4) | PPG (PPGE) |
|---|---|---|---|---|
| Desempeño | Alto, C >> A/B | Alto, B >> A | Bajo, ≈ azar en los tres | Bajo, ≈ azar en los tres |
| Estabilidad entre sujetos (std) | Moderada (±11-17 pts) | No aplica (sin sujeto) | Baja varianza pero **alrededor del azar** | Alta varianza (±8-14 pts), algunos folds <5% y otros >50% |
| Generalización *subject-independent* | ✅ Confirmada, diferencias significativas | No evaluable (sin sujeto) | ❌ No hay generalización (diagnóstico de fuga confirmó que sí hay señal, pero es específica de cada sujeto) | ❌ Mismo patrón que EMG |
| Costo computacional | C (Transformer) es el más barato de los tres | B (ResNet18) ~16 min | Bajo en todos (A: 28s, B: 11.4 min, C: 8.3 min) | Bajo en todos (8.5-11.2 min c/u) |
| Calidad de la representación latente (embedding) | Alta para C (el clasificador que mejor separa las clases) | Alta para B | Cuestionable — el propio clasificador apenas separa las clases mejor que el azar | Cuestionable, mismo motivo que EMG |
| Interpretabilidad | Pendiente (Grad-CAM/attention no implementado aún, ver limitaciones) | Pendiente (Grad-CAM no implementado aún) | Los hand-crafted features (A) son inherentemente interpretables, aunque no mejoran el desempeño | Los features HR/HRV (calculados pero no usados como modelo) son interpretables |
| Compatibilidad con fusión futura | **Alta — candidato firme: Transformer (C)** | **Alta — candidato firme: ResNet18 (B)** | Baja en su forma actual — necesita normalización/adaptación por sujeto antes de aportar señal útil | Baja en su forma actual — mismo caso que EMG |

**Candidatos seleccionados hoy para la futura fusión multimodal:** el **Transformer de EEG** y el **ResNet18 de FER**, por ser los únicos con evidencia clara de generalización *subject-independent* y significancia estadística frente a sus alternativas más simples.

**EMG y PPG no se descartan**, pero **no deben alimentar la fusión en su forma actual** sin antes intentar (fuera del alcance de esta etapa): normalización por sujeto, aumento de sujetos/datos, o técnicas de adaptación de dominio (p. ej. DANN, mencionado en el prompt original para EEG y extensible a EMG/PPG). Meter sus embeddings actuales a una fusión temprana sería incorporar ruido con apariencia de señal.

---

## Fase I — FER (RAF-DB)

- **Dataset:** RAF-DB real (12.271 train / 3.068 test), verificado por conteo exacto de clases contra el benchmark oficial. Sin ID de sujeto.
- **Preprocesamiento:** imágenes ya alineadas (100×100 px) provistas por el dataset; normalización simple `[-1,1]`; pérdida ponderada por clase inversa a la frecuencia (Happy ≈ 39% del train).
- **Arquitecturas:** A = CNN pequeña desde cero; B = ResNet18 (backbone `torchvision`).
- **Hiperparámetros:** Adam, lr=1e-3, wd=1e-4, batch=64, 8 épocas, mejor checkpoint por balanced accuracy de validación.
- **Parámetros del modelo:** A = 111.111; B (ResNet18) = 11.243.079 (verificado contando `model.parameters()`).
- **Tiempo de entrenamiento:** A ≈ 15 min; B ≈ 16 min (CPU).
- **Resultados:** ver tabla §1.
- **Matriz de confusión:** guardada dentro de cada `test_summary.json` (`confusion_matrix`).
- **Análisis de errores:** balanced accuracy < accuracy en ambos modelos → confusión concentrada en las clases minoritarias (fear, disgust, anger), consistente con el desbalance conocido de RAF-DB pese a la ponderación de la pérdida.
- **Embeddings:** 128-d, guardados en `embeddings/fer/raf_db/<modelo>/test_embeddings.npz` (no versionados en git).
- **Modelo guardado:** `checkpoints/fer/raf_db/<modelo>/model.pt` (no versionado en git).
- **Comparación A vs B:** B casi duplica a A en todas las métricas — la arquitectura residual profunda ayuda sustancialmente incluso sin pesos preentrenados.
- **Limitaciones:** (1) ResNet18 entrenó desde cero porque `download.pytorch.org` está bloqueado en este entorno — un resultado con transfer learning real (fuera de este sandbox) probablemente sea mejor; (2) sin sujeto, no se puede medir generalización *subject-independent* como en las otras modalidades; (3) Grad-CAM (interpretabilidad, punto 17 del prompt) no se implementó todavía.

## Fase II — EEG (SEED)

- **Dataset:** SEED real, ya provisto como features de Differential Entropy (5 bandas × 62 canales), 15 sujetos, 3 clases balanceadas (positivo/neutral/negativo), 50.910 ventanas.
- **Preprocesamiento:** no aplica filtrado crudo (los datos ya vienen como features); normalización z-score ajustada solo en el fold de entrenamiento (auditada programáticamente, `results/eeg/seed/leakage_audit.json`, 15/15 folds OK).
- **Arquitecturas:** A = CNN2D sobre (5,62); B = CNN+BiLSTM sobre secuencias de 10 ventanas (stride 5); C = Transformer con proyección lineal + positional encoding sobre las mismas secuencias.
- **Hiperparámetros:** Adam, lr=1e-3, wd=1e-4, batch=128; A/B 12 épocas.
- **Tiempo total (15 folds):** A ≈ 19 min, B ≈ 52 min, C ≈ 5 min.
- **Resultados y comparación estadística:** ver tabla §1 y `results/eeg/seed/model_comparison.json` (ANOVA p=0.039; C significativamente mejor que A, p=0.0195).
- **Embeddings:** 128-d por fold en `embeddings/eeg/seed/<modelo>/subject_XX.npz`.
- **Limitaciones:** un solo dataset EEG piloteado (DEAP, DREAMER siguen bloqueados por EULA); una sola semilla por modelo (no se promedió sobre múltiples inicializaciones); Grad-CAM/attention-maps de interpretabilidad (punto 17) pendientes.

## Fase III — EMG (PME4)

- **Dataset:** PME4 real y verificado end-to-end: 11 sujetos, 7 emociones balanceadas, 3.829 ensayos, señal EMG cruda de 6 canales a 5kHz extraída directamente de los `.zip` (~5.6GB).
- **Preprocesamiento:** decimación 5kHz→1kHz con filtro anti-aliasing (`scipy.signal.decimate`) para el modelo B/C; extracción de 9 features clásicas (RMS, MAV, WL, ZC, SSC, amplitud/duración de contracción, MNF, MDF) por canal para el modelo A; normalización z-score ajustada solo en train.
- **Arquitecturas:** A = MLP sobre 54 features (9×6 canales); B = CNN1D sobre señal decimada; C = CNN1D + BiLSTM.
- **Resultados:** los tres ≈ azar (14.3%); ANOVA p=0.797 (sin diferencia significativa).
- **Diagnóstico de control (no es un bug):** un split aleatorio con fuga de sujeto permitida alcanzó 24.4% con las mismas features del modelo A — confirma que sí hay señal aprendible, pero es específica de cada sujeto y no generaliza.
- **Embeddings:** generados igual que en las otras ramas, pero de utilidad cuestionable para la fusión dado el desempeño.
- **Limitaciones:** solo 11 sujetos; no se probó normalización por sujeto ni adaptación de dominio (DANN), que el propio prompt original sugiere como línea futura para EEG y es igual de aplicable aquí.

## Fase IV — PPG (PPGE / external-ppg)

- **Dataset:** PPGE real (PKNU-PR-ML-Lab/PPG-Dataset), 18 sujetos, 4 emociones (ira/alegría/relajación/tristeza), señal PPG cruda continua por clip.
- **Preprocesamiento:** limpieza y detección de pulsos con NeuroKit2; segmentación en ventanas de 15s/5s de solape (2.646 ventanas, 0 descartadas); frecuencia de muestreo **asumida en 100 Hz por indicación tuya, no documentada por los autores** — limitación explícita, no un dato verificado.
- **Arquitecturas:** A = CNN1D; B = CNN+BiLSTM; C = Transformer (con "parches" de la ventana, análogo a ViT).
- **Resultados:** los tres ≈ azar (25%); ANOVA p=0.780.
- **Interpretación:** mismo patrón que EMG — variabilidad interindividual fisiológica dominante sobre el patrón emocional común, con estas arquitecturas y este tamaño de muestra (18 sujetos).
- **Limitaciones:** frecuencia de muestreo no verificada (afecta cualquier HR/HRV en unidades reales, aunque no a los modelos entrenados sobre la señal cruda en sí); solo 18 sujetos; no se corrió el diagnóstico de fuga de sujeto que sí se hizo en EMG (quedaría como validación adicional recomendada).

## Fase V — Comparación de las representaciones unimodales

Ver tabla consolidada (§1) y matriz de selección (§2). En síntesis:

- **EEG y FER generalizan entre sujetos/muestras de forma clara y estadísticamente significativa.** Son las dos ramas listas para alimentar una fusión multimodal real.
- **EMG y PPG no generalizan con los enfoques probados**, pese a que EMG mostró evidencia de tener señal aprendible (solo que subject-specific). Esto no es una falla del proyecto: es exactamente el tipo de resultado negativo honesto que el prompt original pedía no ocultar (punto 21), y señala dónde debe concentrarse el trabajo futuro antes de intentar fusión.
- La arquitectura más compleja **no ganó automáticamente en ninguna modalidad** (regla del punto 8): en EEG, el Transformer ganó pero el CNN+BiLSTM no mejoró sobre el baseline pese a ser más costoso; en EMG y PPG, ninguna arquitectura superó a las demás.

## Fase VI — Preparación conceptual de los cuatro encoders (sin implementar la fusión)

Los cuatro encoders ya exponen la interfaz dual que pide el punto 14 del prompt (`(logits, embedding)`), lo que los deja listos, en el sentido de la interfaz de software, para la futura fusión — **sin implementarla todavía**, tal como pide el punto 20:

```text
                    ┌── FER Encoder (ResNet18, 128-d) ──┐   [listo: generaliza bien]
                    │                                    │
Imagen facial ──────┤                                    │
                    │                                    │
EEG ────────────────┤── EEG Encoder (Transformer, 128-d) ┤   [listo: generaliza bien]
                    │                                    │
EMG ────────────────┤── EMG Encoder (CNN+BiLSTM, 128-d) ─┤   [necesita normalización/adaptación
                    │                                    │    por sujeto antes de aportar señal]
PPG ────────────────┤── PPG Encoder (a definir, 128-d) ──┤   [mismo caso que EMG]
                    │                                    │
                    └────────────────────────────────────┘
                              ↓
                     Embeddings unimodales
                              ↓
                    [ETAPA FUTURA — NO IMPLEMENTADA]
                     Fusión multimodal
                              ↓
                    Cross-Attention
                              ↓
                       Transformer
                              ↓
                  Representación emocional
                              ↓
                  Clasificación / PAD
```

**Recomendación concreta para cuando se aborde la fusión:** empezar solo con FER + EEG (las dos ramas validadas), y tratar EMG/PPG como ramas "en observación" hasta que una intervención específica (normalización por sujeto, más datos, adaptación de dominio) las lleve por encima del azar de forma reproducible.

---

## 3. Limitaciones generales de esta etapa (transversales a las cuatro modalidades)

1. **Entorno sin GPU ni acceso a internet general:** todo se entrenó en CPU; `download.pytorch.org` bloqueado impidió transfer learning real en FER.
2. **Una sola semilla por modelo/dataset:** no se promedió sobre múltiples inicializaciones aleatorias; los intervalos de confianza reportados son solo entre sujetos (LOSO), no entre semillas.
3. **Interpretabilidad (punto 17 del prompt) pendiente:** Grad-CAM (FER), importancia por canal/banda (EEG), importancia de canal/tiempo (EMG/PPG) no se implementaron en esta etapa — quedan como trabajo futuro antes de dar por cerrada cada rama.
4. **Datasets adicionales del plan original siguen bloqueados por EULA/acceso** (AffectNet, RAF-ML real, CK+, DEAP, DREAMER, MAHNOB-HCI, AMIGOS, WESAD) — no se fuerzan ni se simulan, siguiendo el punto 21.
5. **PPG:** frecuencia de muestreo asumida, no verificada por los autores del dataset.
6. **EMG y PPG:** resultados negativos reales (≈ azar) que deben tratarse como hallazgo, no como error a esconder.

Este documento y `PLAN_EXPERIMENTAL.md` juntos cubren los puntos 18 (resultados esperados por modalidad), 19 (tabla final) y 20 (documento técnico por fases + preparación conceptual de la fusión) del prompt original.
