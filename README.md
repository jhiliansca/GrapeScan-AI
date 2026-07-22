**# 🍇 Uvas API — Clasificación de Enfermedades en Hojas de Vida

Esta API permite **detectar enfermedades en hojas de uva** a partir de imágenes, utilizando **redes neuronales entrenadas en TensorFlow/Keras**.  
El proyecto incluye dos modelos de inteligencia artificial (IA) listos para producción:

- 🧠 **Baseline model:** Red neuronal convolucional (CNN) entrenada desde cero.  
- 🔍 **InceptionV3 model:** Modelo preentrenado con *transfer learning* sobre ImageNet, para mayor precisión.

---

## 📁 Estructura del Proyecto

```
uvas-api/
│
├─ app_main.py              ← Servidor FastAPI principal
├─ models/
│   ├─ inceptionv3_model.keras
│   ├─ baseline_model.keras
│   ├─ labels.txt
│   ├─ metrics_inception.json
│   └─ metrics_baseline.json
├─ .venv/                   ← Entorno virtual de Python
└─ requirements.txt
```

---

## ⚙️ Requisitos

- Python **3.10 o superior**
- FastAPI y Uvicorn
- TensorFlow **2.17+** (incluye Keras 3)
- Dependencias adicionales:
  ```bash
  pip install fastapi uvicorn tensorflow pillow python-multipart
  ```

---

## 🚀 Ejecución del Servidor

1. **Activa el entorno virtual:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Inicia el servidor:**
   ```powershell
   uvicorn app_main:app --host 0.0.0.0 --port 8000 --log-level info
   ```

3. **Abre tu navegador en:**
   ```
   http://127.0.0.1:8000
   ```

---

## 🌐 Endpoints Disponibles

### 🔹 **GET /health**

Verifica el estado de los modelos y los archivos asociados.

📍Ejemplo:
```
http://127.0.0.1:8000/health
```

📤 Respuesta:
```json
{
  "labels_exists": true,
  "models": {
    "inception": { "loaded": true },
    "baseline": { "loaded": true }
  }
}
```

---

### 🔹 **GET /metrics**

Devuelve las métricas de rendimiento del modelo (exactitud, precisión, etc).

📍Ejemplo:
```
http://127.0.0.1:8000/metrics?model=inception
```

📤 Respuesta:
```json
{
  "accuracy": 0.97,
  "precision_macro": 0.95
}
```

📘 Parámetro:
| Nombre | Tipo | Valores | Descripción |
|--------|------|----------|--------------|
| `model` | Query | `inception` / `baseline` | Modelo del cual obtener métricas |

---

### 🔹 **POST /predict**

Clasifica una imagen de hoja de vida y devuelve la predicción del modelo seleccionado.

📍Ejemplo:
```
http://127.0.0.1:8000/predict?model=inception
```

📘 Parámetros:
| Nombre | Tipo | Descripción |
|--------|------|--------------|
| `file` | Form-Data | Imagen (JPG o PNG) |
| `model` | Query | `inception` o `baseline` |

📤 Respuesta:
```json
{
  "model": "inception",
  "predicted_class": "BlackRot",
  "probabilities": {
    "BlackMeasles": 0.02,
    "BlackRot": 0.95,
    "HealthyGrapes": 0.01,
    "LeafBlight": 0.02
  }
}
```

💡 **Tip:** Puedes probar el endpoint fácilmente desde la interfaz interactiva:
```
http://127.0.0.1:8000/docs
```

---

## 🧠 Modelos Utilizados

| Modelo | Descripción | Imagen | Tamaño | Precisión estimada |
|--------|--------------|---------|---------|--------------------|
| `baseline_model.keras` | CNN entrenada desde cero | 256x256 | Pequeño | Media |
| `inceptionv3_model.keras` | Modelo InceptionV3 con transfer learning | 299x299 | Grande | Alta |

---

## 🧩 Carpeta `models/`

| Archivo | Descripción |
|----------|--------------|
| `inceptionv3_model.keras` | Modelo principal con InceptionV3 |
| `baseline_model.keras` | Modelo base (red simple) |
| `labels.txt` | Nombres de las clases (ej. Healthy, BlackRot...) |
| `metrics_inception.json` | Métricas de rendimiento del modelo Inception |
| `metrics_baseline.json` | Métricas de rendimiento del modelo baseline |

---

## 💬 Ejemplo de uso (PowerShell)

```powershell
# Predicción con el modelo Inception
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict?model=inception" -Method Post -Form @{ file = Get-Item "C:\imagenes\hoja_uva.jpg" }

# Predicción con el modelo Baseline
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict?model=baseline" -Method Post -Form @{ file = Get-Item "C:\imagenes\hoja_uva.jpg" }
```

---

## 🧾Trabajo de Universidad

- **Autora:** Jhilian Rosario Sejas Cavero  
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)  
- **IA:** TensorFlow / Keras  
- **Dataset:** Imágenes de hojas de uva (sanas y enfermas) recolectadas desde Google Images  
- **Entrenamiento:** Google Colab con GPU  


---

## 🧭 Resumen

| Endpoint | Método | Descripción | Parámetros |
|-----------|---------|--------------|-------------|
| `/health` | GET | Verifica el estado de los modelos | — |
| `/metrics` | GET | Devuelve métricas del modelo | `model=inception` / `baseline` |
| `/predict` | POST | Clasifica una imagen de hoja | `file`, `model` |

---

> ✨ **Uvas API** — Inteligencia Artificial aplicada al diagnóstico agrícola.
**