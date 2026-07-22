from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np, json, os, contextlib, io, threading
from typing import Optional
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "TU_CLAVE_API_DE_GEMINI_AQUI")

GOOGLE_API_KEY = "AIzaSyBJBm707wtScYpolC0pKUQFLWzjZ8Sblus" 

if GOOGLE_API_KEY == "TU_CLAVE_API_DE_GEMINI_AQUI" or not GOOGLE_API_KEY:
    print("Advertencia: La clave API de Gemini no está configurada o es el valor por defecto. Las consultas a Gemini no funcionarán.")
    genai_configured = False
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_configured = True
        print("Info: La clave API de Gemini ha sido configurada.")
    except Exception as e:
        print(f"Error al configurar la clave API de Gemini: {e}")
        genai_configured = False

gemini_model = None
if genai_configured:
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Info: Modelo Gemini 'gemini-2.5-flash' inicializado correctamente.") 
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini: {e}")
        gemini_model = None
else:
    print("Advertencia: No se intentó cargar el modelo Gemini debido a que la clave API no está configurada.")

class TrueDivide(keras.layers.Layer):
    def __init__(self, divisor=255.0, **kwargs):
        super().__init__(**kwargs)
        self.divisor = float(divisor)

    def call(self, inputs):
        return tf.math.truediv(inputs, self.divisor)

    def get_config(self):
        config = super().get_config()
        config.update({"divisor": self.divisor})
        return config

class Normalize1275(tf.keras.layers.Layer):
    """Reproduce la normalización x/127.5 - 1.0 usada por preprocess_input."""
    def call(self, inputs):
        return (tf.cast(inputs, tf.float32) / 127.5) - 1.0

CUSTOM_OBJECTS = {
     "TrueDivide": TrueDivide,
    "Normalize1275": Normalize1275,
}

# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="Uvas API (dos modelos)", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/static/index.html")

# --------------------------------------------------------------------------------------
# Modelos / Config
# --------------------------------------------------------------------------------------
MODELS = {
    "inception": {
        "path": os.getenv("MODEL_INCEPTION", "models/inception.keras"),
        "metrics": os.getenv("METRICS_INCEPTION", "models/history_inception_augmented_20251112_173224.json"),
        "img_size": (256, 256),
        "loaded": None,
        "load_err": None,
    },
    "baseline": {
        "path": os.getenv("MODEL_BASELINE", "models/baseline.keras"),
        "metrics": os.getenv("METRICS_BASELINE", "models/history_baseline_augmented_20251112_200726.json"),
        "img_size": (256, 256),
        "loaded": None,
        "load_err": None,
    },
}

LABELS_PATH = os.getenv("LABELS_PATH", "models/labels.txt")

labels: Optional[list[str]] = None
_labels_err: Optional[str] = None
_model_lock = threading.Lock()

def _load_labels_once():
    global labels, _labels_err
    if labels is not None:
        return
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
        _labels_err = None
    except Exception as e:
        _labels_err = str(e)
        labels = None

def _load_one_model_silent(key: str):
    cfg = MODELS[key]
    if cfg["loaded"] is not None or cfg["load_err"] is not None:
        return
    with _model_lock:
        if cfg["loaded"] is not None or cfg["load_err"] is not None:
            return
        try:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                custom = dict(CUSTOM_OBJECTS)
                cfg["loaded"] = keras.models.load_model(
                    cfg["path"],
                    custom_objects=custom,
                    compile=False,
                    safe_mode=False,
                )
            cfg["load_err"] = None
        except Exception as e:
            cfg["loaded"] = None
            cfg["load_err"] = str(e)

def _preprocess(pil_img: Image.Image, model_key: str):
    cfg = MODELS[model_key]
    img = pil_img.convert("RGB").resize(cfg["img_size"])
    arr = np.array(img).astype("float32")

    return np.expand_dims(arr, axis=0)

# --------------------------------------------------------------------------------------
# Función para obtener detalles de la enfermedad usando Google AI Studio
# --------------------------------------------------------------------------------------
async def get_disease_details_from_gemini(disease_name: str):
    if gemini_model is None:
        return {
            "symptoms": "Gemini no disponible (clave API no configurada o error de inicialización).",
            "cure": "Gemini no disponible (clave API no configurada o error de inicialización).",
            "prevention": "Gemini no disponible (clave API no configurada o error de inicialización)."
        }

    try:
        prompt_symptoms = f"Dame 3-5 síntomas clave de la enfermedad '{disease_name}' en uvas. Responde solo los síntomas separados por comas."
        response_symptoms = await gemini_model.generate_content_async(prompt_symptoms)
        symptoms = response_symptoms.text.strip() if response_symptoms.text else "No disponible."

        prompt_cure = f"Describe brevemente cómo se cura la enfermedad '{disease_name}' en uvas en 2-3 frases."
        response_cure = await gemini_model.generate_content_async(prompt_cure)
        cure = response_cure.text.strip() if response_cure.text else "No disponible."

        prompt_prevention = f"Describe brevemente cómo se previene la enfermedad '{disease_name}' en uvas en 2-3 frases."
        response_prevention = await gemini_model.generate_content_async(prompt_prevention)
        prevention = response_prevention.text.strip() if response_prevention.text else "No disponible."

        return {
            "symptoms": symptoms,
            "cure": cure,
            "prevention": prevention
        }
    except Exception as e:
        print(f"Error al obtener detalles de la enfermedad '{disease_name}' desde Gemini: {e}")
        return {
            "symptoms": f"Error al consultar Gemini: {e}",
            "cure": f"Error al consultar Gemini: {e}",
            "prevention": f"Error al consultar Gemini: {e}"
        }


# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    _load_labels_once()
    statuses = {}
    for k, cfg in MODELS.items():
        statuses[k] = {
            "model_path": cfg["path"],
            "metrics_path": cfg["metrics"],
            "img_size": cfg["img_size"],
            "model_exists": os.path.exists(cfg["path"]),
            "metrics_exists": os.path.exists(cfg["metrics"]),
            "loaded": cfg["loaded"] is not None,
            "load_error": cfg["load_err"],
        }
    return {
        "labels_path": LABELS_PATH,
        "labels_exists": os.path.exists(LABELS_PATH),
        "labels_error": _labels_err,
        "models": statuses,
        "gemini_api_key_set": GOOGLE_API_KEY != "TU_CLAVE_API_DE_GEMINI_AQUI" and bool(GOOGLE_API_KEY),
        "gemini_model_loaded": gemini_model is not None
    }

@app.get("/metrics")
def metrics(model: str = Query("inception", pattern="^(inception|baseline)$")):
    cfg = MODELS[model]
    try:
        with open(cfg["metrics"], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"accuracy": None, "precision_macro": None, "error": str(e), "model": model}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("inception", pattern="^(inception|baseline)$")
):
    _load_labels_once()
    if _labels_err is not None or labels is None:
        return JSONResponse(status_code=503, content={"error": f"Labels not loaded: {_labels_err}"})
    _load_one_model_silent(model)
    cfg = MODELS[model]
    if cfg["load_err"] is not None or cfg["loaded"] is None:
        return JSONResponse(status_code=503, content={"error": f"Model '{model}' not loaded: {cfg['load_err']}"})
    try:
        pil = Image.open(file.file)
        x = _preprocess(pil, model)
        probs = cfg["loaded"].predict(x, verbose=0)[0].tolist()

        max_probability = np.max(probs)
        predicted_class_index = np.argmax(probs)
        predicted_class_name = labels[predicted_class_index]

        confidence_threshold = 0.70 
        confidence_level = "low"
        final_prediction_message = "undefined"

        if max_probability >= confidence_threshold:
            confidence_level = "high"
            final_prediction_message = predicted_class_name
        else:
            confidence_level = "low"
            final_prediction_message = "undefined"

        if max_probability < (1.0 / len(labels)) + 0.10:
             final_prediction_message = "undefined"
             confidence_level = "low"

        disease_details = {
            "symptoms": "No disponible (baja confianza o no determinado)",
            "cure": "No disponible (baja confianza o no determinado)",
            "prevention": "No disponible (baja confianza o no determinado)"
        }
        if confidence_level == "high" and final_prediction_message not in ["undefined", "Healthy"]:
            disease_details = await get_disease_details_from_gemini(final_prediction_message)


        return {
            "model": model,
            "predicted_class": predicted_class_name,
            "max_probability": float(max_probability),
            "confidence_level": confidence_level,
            "final_prediction_message": final_prediction_message,
            "probabilities": {labels[j]: float(p) for j, p in enumerate(probs)},
            "disease_details": disease_details
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e), "model": model})