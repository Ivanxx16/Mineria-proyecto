import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================

st.set_page_config(
    page_title="Predicción del Esfuerzo de Desarrollo",
    layout="wide"
)

st.title("📊 Predicción del Esfuerzo de Desarrollo de Software")

st.markdown("""
### 🎯 Objetivo del sistema
**Apoyar la planificación de proyectos, la estimación de costos y la toma de decisiones
en etapas tempranas del desarrollo de software.**

---

### 🧠 Descripción del modelo de estimación

Este sistema implementa **modelos de Machine Learning entrenados con el dataset ISBSG**
para estimar el **esfuerzo total de desarrollo**, expresado en **horas/persona**.

El modelo aprende patrones reales de proyectos históricos y utiliza como variables principales:
- **Tamaño funcional del sistema (Puntos de Función)**
- **Plataforma de desarrollo**
- **Tipo de lenguaje**
- **Tipo de desarrollo**
- **Sector industrial**

🔎 Está pensado para ser utilizado **antes de iniciar el desarrollo**, cuando aún no existen
estimaciones detalladas, ayudando a reducir la incertidumbre en la planificación inicial.
""")

# ======================================================
# FUNCIÓN DE EVALUACIÓN
# ======================================================

def evaluar(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred)
    }

# ======================================================
# CARGA DE MODELOS
# ======================================================

@st.cache_resource
def cargar_modelos():
    rf = joblib.load("random_forest_optimized.joblib")
    nn = joblib.load("neural_network_optimized.joblib")
    scaler = joblib.load("scaler.joblib")
    return rf, nn, scaler

try:
    rf_model, nn_model, scaler = cargar_modelos()
except Exception as e:
    st.error("❌ Error al cargar los modelos")
    st.exception(e)
    st.stop()

# ======================================================
# CARGA DEL DATASET (AUTOMÁTICA)
# ======================================================

@st.cache_data
def cargar_dataset():
    return pd.read_csv("isbsg_optimizado.csv")

try:
    df_original = cargar_dataset()
    st.success("✅ Dataset ISBSG cargado automáticamente")
except Exception as e:
    st.error("❌ No se pudo cargar el archivo isbsg_optimizado.csv")
    st.exception(e)
    st.stop()

# ======================================================
# PREPARACIÓN DE DATOS
# ======================================================

if "work_effort" not in df_original.columns:
    st.error("❌ El dataset no contiene la columna 'work_effort'")
    st.stop()

df_features = df_original.copy()

for col in ["work_effort", "work_effort_log", "work_effort_winsorized"]:
    if col in df_features.columns:
        df_features = df_features.drop(columns=[col])

X_columns = df_features.columns.tolist()

# ======================================================
# GUÍA PRÁCTICA – PUNTOS DE FUNCIÓN
# ======================================================

st.subheader("📏 ¿Cómo estimar el Tamaño Funcional (Puntos de Función)?")

st.markdown("""
Los **Puntos de Función (PF)** representan la cantidad de funcionalidad que el sistema
ofrece al usuario, **independientemente de la tecnología utilizada**.

Use la siguiente tabla como **referencia práctica** para asignar un valor razonable:
""")

tabla_pf = pd.DataFrame({
    "Tipo de sistema": [
        "Formulario o módulo simple",
        "Aplicación pequeña",
        "Sistema empresarial mediano",
        "Sistema corporativo grande",
        "Plataforma compleja"
    ],
    "Rango típico de PF": [
        "20 – 50",
        "51 – 150",
        "151 – 500",
        "501 – 1,000",
        "> 1,000"
    ],
    "Ejemplo real": [
        "Registro de usuarios, login básico",
        "CRUD con reportes simples",
        "Sistema académico o comercial",
        "ERP, CRM corporativo",
        "Plataforma bancaria o gubernamental"
    ]
})

st.table(tabla_pf)

# ======================================================
# SELECCIÓN DE MODELO
# ======================================================

st.sidebar.header("🤖 Modelo de Predicción")

modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo",
    [
        "Random Forest Optimizado",
        "Red Neuronal (MLP)"
    ],
    index=0
)

# ======================================================
# PREDICCIÓN MANUAL
# ======================================================

st.subheader("🧮 Predicción Manual de Esfuerzo")

tamano = st.number_input(
    "📐 Tamaño funcional (Puntos de Función)",
    min_value=1,
    value=100,
    step=10
)

plataforma_label = st.selectbox(
    "🖥️ Plataforma de desarrollo",
    [
        "MF – Mainframe (grandes sistemas centrales)",
        "MR – Midrange / Servidores medianos",
        "Multi – Arquitectura multi-tier (web, cliente-servidor)",
        "PC – Computadores personales / escritorio",
        "Proprietary – Plataformas propietarias del proveedor",
        "Unknown – No especificado"
    ]
)

lenguaje_label = st.selectbox(
    "💻 Tipo de lenguaje",
    [
        "3GL – Lenguajes tradicionales (Java, C++, Python)",
        "4GL – Lenguajes orientados a negocio (SQL, ABAP)",
        "5GL – Lenguajes de IA y lógica (Prolog, LISP)",
        "APG – Generadores de aplicaciones (Low-code / No-code)",
        "Unknown – No especificado"
    ]
)

tipo_desarrollo_label = st.selectbox(
    "🛠️ Tipo de desarrollo",
    [
        "New Development – Desarrollo nuevo",
        "Re-development - Rediseño completo",
        "Porting - Adaptación técnica",
        "Other",
        "Not Defined"
    ]
)

# ======================================================
# SECTOR INDUSTRIAL (ESPAÑOL → INGLÉS)
# ======================================================

sector_opciones = {
    "Financiero": "Financial",
    "Gobierno": "Government",
    "Industria de servicios": "Service Industry",
    "Manufactura": "Manufacturing",
    "Educación": "Education",
    "Salud y atención médica": "Medical & Health Care",
    "Comercio mayorista y minorista": "Wholesale & Retail",
    "Construcción": "Construction",
    "Comunicaciones": "Communication",
    "Seguros": "Insurance",
    "Logística": "Logistics",
    "Desconocido": "Unknown"
}

sector_es = st.selectbox(
    "🏭 Sector industrial",
    list(sector_opciones.keys())
)

sector = sector_opciones[sector_es]

# ======================================================
# PREDICCIÓN
# ======================================================

plataforma = plataforma_label.split(" – ")[0]
lenguaje = lenguaje_label.split(" – ")[0]
tipo_desarrollo = tipo_desarrollo_label.split(" – ")[0]

if st.button("🔮 Predecir esfuerzo"):

    input_data = pd.DataFrame(
        np.zeros((1, len(X_columns))),
        columns=X_columns
    )

    if "functional_size" in input_data.columns:
        input_data.loc[0, "functional_size"] = tamano

    posibles_columnas = [
        f"platform_{plataforma}",
        f"language_type_{lenguaje}",
        f"development_type_{tipo_desarrollo}",
        f"industry_sector_{sector}"
    ]

    for col in posibles_columnas:
        if col in input_data.columns:
            input_data.loc[0, col] = 1

    input_scaled = scaler.transform(input_data)

    pred_log = (
        rf_model.predict(input_scaled)[0]
        if modelo_seleccionado == "Random Forest Optimizado"
        else nn_model.predict(input_scaled)[0]
    )

    pred_real = np.expm1(pred_log)

    st.success(
        f"🛠️ Esfuerzo estimado: **{int(pred_real):,} horas/persona**"
    )
