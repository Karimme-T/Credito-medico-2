# Librerías
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import pdfplumber
import re
import io
import uvicorn 
import os

# CARGA DEL MODELO Y PREPROCESAMIENTO
print("Cargando modelo y preprocesadores...")

try:
    model = keras.models.load_model("modelo_credito_medico.h5")
    preprocess = joblib.load("prepro_credito_medico.pkl")

    imputer_num = preprocess["imputer_num"]
    imputer_cat = preprocess["imputer_cat"]
    label_encoders = preprocess["label_encoders"]
    scaler = preprocess["scaler"]
    categorical_cols = preprocess["categorical_cols"]
    numerical_cols = preprocess["numerical_cols"]
    le_target = preprocess["le_target"]

    if "feature_order" in preprocess:
        feature_order = preprocess["feature_order"]
        all_feature_cols = feature_order
        print(f"✅ Configuración cargada correctamente. Features: {len(all_feature_cols)}")
    else:
        all_feature_cols = numerical_cols + categorical_cols
        print(f"⚠️ Usando orden por defecto.")

except Exception as e:
    print(f"ERROR AL CARGAR MODELOS: {e}")
    print("Asegúrate de que los archivos .h5 y .pkl estén en la misma carpeta.")

# CONFIGURACIÓN DE FASTAPI
app = FastAPI(title="API Crédito Médico", version="1.0")

@app.get("/ping")
def ping():
    return {"status": "ok", "mensaje": "Conexión exitosa React-Python"}

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  MODELOS PYDANTIC 
class Solicitud(BaseModel):
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: float
    Num_of_Loan: int
    Type_of_Loan: str
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Changed_Credit_Limit: float
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Credit_History_Age: float
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Payment_Behaviour: str
    Monthly_Balance: float

@app.get("/")
def root():
    return {"mensaje": "API de scoring de tarjeta médica funcionando"}

# FUNCIONES AUXILIARES (PDFs)
def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text += "\n" + txt
    except Exception as e:
        print(f"Error leyendo PDF: {e}")
    return text

def parse_buro_score_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    m = re.search(r"\b([4-8]\d{2})\b\s*(REGULAR|BUENO|MALO)", text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        segmento = m.group(2).upper()
    else:
        score = None
        segmento = None
        m2 = re.search(r"(MI\s+SCORE|SCORE)[^\d]*([4-8]\d{2})", text, re.IGNORECASE)
        if m2:
            score = int(m2.group(2))
        
        if score is None:
            candidatos = re.findall(r"\b([4-8]\d{2})\b", text)
            candidatos_int = [int(c) for c in candidatos]
            if candidatos_int:
                promedio = sum(candidatos_int) / len(candidatos_int)
                score = min(candidatos_int, key=lambda x: abs(x - promedio))

        if score is not None:
            if score >= 690: segmento = "BUENO"
            elif score >= 620: segmento = "REGULAR"
            else: segmento = "MALO"
            
    return {"buro_score_raw": score, "buro_segmento": segmento}

def parse_buro_detalle_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    lines = text.splitlines()
    salarios = []
    for i, line in enumerate(lines):
        if re.search(r"Salario", line, re.IGNORECASE):
            joined = " ".join(lines[i : i + 4])
            matches = re.findall(r"(\d[\d,\.]*)", joined)
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > 500: salarios.append(val)
                except: continue

    salary_from_buro = float(np.mean(salarios)) if salarios else 0.0
    num_credit_cards = len(re.findall(r"TARJETA\s+DE\s+CR[ÉE]DITO", text, re.IGNORECASE))
    num_loans_non_bank = len(re.findall(r"CR[ÉE]DITOS\s+NO\s+BANCARIOS|MUEBLER[IÍ]A|APARATOS/MUEBLES", text, re.IGNORECASE))
    
    if num_credit_cards == 0: num_credit_cards = 1
    num_of_loan = max(num_loans_non_bank, 0)
    
    num_delayed = 0
    delay_from_due_date = 0
    if re.search(r"ATRASO\s+DE\s+1\s+A\s+89\s+DIAS", text, re.IGNORECASE):
        num_delayed += 1
        delay_from_due_date = max(delay_from_due_date, 60)
    if re.search(r"ATRASO\s+MAYOR\s+A\s+90\s+DIAS|DEUDA\s+SIN\s+RECUPERAR", text, re.IGNORECASE):
        num_delayed += 2
        delay_from_due_date = max(delay_from_due_date, 90)

    saldo_matches = re.findall(r"Saldo\s+actual\s*[:\-]?\s*\$?\s*([\d,]+\.\d+|\d+)", text, re.IGNORECASE)
    outstanding_debt = sum([float(m.replace(",", "")) for m in saldo_matches if m.replace(",", "").replace(".", "").isdigit()])
    
    credit_util_ratio = 30.0 if outstanding_debt > 0 else 0.0
    
    payment_min_amount = "Yes" if num_delayed == 0 else "No"
    payment_behaviour = "High_spent_Small_value_payments" if num_delayed == 0 else "Low_spent_Large_value_payments"
    credit_mix = "Good" if (num_delayed == 0 and outstanding_debt > 0) else "Bad" if num_delayed > 0 else "Standard"

    return {
        "Salary_from_buro": salary_from_buro,
        "Num_Credit_Card": num_credit_cards,
        "Num_of_Loan": num_of_loan,
        "Num_of_Delayed_Payment": num_delayed,
        "Delay_from_due_date": delay_from_due_date,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_util_ratio,
        "Credit_History_Age": 12.0,
        "Payment_of_Min_Amount": payment_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Num_Bank_Accounts": max(1, num_credit_cards + num_of_loan),
        "Type_of_Loan": "Credit-Card" if num_of_loan > 0 else "Not Specified",
        "Credit_Mix": credit_mix,
    }

def parse_estado_cuenta_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    saldo_match = re.search(r"Saldo\s+(final|promedio|actual)\s*:?\s*\$?\s*([\d,]+\.\d+|\d+)", text, re.IGNORECASE)
    monthly_balance = float(saldo_match.group(2).replace(",", "")) if saldo_match else 0.0
    return {"Monthly_Balance": monthly_balance}

# PREPROCESAMIENTO 
def preprocesar_solicitud(s: Solicitud) -> np.ndarray:
    data_dict = s.dict()
    data_dict["Num_Credit_Inquiries"] = 0.0 
    df = pd.DataFrame([data_dict])

    def map_payment(x):
        t = str(x).lower().strip()
        return 1 if t in ["yes", "si", "1"] else 0 if t in ["no", "0"] else -1
    
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].apply(map_payment)

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    for col in all_feature_cols:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Reordenar y escalar
    df_ordered = df[all_feature_cols]
    return scaler.transform(df_ordered.values)

# --- VARIABLES GLOBALES PARA RE-ENTRENAMIENTO ---
CSV_FILE = "datos_acumulados.csv"
UMBRAL_REENTRENAMIENTO = 300

def reentrenar_modelo():
    print(f"\n[BACKGROUND] Iniciando proceso de re-entrenamiento con {UMBRAL_REENTRENAMIENTO} nuevos datos...")
    
    global model 
    
    try:
        # Cargar los datos acumulados
        if not os.path.exists(CSV_FILE):
            print("No se encontró el archivo CSV. Abortando re-entrenamiento.")
            return

        df = pd.read_csv(CSV_FILE)
        
        # Verificar si se tienen datos suficientes
        if len(df) < UMBRAL_REENTRENAMIENTO:
            print(f"Aún faltan datos ({len(df)}/{UMBRAL_REENTRENAMIENTO}). Cancelando.")
            return

        print("Datos cargados. Iniciando preprocesamiento masivo...")

        # PREPARAR LA VARIABLE OBJETIVO (TARGET - Y)
        
        mapa_target = {
            "BUENO": "Good",
            "REGULAR": "Standard",
            "MALO": "Bad"
        }
        
        # Filtración de filas que no tengan un segmento válido 
        df = df[df["buro_segmento"].isin(mapa_target.keys())].copy()
        
        # Convertir BUENO -> Good
        df["target_text"] = df["buro_segmento"].map(mapa_target)
        
        # Codificación a números (0, 1, 2) usando el encoder original del entrenamiento
        # le_target viene de tu joblib cargado al inicio
        y_train = le_target.transform(df["target_text"])
        
        # Convertir a categórico para Keras (One-Hot Encoding implícito o sparse)

        # PREPARAR LAS VARIABLES DE ENTRADA (FEATURES - X)
        
        # Mapeo de Payment_of_Min_Amount 
        def map_payment(x):
            t = str(x).lower().strip()
            if t in ["yes", "si", "1"]: return 1
            elif t in ["no", "0"]: return 0
            else: return -1
            
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].apply(map_payment)

        # Rellenar columna auxiliar si falta
        if "Num_Credit_Inquiries" not in df.columns:
            df["Num_Credit_Inquiries"] = 0.0

        # Codificar columnas categóricas (Usando los encoders originales)
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                # Aplicar transformación: si aparece una categoría nueva, ponemos -1
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Convertir columnas numéricas a float
        for col in all_feature_cols:
            if col not in categorical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = 0.0 # Si falta la columna, rellenar con 0

        # Ordenar columnas exactamente como espera el modelo
        df_ordered = df[all_feature_cols]

        # Escalar los datos (Usando el scaler original)
        X_train = scaler.transform(df_ordered.values)

        print(f"🧠 Entrenando modelo con X shape: {X_train.shape}, y shape: {y_train.shape}...")

        # FINE-TUNING (RE-ENTRENAMIENTO)

        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        # GUARDAR Y ACTUALIZAR

        nombre_nuevo_modelo = "modelo_credito_medico_v2.h5"
        model.save(nombre_nuevo_modelo)
        print(f"Modelo actualizado guardado en: {nombre_nuevo_modelo}")

        # Recargar el modelo en la variable global para que la API use el nuevo
        model = keras.models.load_model(nombre_nuevo_modelo)
        print("API actualizada con el nuevo modelo en caliente.")

        #  LIMPIEZA (ARCHIVAR DATOS USADOS)
        # Renombrar el CSV para que quede vacío y empiece a juntar los siguientes 300
        # se guarda con un timestamp o índice para tener historial
        import time
        archivo_historico = f"datos_procesados_{int(time.time())}.csv"
        os.rename(CSV_FILE, archivo_historico)
        print(f"🧹 Archivo {CSV_FILE} movido a {archivo_historico}. Listo para nuevos datos.")

    except Exception as e:
        print(f"ERROR CRÍTICO DURANTE EL RE-ENTRENAMIENTO: {str(e)}")
        import traceback
        traceback.print_exc()

# ENDPOINT PRINCIPAL (/predict)
@app.post("/predict")
async def predict(
    # Inyección de BackgroundTasks (para el re-entrenamiento)
    background_tasks: BackgroundTasks,

    # Datos del Formulario (Texto)
    nombre: str = Form(...),
    direccion: str = Form(...),
    correo: str = Form(...),
    telefono: str = Form(...),
    rfc: str = Form(...),
    ingresoMensual: float = Form(...),
    ingresoAnual: float = Form(...),
    inversionMensual: float = Form(...),

    # Archivos
    pdfBuro: UploadFile = File(...),
    pdfDetallado: UploadFile = File(...),
    ine: UploadFile = File(...),              
    comprobanteDomicilio: UploadFile = File(...),
    estadoCuenta: Optional[UploadFile] = File(None)
):
    print(f"\nRecibiendo solicitud de: {nombre}")
    
    # CONSTANTE DE CONVERSIÓN (MXN -> USD)
    MXN_TO_USD = 20.5

    try:
        # LEER Y PARSEAR LOS PDFS
        score_bytes = await pdfBuro.read()
        detalle_bytes = await pdfDetallado.read()

        buro_score_info = parse_buro_score_pdf(score_bytes)
        buro_detalle_info = parse_buro_detalle_pdf(detalle_bytes)

        estado_features = {}
        if estadoCuenta is not None:
            estado_bytes = await estadoCuenta.read()
            estado_features = parse_estado_cuenta_pdf(estado_bytes)

        # CONVERSIÓN MONETARIA Y PREPARACIÓN DE VARIABLES

        # Convertimos los inputs de Pesos a Dólares para el Modelo
        monthly_salary_usd = float(ingresoMensual) / MXN_TO_USD
        annual_income_usd = float(ingresoAnual) / MXN_TO_USD
        amount_invested_monthly_usd = float(inversionMensual) / MXN_TO_USD
        
        outstanding_debt_mxn = float(buro_detalle_info.get("Outstanding_Debt", 0.0))
        outstanding_debt_usd = outstanding_debt_mxn / MXN_TO_USD
        
        # Balance Mensual
        monthly_balance_mxn = 0.0
        if "Monthly_Balance" in estado_features:
            monthly_balance_mxn = float(estado_features["Monthly_Balance"])
        monthly_balance_usd = monthly_balance_mxn / MXN_TO_USD
        
        # EMI 
        total_emi_usd = outstanding_debt_usd * 0.05
        
        # Credit Mix (Segmento)
        segmento = buro_score_info.get("buro_segmento")
        if segmento == "BUENO": credit_mix = "Good"
        elif segmento == "REGULAR": credit_mix = "Standard"
        elif segmento == "MALO": credit_mix = "Bad"
        else: credit_mix = str(buro_detalle_info.get("Credit_Mix", "Unknown"))

        # DICCIONARIO DE DATOS (EN USD)
        solicitud_data = {
            "Annual_Income": annual_income_usd,
            "Monthly_Inhand_Salary": monthly_salary_usd,
            "Num_Bank_Accounts": int(buro_detalle_info.get("Num_Bank_Accounts", 1)),
            "Num_Credit_Card": int(buro_detalle_info.get("Num_Credit_Card", 1)),
            "Interest_Rate": 25.0,
            "Num_of_Loan": int(buro_detalle_info.get("Num_of_Loan", 1)),
            "Type_of_Loan": "Credit-Card",
            "Delay_from_due_date": int(buro_detalle_info.get("Delay_from_due_date", 0)),
            "Num_of_Delayed_Payment": int(buro_detalle_info.get("Num_of_Delayed_Payment", 0)),
            "Changed_Credit_Limit": 0.0,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": outstanding_debt_usd,
            "Credit_Utilization_Ratio": float(buro_detalle_info.get("Credit_Utilization_Ratio", 0.0)),
            "Credit_History_Age": float(buro_detalle_info.get("Credit_History_Age", 36.0)),
            "Payment_of_Min_Amount": str(buro_detalle_info.get("Payment_of_Min_Amount", "Yes")),
            "Total_EMI_per_month": total_emi_usd,
            "Amount_invested_monthly": amount_invested_monthly_usd,
            "Payment_Behaviour": str(buro_detalle_info.get("Payment_Behaviour", "Unknown")),
            "Monthly_Balance": monthly_balance_usd,
        }

        # LÓGICA DE RE-ENTRENAMIENTO 
        try:
            # Crear registro para guardar (Features + Target Real del PDF)
            registro_guardar = solicitud_data.copy()
            registro_guardar["buro_segmento"] = segmento 
            registro_guardar["buro_score_raw"] = buro_score_info.get("buro_score_raw")

            # Guardar en CSV
            df_nuevo = pd.DataFrame([registro_guardar])
            escribir_header = not os.path.exists(CSV_FILE)
            df_nuevo.to_csv(CSV_FILE, mode='a', header=escribir_header, index=False)
            
            # Verificar si son 300
            if os.path.exists(CSV_FILE):
                df_acumulado = pd.read_csv(CSV_FILE)
                total_datos = len(df_acumulado)
                print(f"Datos acumulados para re-entrenamiento: {total_datos}/{UMBRAL_REENTRENAMIENTO}")

                if total_datos >= UMBRAL_REENTRENAMIENTO:
                    print("¡Umbral alcanzado! Programando re-entrenamiento en segundo plano...")
                    background_tasks.add_task(reentrenar_modelo)
        
        except Exception as e_save:
            print(f"Error al guardar datos para reentrenamiento (No afecta al usuario): {e_save}")


        # PREDICCIÓN DEL MODELO

        solicitud_obj = Solicitud(**solicitud_data)
        X_scaled = preprocesar_solicitud(solicitud_obj)
        
        proba = model.predict(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = le_target.inverse_transform([pred_idx])[0]


        # RESULTADO Y CONVERSIÓN A MXN
 
        class_probs = {clase: float(prob) for clase, prob in zip(le_target.classes_, proba)}
        
        label_lower = str(pred_label).lower()
        if "good" in label_lower: linea_usd = 60000
        elif "standard" in label_lower: linea_usd = 20000
        else: linea_usd = 5000
        
        linea_mxn = linea_usd * MXN_TO_USD

        print(f"Análisis exitoso. Resultado: {pred_label}, Monto: ${linea_mxn:,.2f} MXN")

        return {
            "mensaje": "Análisis completado exitosamente",
            "monto": linea_mxn,
            "credit_score_predicho": pred_label,
            "probabilidades": class_probs,
            "usuario": nombre
        }

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "mensaje": "Error interno al procesar la solicitud"
        }

# ARRANQUE DEL SERVIDOR 
if __name__ == "__main__":
    # Esto permite correr el archivo directamente con: python main.py
    # Y fuerza a que escuche en todas las direcciones (0.0.0.0)
    print("Iniciando servidor FastAPI en puerto 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)