import os
import uuid
from typing import List, Dict
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import asyncio
import base64
import re
import time
import easyocr
from datetime import datetime
from pymongo import MongoClient
import io
import tempfile

# Configuración de MongoDB
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["train_detection"]
vagones_collection = db["vagones"]
reportes_collection = db["reportes"]

# Crear directorio temporal si es necesario
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="Train Car Detection API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"]
)

# Cargar el modelo YOLO - usar versión pequeña para velocidad
model = YOLO("yolov8n.pt")

# Inicializar EasyOCR (una sola vez)
print("Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)  # Cambiar a True si tienes GPU
print("EasyOCR inicializado con éxito")

# Clases relevantes - ampliar rango para detectar más tipos de vagones
TRAIN_RELATED_CLASSES = [7, 6, 3, 2, 8]  # tren, bus, coche, camión, etc.

class TrainCar(BaseModel):
    id: int
    confidence: float
    position: Dict[str, int]
    text_detected: str = ""
    alfanumeric_codes: List[str] = []
    image: str = ""  # Imagen en base64

class DetectionResult(BaseModel):
    id: str
    train_cars_count: int
    train_cars: List[TrainCar]
    processing_time: float = 0
    timestamp: str = ""

@app.post("/upload-video/", response_model=str)
async def upload_video(file: UploadFile = File(...)):
    """Subir un archivo de video directamente para análisis"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Use .mp4, .avi, .mov o .mkv")
    
    # Generar un ID único para este análisis
    analysis_id = str(uuid.uuid4())
    
    # Guardar temporalmente el archivo para procesarlo
    temp_file_path = os.path.join(TEMP_DIR, f"{analysis_id}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Realizar el análisis en segundo plano
    asyncio.create_task(process_video_mongodb(analysis_id, temp_file_path))
    
    return analysis_id

@app.get("/results/{analysis_id}", response_model=DetectionResult)
async def get_results(analysis_id: str):
    """Obtener los resultados del análisis desde MongoDB"""
    # Buscar el reporte en MongoDB
    reporte = reportes_collection.find_one({"id": analysis_id})
    
    if not reporte:
        # Verificar si el análisis está en progreso
        temp_files = [f for f in os.listdir(TEMP_DIR) if f.startswith(analysis_id)]
        if temp_files:
            return JSONResponse(status_code=202, content={"message": "Análisis en progreso"})
        else:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    # Obtener los vagones relacionados con este análisis
    vagones = list(vagones_collection.find({"analysis_id": analysis_id}))
    
    # Construir el resultado con la estructura esperada por el frontend
    result = {
        "id": reporte["id"],
        "train_cars_count": reporte["train_cars_count"],
        "processing_time": reporte["processing_time"],
        "timestamp": reporte["timestamp"],
        "train_cars": []
    }
    
    # Añadir los vagones, asegurándose de que las imágenes estén incluidas
    for vagon in vagones:
        # Convertir ObjectId a string para serialización JSON
        vagon["_id"] = str(vagon["_id"])
        
        # Asegurarse de que la imagen base64 está presente
        if "image" not in vagon or not vagon["image"]:
            print(f"Advertencia: Vagón {vagon['id']} no tiene imagen")
            vagon["image"] = ""  # Proporcionar un valor predeterminado
        
        result["train_cars"].append(vagon)
    
    return result

@app.get("/status/{analysis_id}")
async def check_status(analysis_id: str):
    """Verificar el estado del análisis"""
    # Buscar el reporte en MongoDB
    reporte = reportes_collection.find_one({"id": analysis_id})
    
    if reporte:
        return {"status": "completed", "analysis_id": analysis_id}
    
    # Verificar si el análisis está en progreso
    temp_files = [f for f in os.listdir(TEMP_DIR) if f.startswith(analysis_id)]
    if temp_files:
        return {"status": "processing", "analysis_id": analysis_id}
    
    raise HTTPException(status_code=404, detail="Análisis no encontrado")

@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """Obtener una imagen de vagón específica desde MongoDB"""
    vagon = vagones_collection.find_one({"image_id": image_id})
    
    if not vagon:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Devolver la imagen en base64
    return JSONResponse(content={"image": vagon["image"]})

def extract_text_with_easyocr(img):
    """Extraer texto de una imagen usando EasyOCR"""
    if img is None or img.size == 0:
        return ""
    
    # Redimensionar para mejorar detección si la imagen es pequeña
    height, width = img.shape[:2]
    if width < 100 or height < 50:
        scale_factor = max(100 / width, 50 / height)
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Preprocesamiento básico
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mejorar contraste para facilitar la detección de texto
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    try:
        # Detectar texto usando EasyOCR
        results = reader.readtext(gray, detail=0, paragraph=False, 
                                  allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.')
        
        # Unir todos los textos detectados
        combined_text = ' '.join(results)
        
        # Limpiar y normalizar texto
        cleaned = combined_text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9\-\. ]', '', cleaned)
        
        return cleaned
    except Exception as e:
        print(f"Error en OCR: {str(e)}")
        return ""

def find_alphanumeric_codes(text):
    """Extraer códigos alfanuméricos de un texto"""
    if not text:
        return []
    
    # Eliminar espacios adicionales y normalizar
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Patrones para códigos alfanuméricos en vagones
    patterns = [
        r'[A-Z0-9]{3,}',          # Códigos alfanuméricos de 3+ caracteres
        r'[A-Z]{1,2}-\d{2,}',     # Formato tipo XX-12345
        r'\d{2,}-[A-Z\d]{2,}',    # Formato tipo 12345-XX
        r'[A-Z]{1,2}\d{3,}',      # Formato tipo AB123
        r'\d{3,}[A-Z]{1,2}'       # Formato tipo 123AB
    ]
    
    found_codes = []
    # Buscar en el texto completo
    for pattern in patterns:
        matches = re.findall(pattern, text)
        found_codes.extend(matches)
    
    # También buscar en las palabras individuales
    words = text.split()
    for word in words:
        if len(word) >= 3 and re.match(r'^[A-Z0-9\-\.]+$', word):
            if word not in found_codes:
                found_codes.append(word)
    
    # Eliminar duplicados y filtrar códigos demasiado cortos
    unique_codes = []
    for code in found_codes:
        if code not in unique_codes and len(code) >= 3:
            unique_codes.append(code)
    
    return unique_codes

def get_class_name(class_id):
    """Obtener el nombre de la clase según su ID"""
    class_names = {
        0: "person",
        1: "bicycle", 
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat"
    }
    return class_names.get(class_id, "unknown")

def detect_train_cars(frame, frame_count, analysis_id, sampling_rate=15):
    """Detectar vagones de tren en un frame y preparar para MongoDB"""
    results = model(frame, conf=0.4)  # Umbral de confianza
    cars_detected = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filtrar por clase y confianza
            if cls in TRAIN_RELATED_CLASSES or conf > 0.6:
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Asegurar que el tamaño es adecuado (filtrar detecciones demasiado pequeñas)
                w, h = x2 - x1, y2 - y1
                min_size = min(frame.shape[0], frame.shape[1]) * 0.02  # 2% del tamaño mínimo
                
                if w < min_size or h < min_size:
                    continue  # Omitir objetos muy pequeños
                
                # Recortar la región del vagón
                car_img = frame[y1:y2, x1:x2]
                
                # Extraer texto solo si tenemos una buena muestra
                text = ""
                if frame_count % sampling_rate == 0:  # Analizar OCR solo cada ciertos frames
                    text = extract_text_with_easyocr(car_img)
                
                # Extraer códigos alfanuméricos
                alfanumeric_codes = find_alphanumeric_codes(text)
                
                # Generar ID único para la imagen
                image_id = f"{analysis_id}_{len(cars_detected)}"
                
                # Comprimir y codificar la imagen a base64
                _, buffer = cv2.imencode('.jpg', car_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                car_img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Crear objeto de vagón
                car = {
                    "id": len(cars_detected),
                    "confidence": conf,
                    "position": {
                        "x": x1,
                        "y": y1,
                        "width": w,
                        "height": h
                    },
                    "text_detected": text,
                    "alfanumeric_codes": alfanumeric_codes,
                    "image_id": image_id,
                    "image": car_img_base64,  # Guardar la imagen en base64
                    "analysis_id": analysis_id,  # Referencia al análisis
                    "detected_frame": frame_count,
                    "class_id": cls,
                    "class_name": get_class_name(cls)
                }
                
                cars_detected.append(car)
    
    return cars_detected

def is_new_car(new_car, existing_cars, threshold=0.3):
    """Determinar si un vagón es nuevo (optimizado)"""
    new_box = new_car["position"]
    
    for car in existing_cars:
        old_box = car["position"]
        
        # Calcular solapamiento
        x1 = max(new_box["x"], old_box["x"])
        y1 = max(new_box["y"], old_box["y"])
        x2 = min(new_box["x"] + new_box["width"], old_box["x"] + old_box["width"])
        y2 = min(new_box["y"] + new_box["height"], old_box["y"] + old_box["height"])
        
        # Comprobar si hay intersección
        if x2 < x1 or y2 < y1:
            continue
        
        # Calcular IoU
        intersection = (x2 - x1) * (y2 - y1)
        new_area = new_box["width"] * new_box["height"]
        old_area = old_box["width"] * old_box["height"]
        union = new_area + old_area - intersection
        iou = intersection / union if union > 0 else 0
        
        if iou > threshold:
            # Si el nuevo tiene texto y el anterior no, actualizar el texto
            if new_car["text_detected"] and not car["text_detected"]:
                car["text_detected"] = new_car["text_detected"]
                car["alfanumeric_codes"] = new_car["alfanumeric_codes"]
            
            # Actualizar confianza si es mayor
            if new_car["confidence"] > car["confidence"]:
                car["confidence"] = new_car["confidence"]
                car["image"] = new_car["image"]  # Actualizar también la imagen
                
            # No es un nuevo vagón
            return False
    
    # Es un nuevo vagón
    return True

async def process_video_mongodb(analysis_id: str, video_path: str):
    """Procesar el video para detectar vagones y guardar en MongoDB"""
    start_time = time.time()
    print(f"Iniciando procesamiento del video {analysis_id}")
    
    try:
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Error al abrir el video: {video_path}")
        
        # Obtener información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular intervalo de muestreo según la duración
        if frame_count > 1000:
            interval = max(5, int(fps/2))  # 2 frames por segundo o cada 5 frames mínimo
        else:
            interval = max(3, int(fps/4))  # 4 frames por segundo o cada 3 frames mínimo
            
        print(f"Analizando cada {interval} frames, total: {frame_count}")
        
        train_cars = []
        frame_number = 0
        
        # Procesar frames a intervalos
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar solo frames en el intervalo
            if frame_number % interval == 0:
                # Detectar vagones
                cars_in_frame = detect_train_cars(frame, frame_number, analysis_id, sampling_rate=interval*2)
                
                # Filtrar duplicados
                for car in cars_in_frame:
                    if is_new_car(car, train_cars):
                        # Reasignar ID para que sea secuencial
                        car["id"] = len(train_cars)
                        car["image_id"] = f"{analysis_id}_{car['id']}"
                        train_cars.append(car)
            
            frame_number += 1
            
            # Mostrar progreso cada 100 frames
            if frame_number % 100 == 0:
                progress = (frame_number / frame_count) * 100
                print(f"Progreso: {progress:.1f}% - Vagones detectados: {len(train_cars)}")
        
        # Liberar recursos
        cap.release()
        
        # Datos para el reporte
        processing_time = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Guardando {len(train_cars)} vagones en MongoDB...")
        
        # Guardar los vagones en MongoDB
        for car in train_cars:
            # Verificar que la imagen esté presente
            if "image" not in car or not car["image"]:
                print(f"Advertencia: Vagón {car['id']} no tiene imagen")
            
            # Insertarlo en la colección
            vagones_collection.insert_one(car)
            print(f"Vagón {car['id']} guardado con ID de imagen: {car['image_id']}")
        
        # Crear y guardar el reporte en MongoDB
        reporte = {
            "id": analysis_id,
            "train_cars_count": len(train_cars),
            "processing_time": processing_time,
            "timestamp": timestamp,
            "fps": fps,
            "total_frames": frame_count,
            "interval_used": interval
        }
        
        reportes_collection.insert_one(reporte)
        
        print(f"Procesamiento completado en {processing_time:.2f} segundos - {len(train_cars)} vagones detectados")
        print(f"Datos guardados en MongoDB - ID del reporte: {analysis_id}")
        
        # Eliminar el archivo temporal
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Archivo temporal eliminado: {video_path}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        
        # Guardar el error en MongoDB
        error_report = {
            "id": analysis_id,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        db["errores"].insert_one(error_report)
        
        # Eliminar el archivo temporal si existe
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)