import os
import shutil
import uuid
from typing import List, Dict
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import asyncio
import base64
import pytesseract  # Para OCR
import re

# Crear directorios para almacenar archivos
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Train Car Detection API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes en desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"]  # Importante para streaming
)

# Cargar el modelo YOLO una vez al iniciar la aplicación
# Nota: Puedes usar un modelo más específico o uno fine-tuned para trenes si está disponible
model = YOLO("yolov8n.pt")

# Clases relevantes para trenes en COCO dataset (usado por YOLOv8 por defecto)
# 6: 'bus', 7: 'train', 3: 'car', etc.
TRAIN_RELATED_CLASSES = [7]  # Principalmente la clase 'train'

class TrainCar(BaseModel):
    id: int
    confidence: float
    position: Dict[str, int]  # x, y, width, height
    text_detected: str = ""
    image: str = ""  # Base64 de la imagen recortada del vagón

class DetectionResult(BaseModel):
    id: str
    train_cars_count: int
    train_cars: List[TrainCar]
    frame_samples: List[str]  # Base64 encoded images
    video_info: dict
    processed_video_available: bool = False

@app.post("/upload-video/", response_model=str)
async def upload_video(file: UploadFile = File(...)):
    """
    Subir un archivo de video al servidor
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Use .mp4, .avi, .mov o .mkv")
    
    # Generar un ID único para este análisis
    analysis_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_{file.filename}")
    
    # Guardar el archivo
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return analysis_id

@app.get("/analyze/{analysis_id}")
async def analyze_video(analysis_id: str):
    """
    Analizar un video previamente subido
    """
    # Buscar el archivo por ID
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(analysis_id)]
    if not files:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    video_path = os.path.join(UPLOAD_DIR, files[0])
    
    # Realizar el análisis en segundo plano
    asyncio.create_task(process_video(analysis_id, video_path))
    
    return {"message": "Análisis iniciado", "analysis_id": analysis_id}

@app.get("/results/{analysis_id}", response_model=DetectionResult)
async def get_results(analysis_id: str):
    """
    Obtener los resultados del análisis
    """
    result_path = os.path.join(RESULTS_DIR, f"{analysis_id}_result.json")
    
    if not os.path.exists(result_path):
        # Verificar si el análisis está en progreso
        files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(analysis_id)]
        if files:
            return JSONResponse(status_code=202, content={"message": "Análisis en progreso"})
        else:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    # Leer los resultados del análisis
    with open(result_path, "r") as f:
        import json
        result = json.load(f)
    
    return result

@app.get("/stream-video/{analysis_id}")
async def stream_processed_video(analysis_id: str):
    """
    Streaming del video procesado para reproducción en el navegador
    """
    video_path = os.path.join(RESULTS_DIR, f"{analysis_id}_processed.mp4")
    
    if not os.path.exists(video_path):
        # Verificar si el procesamiento está completo
        result_path = os.path.join(RESULTS_DIR, f"{analysis_id}_result.json")
        if os.path.exists(result_path):
            return JSONResponse(status_code=202, content={"message": "Video procesado no disponible aún"})
        else:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    # Configuración para streaming adecuado
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename=train_detection_{analysis_id}.mp4"
    }
    
    return FileResponse(
        path=video_path, 
        media_type="video/mp4", 
        headers=headers
    )

@app.get("/download-video/{analysis_id}")
async def download_processed_video(analysis_id: str):
    """
    Descargar el video procesado con detecciones marcadas
    """
    video_path = os.path.join(RESULTS_DIR, f"{analysis_id}_processed.mp4")
    
    if not os.path.exists(video_path):
        # Verificar si el procesamiento está completo
        result_path = os.path.join(RESULTS_DIR, f"{analysis_id}_result.json")
        if os.path.exists(result_path):
            return JSONResponse(status_code=202, content={"message": "Video procesado no disponible aún"})
        else:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    return FileResponse(video_path, media_type="video/mp4", filename=f"train_detection_{analysis_id}.mp4")

@app.get("/status/{analysis_id}")
async def check_status(analysis_id: str):
    """
    Verificar el estado del análisis
    """
    result_path = os.path.join(RESULTS_DIR, f"{analysis_id}_result.json")
    
    if os.path.exists(result_path):
        return {"status": "completed", "analysis_id": analysis_id}
    
    # Verificar si el análisis está en progreso
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(analysis_id)]
    if files:
        return {"status": "processing", "analysis_id": analysis_id}
    
    raise HTTPException(status_code=404, detail="Análisis no encontrado")

def extract_text_from_image(img):
    """
    Extraer texto de una imagen usando OCR
    """
    # Preprocesar imagen para mejorar OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Aplicar OCR
    try:
        text = pytesseract.image_to_string(thresh, config='--psm 11 --oem 3')
        # Limpiar el texto
        text = re.sub(r'\W+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error en OCR: {str(e)}")
        return ""

def detect_train_cars_in_frame(frame, frame_index, all_cars):
    """
    Detectar vagones de tren en un frame
    """
    results = model(frame)
    new_cars_detected = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            conf = float(box.conf[0])
            
            # Si la clase es relacionada con trenes (o tiene alta confianza)
            if (cls in TRAIN_RELATED_CLASSES or conf > 0.7):
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crear un ID único para este vagón
                car_id = len(all_cars) + len(new_cars_detected)
                
                # Recortar la región del vagón
                car_img = frame[y1:y2, x1:x2]
                
                # Extraer texto (números o identificadores)
                text = extract_text_from_image(car_img) if car_img.size > 0 else ""
                
                # Codificar la imagen a base64
                _, buffer = cv2.imencode('.jpg', car_img)
                car_img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Crear objeto de vagón
                car = {
                    "id": car_id,
                    "confidence": conf,
                    "position": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "text_detected": text,
                    "image": car_img_base64
                }
                
                new_cars_detected.append(car)
    
    return new_cars_detected

def is_new_car(new_car, existing_cars, threshold=0.3):
    """
    Determinar si un vagón detectado es nuevo o ya existe en nuestra lista
    basado en la superposición de bounding boxes (IoU)
    """
    for car in existing_cars:
        # Calcular superposición entre bounding boxes
        new_box = new_car["position"]
        old_box = car["position"]
        
        # Calcular coordenadas de intersección
        x1 = max(new_box["x"], old_box["x"])
        y1 = max(new_box["y"], old_box["y"])
        x2 = min(new_box["x"] + new_box["width"], old_box["x"] + old_box["width"])
        y2 = min(new_box["y"] + new_box["height"], old_box["y"] + old_box["height"])
        
        # Comprobar si hay intersección
        if x2 < x1 or y2 < y1:
            continue
        
        # Calcular áreas
        intersection = (x2 - x1) * (y2 - y1)
        new_area = new_box["width"] * new_box["height"]
        old_area = old_box["width"] * old_box["height"]
        union = new_area + old_area - intersection
        
        # Calcular IoU
        iou = intersection / union if union > 0 else 0
        
        if iou > threshold:
            # No es un nuevo vagón, es el mismo
            return False
    
    # Es un nuevo vagón
    return True

async def process_video(analysis_id: str, video_path: str):
    """
    Procesar el video para detectar vagones de tren y crear un video con anotaciones
    """
    try:
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Error al abrir el video: {video_path}")
        
        # Obtener información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        video_info = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
        
        # Configurar el escritor de video - intentamos con un formato más compatible para web
        # Muchos navegadores tienen mejor compatibilidad con H.264
        if cv2.__version__ >= '4.0.0':
            # En versiones más recientes de OpenCV, podemos usar mp4v (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Alternativa: 'mp4v'
        else:
            # Fallback para versiones más antiguas
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Más compatible
        
        processed_video_path = os.path.join(RESULTS_DIR, f"{analysis_id}_processed.mp4")
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
        
        # Analizar frames para detectar vagones
        interval = max(1, int(fps/2))  # Analizar cada medio segundo
        frame_samples = []
        train_cars = []
        
        frame_number = 0
        total_frames_processed = 0
        last_detection_frame = 0
        
        # Primera pasada: detectar todos los vagones
        print("Primera pasada: detectando vagones...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames_processed += 1
            
            # Mostrar progreso
            if total_frames_processed % 100 == 0:
                progress = (total_frames_processed / frame_count) * 100
                print(f"Progreso: {progress:.2f}% ({total_frames_processed}/{frame_count})")
            
            # Procesar solo frames en el intervalo especificado para detección
            if frame_number % interval == 0:
                # Detectar vagones en este frame
                new_cars = detect_train_cars_in_frame(frame, frame_number, train_cars)
                
                # Filtrar para añadir solo vagones nuevos (no duplicados)
                for new_car in new_cars:
                    if is_new_car(new_car, train_cars):
                        train_cars.append(new_car)
                        last_detection_frame = frame_number
                
                # Guardar una muestra de frames para visualización
                if len(frame_samples) < 5 and frame_number % (interval * 5) == 0:
                    # Dibujar las detecciones en este frame
                    annotated_frame = frame.copy()
                    
                    # Dibujar bounding boxes
                    for car in train_cars:
                        pos = car["position"]
                        cv2.rectangle(
                            annotated_frame, 
                            (pos["x"], pos["y"]), 
                            (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                            (0, 255, 0), 
                            2
                        )
                        # Escribir ID y texto detectado
                        label = f"ID: {car['id']}"
                        if car["text_detected"]:
                            label += f" - {car['text_detected']}"
                        cv2.putText(
                            annotated_frame, 
                            label, 
                            (pos["x"], pos["y"] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
                    
                    # Convertir la imagen a base64
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    frame_samples.append(img_base64)
            
            frame_number += 1
        
        # Reiniciar el video para la segunda pasada
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        # Segunda pasada: crear el video con anotaciones
        print("Segunda pasada: creando video anotado...")
        frame_number = 0
        total_frames_processed = 0
        
        # Contador para mostrar vagones totales en el video
        train_cars_count = len(train_cars)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames_processed += 1
            if total_frames_processed % 100 == 0:
                progress = (total_frames_processed / frame_count) * 100
                print(f"Generando video: {progress:.2f}% ({total_frames_processed}/{frame_count})")
            
            # Crear una copia del frame para dibujar
            annotated_frame = frame.copy()
            
            # Dibujar todos los vagones detectados
            for car in train_cars:
                pos = car["position"]
                # Dibujar rectángulo verde
                cv2.rectangle(
                    annotated_frame, 
                    (pos["x"], pos["y"]), 
                    (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                    (0, 255, 0), 
                    2
                )
                
                # Escribir ID y texto detectado
                label = f"ID: {car['id'] + 1}"
                if car["text_detected"]:
                    label += f" - {car['text_detected']}"
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (pos["x"], pos["y"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            # Añadir información general en la esquina superior
            cv2.putText(
                annotated_frame,
                f"Vagones detectados: {train_cars_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Escribir el frame al video
            out.write(annotated_frame)
            
            frame_number += 1
        
        # Liberar recursos
        cap.release()
        out.release()
        print(f"Video procesado guardado en: {processed_video_path}")
        
        # Verificamos que el video se haya generado correctamente
        if os.path.exists(processed_video_path) and os.path.getsize(processed_video_path) > 0:
            video_available = True
            print(f"Video generado correctamente: {os.path.getsize(processed_video_path)} bytes")
        else:
            video_available = False
            print("Error: El video no se generó correctamente o está vacío")
        
        # Guardar los resultados
        result = {
            "id": analysis_id,
            "train_cars_count": train_cars_count,
            "train_cars": train_cars,
            "frame_samples": frame_samples,
            "video_info": video_info,
            "processed_video_available": video_available
        }
        
        result_path = os.path.join(RESULTS_DIR, f"{analysis_id}_result.json")
        with open(result_path, "w") as f:
            import json
            json.dump(result, f)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Guardar el error
        error_path = os.path.join(RESULTS_DIR, f"{analysis_id}_error.txt")
        with open(error_path, "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)