import React, { useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisId, setAnalysisId] = useState(null);
  const [results, setResults] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);
  
  // Manejar selección de archivo
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Verificar el tipo de archivo
      if (!file.type.match('video.*')) {
        setError('Por favor selecciona un archivo de video.');
        return;
      }
      setSelectedFile(file);
      setResults(null);
      setError(null);
    }
  };

  // Subir el archivo
  const uploadFile = async () => {
    if (!selectedFile) {
      setError('Por favor selecciona un archivo de video primero.');
      return;
    }

    setIsUploading(true);
    setStatusMessage('Subiendo video y analizando...');
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Subir el archivo directamente para análisis
      const response = await fetch(`${API_URL}/upload-video/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error al subir el archivo: ${response.status}`);
      }

      const responseText = await response.text();
      // Eliminar comillas si están presentes
      const cleanId = responseText.replace(/^"|"$/g, '');
      
      setAnalysisId(cleanId);
      setIsUploading(false);
      setIsAnalyzing(true);

      // Verificar el estado del análisis
      checkAnalysisStatus(cleanId);
    } catch (error) {
      console.error('Error al subir el archivo:', error);
      setIsUploading(false);
      setError('Error al subir el archivo: ' + error.message);
    }
  };

  // Verificar estado del análisis
  const checkAnalysisStatus = async (id) => {
    try {
      const cleanId = id.replace(/^"|"$/g, '');
      
      const response = await fetch(`${API_URL}/status/${cleanId}`);
      
      if (!response.ok) {
        throw new Error(`Error al verificar el estado: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'completed') {
        // Análisis completado, obtener resultados
        getResults(cleanId);
      } else if (data.status === 'processing') {
        // Análisis en progreso, verificar nuevamente después de un tiempo
        setTimeout(() => checkAnalysisStatus(cleanId), 2000);
      }
    } catch (error) {
      console.error('Error al verificar el estado del análisis:', error);
      setIsAnalyzing(false);
      setError('Error al verificar el estado: ' + error.message);
    }
  };

  // Obtener resultados
  const getResults = async (id) => {
    try {
      const cleanId = id.replace(/^"|"$/g, '');
      
      const response = await fetch(`${API_URL}/results/${cleanId}`);
      
      if (response.status === 202) {
        // Análisis aún en progreso
        setTimeout(() => checkAnalysisStatus(cleanId), 2000);
        return;
      }
      
      if (!response.ok) {
        throw new Error(`Error al obtener resultados: ${response.status}`);
      }

      const data = await response.json();
      console.log("Datos recibidos:", data);
      
      // Verificar que las imágenes estén presentes en los datos
      if (data.train_cars && data.train_cars.length > 0) {
        let missingImages = false;
        
        // Comprobar cada vagón para asegurarse de que tiene imagen
        data.train_cars.forEach((car, index) => {
          if (!car.image) {
            console.warn(`Advertencia: El vagón #${index} no tiene imagen`);
            missingImages = true;
          }
        });
        
        if (missingImages) {
          console.warn("Algunas imágenes de vagones no están disponibles");
        }
      }
      
      setResults(data);
      setIsAnalyzing(false);
      setStatusMessage('Análisis completado');
    } catch (error) {
      console.error('Error al obtener resultados:', error);
      setIsAnalyzing(false);
      setError('Error al obtener resultados: ' + error.message);
    }
  };

  // Obtener color según nivel de confianza
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return '#4caf50'; // Verde
    if (confidence >= 0.5) return '#ff9800'; // Naranja
    return '#f44336'; // Rojo
  };

  // Cargar imagen específica si no está presente
  const loadCarImage = async (car) => {
    if (car.image) return; // Ya tiene imagen
    
    if (!car.image_id) {
      console.error("Vagón sin ID de imagen");
      return;
    }
    
    try {
      const response = await fetch(`${API_URL}/image/${car.image_id}`);
      if (!response.ok) {
        throw new Error(`Error al cargar imagen: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Actualizar la imagen en el vagón
      if (data.image) {
        const updatedCars = results.train_cars.map(c => {
          if (c.id === car.id) {
            return { ...c, image: data.image };
          }
          return c;
        });
        
        setResults({
          ...results,
          train_cars: updatedCars
        });
      }
    } catch (error) {
      console.error(`Error al cargar imagen para vagón ${car.id}:`, error);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Reporte de Detección de Vagones</h1>
        <p>Sube un video para generar un reporte detallado de vagones (almacenado en MongoDB)</p>
      </header>

      <main>
        <section className="upload-section">
          <div className="file-selection">
            <input
              type="file"
              accept="video/*"
              id="file-upload"
              onChange={handleFileChange}
              className="file-input"
            />
            <div className="file-controls">
              <label htmlFor="file-upload" className="file-button">
                Seleccionar Video
              </label>
              <span className="file-name">
                {selectedFile ? selectedFile.name : 'Ningún archivo seleccionado'}
              </span>
              <button
                className="upload-button"
                onClick={uploadFile}
                disabled={!selectedFile || isUploading || isAnalyzing}
              >
                {isUploading ? 'Subiendo...' : 'Subir y Analizar'}
              </button>
            </div>
          </div>

          {(isUploading || isAnalyzing) && (
            <div className="progress-container">
              <p>{statusMessage}</p>
              <div className="progress-bar-container">
                <div 
                  className="progress-bar"
                  style={{
                    width: '100%',
                    animation: 'pulse 1.5s infinite'
                  }}
                ></div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">{error}</div>
          )}
        </section>

        {results && (
          <div className="results-container">
            <section className="report-header">
              <h2>Reporte de Vagones Detectados</h2>
              <div className="report-summary">
                <div className="summary-item">
                  <span>Total de vagones:</span>
                  <strong>{results.train_cars_count}</strong>
                </div>
                <div className="summary-item">
                  <span>Tiempo de procesamiento:</span>
                  <strong>{results.processing_time.toFixed(2)} segundos</strong>
                </div>
                {results.timestamp && (
                  <div className="summary-item">
                    <span>Fecha del análisis:</span>
                    <strong>{results.timestamp}</strong>
                  </div>
                )}
              </div>
            </section>

            <section className="train-cars-report">
              {results.train_cars_count === 0 ? (
                <div className="no-cars">
                  <p>No se detectaron vagones en el video.</p>
                </div>
              ) : (
                <div className="cars-report-grid">
                  {results.train_cars.map((car) => {
                    // Intentar cargar imagen si no está presente
                    if (!car.image && car.image_id) {
                      loadCarImage(car);
                    }
                    
                    return (
                      <div className="car-report-card" key={car.id}>
                        <div className="car-header">
                          <h3>Vagón #{car.id + 1}</h3>
                          <div className="confidence-badge" 
                               style={{backgroundColor: getConfidenceColor(car.confidence)}}>
                            {(car.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                        
                        <div className="car-image-container">
                          {car.image ? (
                            <img
                              src={`data:image/jpeg;base64,${car.image}`}
                              alt={`Vagón ${car.id}`}
                              className="car-image"
                            />
                          ) : (
                            <div className="loading-image">
                              <p>Cargando imagen...</p>
                            </div>
                          )}
                        </div>
                        
                        <div className="car-details">
                          {car.class_name && (
                            <div className="detail-row">
                              <span>Clase detectada:</span>
                              <strong>{car.class_name}</strong>
                            </div>
                          )}
                          
                          {car.text_detected && (
                            <div className="detail-row">
                              <span>Texto detectado:</span>
                              <strong className="alfanum-value">{car.text_detected}</strong>
                            </div>
                          )}
                          
                          {car.alfanumeric_codes && car.alfanumeric_codes.length > 0 ? (
                            <div className="detail-row">
                              <span>Códigos Alfanuméricos:</span>
                              <div className="codes-list">
                                {car.alfanumeric_codes.map((code, idx) => (
                                  <span key={idx} className="code-tag">{code}</span>
                                ))}
                              </div>
                            </div>
                          ) : (
                            car.text_detected && (
                              <div className="detail-row">
                                <span>Códigos Alfanuméricos:</span>
                                <span className="no-codes">No se detectaron códigos específicos</span>
                              </div>
                            )
                          )}
                          
                          <div className="detail-row">
                            <span>ID en MongoDB:</span>
                            <code className="mongo-id">{car.image_id || car._id}</code>
                          </div>
                          
                          <div className="detail-row">
                            <span>Dimensiones:</span>
                            <strong>{car.position.width} x {car.position.height} px</strong>
                          </div>
                          
                          {car.detected_frame && (
                            <div className="detail-row">
                              <span>Frame de detección:</span>
                              <strong>{car.detected_frame}</strong>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <p>Sistema de Reporte de Vagones con MongoDB</p>
        <p className="small-text">Las imágenes y datos de los vagones se almacenan en MongoDB</p>
      </footer>
    </div>
  );
}

export default App;