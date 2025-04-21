import React, { useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisId, setAnalysisId] = useState(null);
  const [results, setResults] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);
  const [selectedCar, setSelectedCar] = useState(null);
  const [videoError, setVideoError] = useState(false);
  
  // Manejar errores en la carga del video
  const handleVideoError = () => {
    console.error("Error al cargar el video");
    setVideoError(true);
  };
  
  // Intentar recargar el video
  const retryLoadVideo = () => {
    setVideoError(false);
    // Forzar recarga del elemento video
    const videoElement = document.querySelector('.video-player');
    if (videoElement) {
      videoElement.load();
    }
  };

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
      setAnalysisId(null);
      setError(null);
      setSelectedCar(null);
    }
  };

  // Subir el archivo
  const uploadFile = async () => {
    if (!selectedFile) {
      setError('Por favor selecciona un archivo de video primero.');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setStatusMessage('Subiendo video...');
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Simular progreso de carga
      const uploadTimer = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 95) {
            clearInterval(uploadTimer);
            return 95;
          }
          return prev + 5;
        });
      }, 200);

      // Subir el archivo
      const response = await fetch(`${API_URL}/upload-video/`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(uploadTimer);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`Error al subir el archivo: ${response.status}`);
      }

      const responseText = await response.text();
      // Eliminar comillas si están presentes
      const cleanId = responseText.replace(/^"|"$/g, '');
      console.log("ID recibido:", responseText);
      console.log("ID limpio:", cleanId);
      
      setAnalysisId(cleanId);

      // Iniciar análisis
      setIsUploading(false);
      startAnalysis(cleanId);
    } catch (error) {
      console.error('Error al subir el archivo:', error);
      setIsUploading(false);
      setError('Error al subir el archivo: ' + error.message);
    }
  };

  // Iniciar análisis
  const startAnalysis = async (id) => {
    setIsAnalyzing(true);
    setStatusMessage('Analizando video para detectar vagones...');

    try {
      // Asegurarse de que el ID esté limpio
      const cleanId = id.replace(/^"|"$/g, '');
      console.log("Iniciando análisis con ID:", cleanId);
      
      // Iniciar análisis
      const response = await fetch(`${API_URL}/analyze/${cleanId}`);
      
      if (!response.ok) {
        throw new Error(`Error al iniciar el análisis: ${response.status}`);
      }

      // Verificar estado hasta que esté completado
      checkAnalysisStatus(cleanId);
    } catch (error) {
      console.error('Error al iniciar el análisis:', error);
      setIsAnalyzing(false);
      setError('Error al iniciar el análisis: ' + error.message);
    }
  };

  // Verificar estado del análisis
  const checkAnalysisStatus = async (id) => {
    try {
      // Asegurarse de que el ID esté limpio
      const cleanId = id.replace(/^"|"$/g, '');
      console.log("Verificando estado con ID:", cleanId);
      
      const response = await fetch(`${API_URL}/status/${cleanId}`);
      
      if (!response.ok) {
        throw new Error(`Error al verificar el estado: ${response.status}`);
      }

      const data = await response.json();
      console.log("Estado recibido:", data);
      
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
      // Asegurarse de que el ID esté limpio
      const cleanId = id.replace(/^"|"$/g, '');
      console.log("Obteniendo resultados con ID:", cleanId);
      
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
      setResults(data);
      setIsAnalyzing(false);
      setStatusMessage('Análisis completado');
    } catch (error) {
      console.error('Error al obtener resultados:', error);
      setIsAnalyzing(false);
      setError('Error al obtener resultados: ' + error.message);
    }
  };

  // Manejar selección de vagón para ver detalles
  const handleCarSelect = (car) => {
    setSelectedCar(car === selectedCar ? null : car);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Análisis de Video y Detección de Vagones de Tren</h1>
        <p>Sube un video de trenes para analizar y contar vagones</p>
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
                    width: isUploading ? `${uploadProgress}%` : '100%',
                    animation: isAnalyzing ? 'pulse 1.5s infinite' : 'none'
                  }}
                ></div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {results && (
          <div className="results-container">
            <section className="video-info">
              <h2>Información del Video</h2>
              <div className="stats-grid">
                <div className="stat-card">
                  <h3>Duración</h3>
                  <p>{results.video_info.duration.toFixed(2)} segundos</p>
                </div>
                <div className="stat-card">
                  <h3>Frames</h3>
                  <p>{results.video_info.frame_count}</p>
                </div>
                <div className="stat-card">
                  <h3>Resolución</h3>
                  <p>{results.video_info.width} x {results.video_info.height}</p>
                </div>
                <div className="stat-card highlight-card">
                  <h3>Vagones Detectados</h3>
                  <p>{results.train_cars_count}</p>
                </div>
              </div>
              
              {results.processed_video_available && (
                <div className="video-section">
                  <h3>Video Procesado</h3>
                  <p>Video con las detecciones de vagones marcadas:</p>
                  
                  <div className="video-player-container">
                    {videoError ? (
                      <div className="video-error">
                        <p>Error al cargar el video. El video podría no estar listo o hay un problema con el formato.</p>
                        <button onClick={retryLoadVideo} className="retry-button">
                          Intentar de nuevo
                        </button>
                      </div>
                    ) : (
                      <video 
                        className="video-player"
                        controls
                        preload="metadata"
                        playsInline
                        onError={handleVideoError}
                        poster={results.frame_samples.length > 0 ? `data:image/jpeg;base64,${results.frame_samples[0]}` : ''}
                      >
                        <source src={`${API_URL}/stream-video/${results.id}`} type="video/mp4" />
                        Tu navegador no soporta la reproducción de video.
                      </video>
                    )}
                  </div>
                  
                  <div className="video-actions">
                    <a 
                      href={`${API_URL}/download-video/${results.id}`} 
                      className="download-button"
                      download
                    >
                      Descargar Video
                    </a>
                  </div>
                </div>
              )}
            </section>

            <section className="train-cars">
              <h2>Vagones Detectados</h2>
              {results.train_cars_count === 0 ? (
                <div className="no-objects">
                  <p>No se detectaron vagones en el video.</p>
                </div>
              ) : (
                <div className="cars-container">
                  <div className="cars-grid">
                    {results.train_cars.map((car) => (
                      <div 
                        className={`car-card ${selectedCar && selectedCar.id === car.id ? 'selected' : ''}`}
                        key={car.id}
                        onClick={() => handleCarSelect(car)}
                      >
                        <div className="car-img-container">
                          <img
                            src={`data:image/jpeg;base64,${car.image}`}
                            alt={`Vagón ${car.id}`}
                            className="car-thumbnail"
                          />
                        </div>
                        <div className="car-info">
                          <h3>Vagón #{car.id + 1}</h3>
                          {car.text_detected && (
                            <p className="car-text">ID: <span className="alfanum-id">{car.text_detected}</span></p>
                          )}
                          <p className="car-confidence">
                            Confianza: {(car.confidence * 100).toFixed(2)}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {selectedCar && (
                    <div className="car-details">
                      <h3>Detalles del Vagón #{selectedCar.id + 1}</h3>
                      <div className="car-detail-content">
                        <div className="car-image-large">
                          <img
                            src={`data:image/jpeg;base64,${selectedCar.image}`}
                            alt={`Detalle de vagón ${selectedCar.id}`}
                          />
                        </div>
                        <div className="car-metadata">
                          <div className="metadata-item">
                            <span>Identificador alfanumérico:</span>
                            <strong className="alfanum-id">{selectedCar.text_detected || "No detectado"}</strong>
                          </div>
                          <div className="metadata-item">
                            <span>Confianza de detección:</span>
                            <strong>{(selectedCar.confidence * 100).toFixed(2)}%</strong>
                          </div>
                          <div className="metadata-item">
                            <span>Posición en frame:</span>
                            <strong>X: {selectedCar.position.x}, Y: {selectedCar.position.y}</strong>
                          </div>
                          <div className="metadata-item">
                            <span>Dimensiones:</span>
                            <strong>{selectedCar.position.width} x {selectedCar.position.height} px</strong>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>

            <section className="frame-samples">
              <h2>Frames de Muestra</h2>
              {results.frame_samples.length === 0 ? (
                <p>No hay muestras de frames disponibles.</p>
              ) : (
                <div className="frames-grid">
                  {results.frame_samples.map((frame, index) => (
                    <div className="frame-card" key={index}>
                      <img
                        src={`data:image/jpeg;base64,${frame}`}
                        alt={`Frame ${index + 1}`}
                      />
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;