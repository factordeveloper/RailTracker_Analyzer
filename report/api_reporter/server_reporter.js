// server.js - Archivo principal del servidor Express
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Conexión a MongoDB
mongoose.connect('mongodb://localhost:27017/train_detection', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('Conexión a MongoDB establecida'))
.catch(err => console.error('Error al conectar a MongoDB:', err));

// Definición de esquemas
const wagonSchema = new mongoose.Schema({
    _id: mongoose.Schema.Types.ObjectId,
    id: Number,
    confidence: Number,
    position: {
        x: Number,
        y: Number,
        width: Number,
        height: Number
    },
    text_detected: String,
    alfanumeric_codes: [String],
    image_id: String,
    image: String,
    analysis_id: String,
    detected_frame: Number,
    class_id: Number,
    class_name: String
});

const reportSchema = new mongoose.Schema({
    _id: mongoose.Schema.Types.ObjectId,
    id: String,
    train_cars_count: Number,
    processing_time: Number,
    timestamp: String,
    fps: Number,
    total_frames: Number,
    interval_used: Number
});

// Definición de modelos
const Wagon = mongoose.model('Wagon', wagonSchema, 'vagones');
const Report = mongoose.model('Report', reportSchema, 'reportes');

// Rutas API
// Obtener reportes con paginación y filtros
app.get('/api/reports', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const skip = (page - 1) * limit;
        
        // Construir filtros
        const filter = {};
        
        // Filtrar por fecha
        if (req.query.dateFrom || req.query.dateTo) {
            filter.timestamp = {};
            
            if (req.query.dateFrom) {
                filter.timestamp.$gte = req.query.dateFrom + ' 00:00:00';
            }
            
            if (req.query.dateTo) {
                filter.timestamp.$lte = req.query.dateTo + ' 23:59:59';
            }
        }
        
        // Filtrar por cantidad de vagones
        if (req.query.minCars) {
            filter.train_cars_count = { $gte: parseInt(req.query.minCars) };
            
            if (req.query.maxCars) {
                filter.train_cars_count.$lte = parseInt(req.query.maxCars);
            }
        }
        
        // Contar total de documentos que coinciden con el filtro
        const total = await Report.countDocuments(filter);
        
        // Obtener reportes
        const reports = await Report.find(filter)
            .sort({ timestamp: -1 })
            .skip(skip)
            .limit(limit);
        
        res.json({
            total,
            page,
            pages: Math.ceil(total / limit),
            reports
        });
    } catch (error) {
        console.error('Error al obtener reportes:', error);
        res.status(500).json({ error: 'Error al obtener reportes' });
    }
});

// Obtener vagones por ID de reporte
app.get('/api/wagons', async (req, res) => {
    try {
        const reportId = req.query.reportId;
        
        if (!reportId) {
            return res.status(400).json({ error: 'Se requiere el ID del reporte' });
        }
        
        // Buscar vagones que pertenecen al reporte
        const wagons = await Wagon.find({ analysis_id: reportId });
        
        res.json(wagons);
    } catch (error) {
        console.error('Error al obtener vagones:', error);
        res.status(500).json({ error: 'Error al obtener vagones' });
    }
});

// Obtener un reporte específico por ID
app.get('/api/reports/:id', async (req, res) => {
    try {
        const report = await Report.findOne({ id: req.params.id });
        
        if (!report) {
            return res.status(404).json({ error: 'Reporte no encontrado' });
        }
        
        res.json(report);
    } catch (error) {
        console.error('Error al obtener el reporte:', error);
        res.status(500).json({ error: 'Error al obtener el reporte' });
    }
});

// Servir archivos estáticos desde el directorio 'public'
app.use(express.static('public'));

// Ruta para servir la aplicación frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Iniciar el servidor
app.listen(PORT, () => {
    console.log(`Servidor corriendo en http://localhost:${PORT}`);
    console.log(`Aplicación frontend disponible en http://localhost:${PORT}`);
    console.log(`API disponible en http://localhost:${PORT}/api`);
});