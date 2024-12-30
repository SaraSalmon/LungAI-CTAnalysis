# LungAI
LungAI - Plataforma de Apoyo para la Detección de Cáncer de Pulmón

# Plataforma AI para Detección de Cáncer de Pulmón

Esta plataforma es una herramienta avanzada impulsada por inteligencia artificial (IA) diseñada para asistir a médicos en la detección y análisis de cáncer de pulmón a partir de imágenes de tomografías computarizadas (CT-Scans). Combina modelos de aprendizaje profundo con visualización de mapas de calor (Grad-CAM) para identificar posibles áreas de interés, y permite generar informes profesionales con predicciones y observaciones.

---

## **Características principales**

### 1. **Clasificación de imágenes con IA**
   - Suba imágenes de tomografías computarizadas (CT-Scans) para su análisis.
   - La IA determina si hay presencia de cáncer con un porcentaje de confianza asociado.
   - Resultados rápidos y fáciles de interpretar para apoyar en la toma de decisiones.

### 2. **Identificación de Tumores con Grad-CAM**
   - Visualice mapas de calor generados por Grad-CAM para localizar áreas relevantes en la imagen que influyen en la predicción del modelo.
   - Ayuda a los médicos a comprender mejor los resultados del modelo y a evaluar posibles regiones tumorales.

### 3. **Generación de informes médicos**
   - Los médicos pueden añadir observaciones y comentarios personalizados.
   - Descargue informes detallados en formato PDF para registro o comunicación con otros especialistas.

### 4. **Propósito educativo y diagnóstico**
   - **Educativo:** Diseñada para demostrar el potencial de la IA en el análisis de imágenes médicas.
   - **Diagnóstico:** Soporte para médicos, no reemplaza la evaluación clínica ni los estudios profesionales.

---

## **Cómo usar la plataforma**

1. **Clasificación de imágenes**
   - Navegue a la pestaña **Análisis de CT-Scans**.
   - Suba una imagen en formato JPG, JPEG o PNG.
   - Revise los resultados del modelo, incluyendo predicción y confianza.

2. **Visualización de Grad-CAM**
   - Navegue a la pestaña **Identificación de Tumores con Grad-CAM**.
   - Suba una imagen para generar un mapa de calor indicando posibles áreas de interés.

3. **Generación de informes**
   - En la pestaña de **Análisis de CT-Scans**, añada observaciones y comentarios.
   - Descargue el informe generado como un archivo PDF.

---

## **Requisitos del sistema**

- **Lenguaje:** Python 3.8 o superior.
- **Librerías necesarias:**
  - `streamlit`
  - `tensorflow`
  - `numpy`
  - `Pillow`
  - `matplotlib`
  - `opencv-python`
  - `reportlab`
- **Hardware recomendado:**
  - CPU de alto rendimiento o GPU para procesar modelos de aprendizaje profundo eficientemente.

---

## **Instalación y configuración**

1. Clone este repositorio:
   ```bash
   git clone https://github.com/tuusuario/plataforma-ai-cancer-pulmon.git
   cd plataforma-ai-cancer-pulmon
   
2. Cree un entorno virtual e instale las dependencias:
 ```bash
   python -m venv env
    source env/bin/activate  # En Windows: env\Scripts\activate
    pip install -r requirements.txt
```
3. Ejecute la aplicación con Streamlit:
   ```bash
   streamlit run app.py
4. Acceda a la plataforma en su navegador en: http://localhost:8501.

---

## Estructura del proyecto

plataforma-ai-cancer-pulmon/
├── app.py                 # Archivo principal de Streamlit
├── modelo_con_gradcam.h5  # Modelo preentrenado de TensorFlow
├── utils/
│   ├── grad_cam.py        # Funciones para calcular y mostrar Grad-CAM
│   ├── pdf_generator.py   # Función para generar informes en PDF
│   └── preprocess.py      # Funciones para preprocesamiento de imágenes
├── assets/
│   ├── logo.png           # Logo de la plataforma
│   └── ejemplo.png        # Imagen de ejemplo
├── README.md              # Descripción del proyecto
└── requirements.txt       # Dependencias del proyecto


---

## Aclaraciones importantes
- Esta plataforma no reemplaza el diagnóstico médico profesional.
- Los resultados del modelo están diseñados para asistir y complementar el juicio clínico, pero siempre deben confirmarse mediante estudios adicionales y evaluación médica.

---
## Contribuciones

¡Tu colaboración es bienvenida! Si tienes sugerencias, ideas o deseas contribuir al proyecto:

1. Haz un fork del repositorio.
2. Crea una rama para tus cambios:
   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```
3. Realiza un pull request describiendo tus cambios.
---
## Créditos

- Desarrollador principal: Sara Salmón Fernández
- Inspiración y objetivo: Crear una plataforma educativa y de soporte para análisis médico con IA.
