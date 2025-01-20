import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
import cv2
from fpdf import FPDF
import os
import datetime

# Configuración de la página
st.set_page_config(page_title="LungAI", page_icon=":lungs:", layout="wide")

# Cargar el modelo
model = tf.keras.models.load_model("/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/modelo_con_gradcam.h5")

# Funciones principales
def page_principal():
    st.title("LungAI - Plataforma de Apoyo para la Detección de Cáncer de Pulmón")
    st.write("""
        Bienvenido a LungAI, una plataforma que utiliza IA para asistir en la detección de cáncer de pulmón.
        Explora las funcionalidades de esta herramienta para el análisis médico de imágenes.
    """)

def page_evaluation():
    st.title("Evaluación Inicial del Cáncer de Pulmón")
    uploaded_file = st.file_uploader("Sube una imagen de un escaneo CT para predecir si contiene señales de cáncer.", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_column_width=True)
        prediction = predict_image(image)
        diagnosis = "Normal" if prediction[0][0] > 0.5 else "Cáncer"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        st.write(f"*Diagnóstico:* {diagnosis}")
        st.write(f"*Confianza:* {confidence * 100:.2f}%")
    else:
        st.warning("Por favor, sube una imagen para continuar.")

def page_tumor_localization():
    st.title("📍Localización de Tumores")
    st.write("Esta funcionalidad está actualmente en desarrollo. ¡Pronto estará disponible!")

def page_tumor_type():
    st.title("Identificación del Tipo Tumoral")
    st.write("Esta funcionalidad está actualmente en desarrollo. ¡Pronto estará disponible!")

def page_help_and_support():
    st.title("Ayuda y Soporte")
    st.write("Consulta tutoriales y obtén respuestas a preguntas frecuentes sobre LungAI.")

# Predicción de imágenes
def predict_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Función para la página principal
def page_principal():
    # Título principal de la plataforma
    st.title("LungAI - Plataforma de Apoyo para la Detección de Cáncer de Pulmón")

    # Breve descripción del propósito de la plataforma
    st.header("Propósito de la Plataforma")
    st.write("""
        Esta plataforma está diseñada para asistir a los profesionales médicos en el análisis de imágenes médicas,
        particularmente en la **detección temprana de cáncer de pulmón** mediante **Deep Learning**. Usando un modelo de IA entrenado con imágenes de escáneres de TAC, la plataforma puede predecir si una imagen muestra signos de cáncer de pulmón, proporcionando resultados con un alto nivel de confianza.
        
        Además de la predicción, la plataforma ofrecerá un enfoque visual mediante el uso de **Grad-CAM**, que resalta las áreas clave de la imagen para facilitar la interpretación del diagnóstico. Este enfoque visual puede ayudar a los médicos a identificar la ubicación exacta del tumor, mejorando la precisión del diagnóstico.
    
        **Características principales:**
        - **Análisis de imágenes médicas (TAC de pulmón) para detección de cáncer**.
        - **Predicciones de presencia de cáncer con un alto nivel de confianza**.
        - **Generación de informes descargables con resultados y recomendaciones**.
        - **Próximos: Generación de mapas de calor con Grad-CAM para identificar la ubicación de tumores**.
        - **Próximos: Identificación de tipo de tumor.
    """)

    # Explicación del proceso
    st.header("Explicación del Proceso")
    st.write("""
        El proceso en la plataforma es sencillo y está diseñado para facilitar su uso por profesionales de la salud:
        
        1. **Subir la imagen**: Carga una imagen de TAC de pulmón desde tu dispositivo.
        2. **Generar predicción**: La plataforma utiliza un modelo de **Deep Learning** para analizar la imagen y predecir si muestra signos de cáncer, mediante Computer Vision.
        3. **Visualizar Grad-CAM**: Puedes activar una opción para visualizar el **mapa de calor Grad-CAM**, que resalta las áreas relevantes de la imagen, mostrando la ubicación exacta de la posible anomalía.
        4. **Generar informe**: Finalmente, el sistema te permite descargar un informe detallado con los resultados de la predicción, el mapa de calor y recomendaciones.
        
        Este flujo de trabajo está diseñado para facilitar la interpretación rápida y precisa de las imágenes médicas, proporcionando a los médicos herramientas útiles para su diagnóstico.
    """)

    # Aclaración sobre la finalidad educativa o diagnóstica del sistema
    st.header("Finalidad Educativa y Diagnóstica")
    st.write("""
        **Importante**: Esta plataforma tiene fines educativos y de demostración. Si bien utiliza **Inteligencia Artificial** para procesar las imágenes y generar predicciones sobre la presencia de cáncer, los resultados deben ser interpretados por profesionales médicos. No se debe utilizar esta herramienta como sustituto de un diagnóstico profesional.
        
        La plataforma está pensada para asistir a los médicos en la toma de decisiones informadas, pero siempre se debe realizar una revisión clínica exhaustiva antes de llegar a una conclusión final. Para obtener un diagnóstico definitivo, consulte con un especialista en radiología o oncología.
    """)

    # Enlaces a estudios relevantes o publicaciones
    st.header("Estudios y Publicaciones Relevantes")
    st.write("""
        Para entender mejor cómo funciona el **Deep Learning** en el análisis de imágenes médicas y la detección de cáncer, te proporcionamos algunos estudios relevantes y recursos:
    
        - [Estudio sobre el uso de Deep Learning para diagnóstico de cáncer en imágenes médicas](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6630141/)
        - [Deep Learning for Lung Cancer Diagnosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8023872/)
        - [Aplicación de Grad-CAM para la visualización de modelos de IA en medicina](https://arxiv.org/abs/1610.02391)
        
        Estos recursos te proporcionarán más información sobre el uso de la **Inteligencia Artificial** y **Grad-CAM** en el análisis de imágenes médicas, ayudando a mejorar la precisión diagnóstica.
    """)


# Función para preprocesar la imagen
def predict_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))  # El tamaño esperado por el modelo
    img_array = np.array(image)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Expandir las dimensiones para que sea compatible con la entrada del modelo
    
    prediction = model.predict(img_array)
    return prediction


# Función para calcular Grad-CAM
def grad_cam(model, image, layer_name="conv5_block3_out"):
    image_array = np.expand_dims(image, axis=0) / 255.0
    
    last_conv_layer = model.get_layer(layer_name)
    
    heatmap_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(image_array)
        class_channel = predictions[:, 0]
    
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap.numpy()


# Función para mostrar Grad-CAM sobre la imagen
def display_grad_cam(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convertir la imagen PIL a un arreglo de OpenCV (BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir RGB a BGR
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlayed_image

# Función para guardar la imagen cargada en disco
def save_uploaded_image(uploaded_file):
    # Crear una carpeta temporal para guardar las imágenes subidas
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Guardar la imagen en el disco
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Guardar el archivo subido en el disco
    
    return image_path

# Función para generar el PDF con los detalles del informe
def generate_pdf(file_name, diagnosis, comments, prediction, confidence, image_path, logo_path="/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/valdecilla_logo.png"):
    # Crear el objeto FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Añadir logo de la clínica
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=10, w=40)  # Ajusta la posición y tamaño del logo
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Logo de la Clínica", ln=True, align="C")

    # Título del informe
    pdf.set_font("Arial", style="B", size=16)
    pdf.ln(20)
    pdf.cell(200, 10, txt="Informe Médico - Diagnóstico de Cáncer de Pulmón", ln=True, align="C")

    # Predicción y confianza
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Diagnóstico: {diagnosis}", ln=True)
    pdf.cell(200, 10, txt=f"Confianza: {confidence * 100:.2f}%", ln=True)


    # Comentarios médicos
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Comentarios del médico: {comments}", align="L")

    # Imagen analizada
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Imagen analizada:", ln=True)

    # Insertar la imagen analizada en el PDF
    if os.path.exists(image_path):
        pdf.ln(10)
        pdf.image(image_path, x=10, w=180)  # Ajusta la posición y tamaño de la imagen

    # Crear nombre del archivo PDF
    report_name = f"Reporte_{file_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_name)

    return report_name

# Función para la sección de Ayuda y Soporte
def page_help_and_support():
    st.title("💬Ayuda y Soporte")
    
    # Tutoriales breves
    st.header("Tutoriales")
    st.write("""
        A continuación, te mostramos un resumen rápido de cómo usar la plataforma LungAI:
        
        1. **Página Principal:** Explora las funcionalidades de la plataforma y revisa información importante sobre el cáncer de pulmón y el Deep Learning.
        2. **Evaluación Inicial del Cáncer de Pulmón:** Sube una imagen de TC y obtén predicciones sobre la existencia de cáncer.
        3. **Localización de Tumores:** Genera un mapa de calor (Grad-CAM) para identificar la ubicación del tumor.
        4. **Identificación del Tipo Tumoral:** Clasifica el tipo de tumor presente según las imágenes médicas analizadas.
        5. **Generación de Informes:** Descarga un informe PDF detallado con las predicciones, mapas de calor y comentarios médicos.
    """)
    st.video("/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/Tutorial.mov")  # Cambia el enlace a tu tutorial si tienes un video

    # Sección de Preguntas Frecuentes (FAQ)
    st.header("Preguntas Frecuentes (FAQ)")
    
    with st.expander("¿Qué tipo de imágenes acepta la plataforma?"):
        st.write("La plataforma acepta imágenes en formato `.jpg`, `.jpeg` y `.png`. Asegúrate de que las imágenes sean claras y estén en alta resolución.")
    
    with st.expander("¿Puedo usar la plataforma para diagnósticos definitivos?"):
        st.write("No, esta plataforma tiene fines educativos y de demostración. Los resultados deben ser interpretados por un profesional médico certificado.")
    
    with st.expander("¿Es necesario registrarse para usar la plataforma?"):
        st.write("Actualmente, no es necesario registrarse. Sin embargo, en el futuro podríamos implementar un sistema de usuario para guardar historiales y facilitar el acceso a informes.")
    
    with st.expander("¿Puedo usar la plataforma desde dispositivos móviles?"):
        st.write("Sí, la plataforma es compatible con navegadores web en dispositivos móviles y de escritorio. Sin embargo, se recomienda usarla en una pantalla grande para mayor comodidad.")

    # Formulario de Contacto
    st.header("Contacto con el Equipo de Soporte")
    st.write("Si tienes dudas, sugerencias o necesitas asistencia técnica, completa el siguiente formulario para contactarnos:")
    
    contact_form = st.form("contact_form")
    with contact_form:
        name = st.text_input("Nombre:")
        email = st.text_input("Correo Electrónico:")
        message = st.text_area("Mensaje:")
        submitted = st.form_submit_button("Enviar")
        
        if submitted:
            if name and email and message:
                st.success("¡Gracias por tu mensaje! Nos pondremos en contacto contigo pronto.")
                # Aquí puedes agregar lógica para enviar el mensaje, como integrarlo con un servicio de correo electrónico
            else:
                st.error("Por favor, completa todos los campos antes de enviar.")

# Diseño con columnas
col1, col2 = st.columns([1, 4])  # Columna 1 para menú, columna 2 para contenido
with col1:
    # Personalización con un contenedor HTML
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.image("/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/image22.jpg", width=150)
    st.markdown("<h2>Menú LungAI</h2>", unsafe_allow_html=True)  # Cambiar "Navegación" por algo más representativo
    page = st.radio(
        "Selecciona una opción:",
        ["Inicio", "Evaluación Inicial", "Localización de Tumores", "Identificación Tipo Tumoral", "💬Ayuda y Soporte"],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Mostrar la página seleccionada en la columna derecha
with col2:
    if page == "Inicio":
        page_principal()
    elif page == "Evaluación Inicial":
        st.markdown("<h2>Clasificador de Cáncer en escáner TAC</h2>", unsafe_allow_html=True)
         # Cargar una imagen
        uploaded_file = st.file_uploader("Sube una imagen de un escaneo CT para predecir si contiene señales de cáncer.", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Leer y mostrar la imagen cargada
                image = Image.open(uploaded_file)  # Usa solo la carga de la imagen sin .convert('RGB')
                image = image.convert('RGB')  # Ahora conviértela a RGB
                st.image(image, caption="Imagen cargada", use_container_width=True)
                
                # Realizar la predicción
                prediction = predict_image(image)
                predicted_class = "Normal" if prediction[0][0] > 0.5 else "Cáncer"
                confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                
                st.write(f"*Predicción:* {predicted_class}")
                st.write(f"*Confianza:* {confidence * 100:.2f}%")
                
                # Caja de comentarios
                st.subheader("Comentarios del médico:")
                comments = st.text_area("Escribe tus observaciones sobre la imagen de TC:", height=150)

                # Generación de archivo PDF cuando el médico hace comentarios
                if uploaded_file and comments:
                    # Guardar temporalmente la imagen subida
                    image_path = save_uploaded_image(uploaded_file)
                    
                    # Generar el informe PDF con la ruta de la imagen
                    report_file = generate_pdf(uploaded_file.name, predicted_class, comments, prediction, confidence, image_path)
                    
                    # Botón para descargar el reporte PDF
                    st.download_button(
                        label="Descargar Reporte PDF",
                        data=open(report_file, "rb").read(),
                        file_name=report_file,
                        mime="application/pdf"
                    )
            
                # Consejos médicos
                    st.write("---")
                    st.write("*Advertencia:* Este sistema es para propósitos educativos y de demostración. Siempre consulte a un médico o radiólogo para un diagnóstico preciso y profesional.")
            
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")
        
        else:
            st.warning("Por favor, sube una imagen para realizar la predicción.")
            
    elif page == "Localización de Tumores":
        st.markdown("<h2 style='text-align: center;'>🛠️⚙️ Esta funcionalidad está actualmente en desarrollo. <br>¡Pronto estará disponible!</h2>", unsafe_allow_html=True)

    # Crear el diseño con columnas
        col1, col2, col3 = st.columns([1, 2, 1])

    # Centrar contenido en la columna central
        with col2:
            st.image("/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/gradcam.png", caption="Ejemplo Grad-CAM", use_container_width=True)


        
            #image = Image.open(uploaded_file).convert('RGB')
            #heatmap = grad_cam(model, np.array(image), layer_name="conv5_block3_out")
            #overlayed_image = display_grad_cam(heatmap, image)
        
            # Mostrar Grad-CAM
            #st.image(overlayed_image, caption="Mapa de calor Grad-CAM", use_container_width=True)

    elif page == "Identificación Tipo Tumoral":
        st.markdown("<h2 style='text-align: center;'>🛠️⚙️ Esta funcionalidad está actualmente en desarrollo. <br>¡Pronto estará disponible!</h2>", unsafe_allow_html=True)

    # Crear el diseño con columnas
        col1, col2, col3 = st.columns([1, 2, 1])

    # Centrar contenido en la columna central
        with col2:
            st.image("/Users/sarasalmon/Desktop/Streamlit/Proyecto_final/image_lungai.webp", caption="Logo LungAI", use_container_width=True)



        #uploaded_file = st.file_uploader("Sube una imagen de un escaneo CT para predecir si contiene señales de cáncer.", type=["jpg", "jpeg", "png"])
        
    elif page == "💬Ayuda y Soporte":
        page_help_and_support()

# Ejecuta la aplicación
#if __name__ == "__main__":
#    main()


