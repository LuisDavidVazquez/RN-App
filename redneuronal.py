import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import sys
import threading
import queue
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import datetime
from shutil import copy2

# Configurar semillas aleatorias para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Cola global para comunicación entre hilos
log_queue = queue.Queue()

# Función para generar logs con timestamps y niveles
def log(mensaje, nivel="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    prefijos = {
        "INFO": "ℹ️",
        "WARN": "⚠️",
        "ERROR": "❌",
        "SUCCESS": "✅",
        "PROGRESS": "🔄"
    }
    prefijo = prefijos.get(nivel, "ℹ️")
    mensaje_formateado = f"[{timestamp}] {prefijo} {mensaje}"
    print(mensaje_formateado)
    # También poner en la cola para actualización de la GUI
    log_queue.put(mensaje_formateado)
    return mensaje_formateado

# Callback personalizado para mostrar progreso en tiempo real
class LogProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        log("Comenzando entrenamiento...", "INFO")
        
    def on_epoch_begin(self, epoch, logs=None):
        log(f"Iniciando época {epoch+1}/{self.total_epochs}...", "PROGRESS")
        self.epoch_start_time = datetime.datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        # Calcular tiempo transcurrido
        epoch_time = datetime.datetime.now() - self.epoch_start_time
        total_time = datetime.datetime.now() - self.start_time
        
        # Calcular tiempo estimado restante
        if epoch > 0:
            time_per_epoch = total_time.total_seconds() / (epoch + 1)
            remaining_epochs = self.total_epochs - (epoch + 1)
            eta_seconds = time_per_epoch * remaining_epochs
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta = "Calculando..."
        
        # Mostrar métricas de esta época
        mensaje = (f"Época {epoch+1}/{self.total_epochs} completada en {epoch_time.total_seconds():.1f}s - "
                  f"loss: {logs.get('loss'):.4f}, acc: {logs.get('accuracy'):.4f}, "
                  f"val_loss: {logs.get('val_loss'):.4f}, val_acc: {logs.get('val_accuracy'):.4f}")
        
        # Añadir información de tiempo restante
        mensaje += f" - Tiempo restante estimado: {eta}"
        
        log(mensaje, "PROGRESS")
        
    def on_train_end(self, logs=None):
        total_time = datetime.datetime.now() - self.start_time
        log(f"Entrenamiento completado en {total_time}", "SUCCESS")

# Callback para actualizar la barra de progreso
class ProgressBarCallback(Callback):
    def __init__(self, progress_var, epochs):
        super().__init__()
        self.progress_var = progress_var
        self.epochs = epochs
        self.offset = 25  # Empezar desde 25% (después de la evaluación)
    
    def on_epoch_end(self, epoch, logs=None):
        # Actualizar la barra de progreso
        # La parte del entrenamiento representa el 75% del progreso total
        progress = self.offset + ((epoch + 1) / self.epochs * (100 - self.offset))
        self.progress_var.set(progress)

class ConectorClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Conectores - CNN")
        self.root.geometry("1200x800")
        
        log("Iniciando aplicación...")
        
        # Configuración principal
        self.IMG_SIZE = 224  # Tamaño estándar para MobileNetV2
        self.BATCH_SIZE = 16  # Reducido para mejor estabilidad
        self.EPOCHS = 100     # Aumentado a 200 para entrenamientos más completos
        self.CV_EPOCHS = 10   # Épocas para validación cruzada
        self.FT_EPOCHS = 30   # Épocas para fine-tuning
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame izquierdo
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5)
        
        # Frame derecho
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5)
        
        # Botones - Primera fila
        button_frame1 = ttk.Frame(left_frame)
        button_frame1.grid(row=0, column=0, pady=2)
        ttk.Button(button_frame1, text="Cargar Dataset", command=self.cargar_dataset).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame1, text="Equilibrar Clases", command=self.equilibrar_clases).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame1, text="Entrenar Modelo", command=self.entrenar_modelo).grid(row=0, column=2, padx=2)
        
        # Botones - Segunda fila
        button_frame2 = ttk.Frame(left_frame)
        button_frame2.grid(row=1, column=0, pady=2)
        ttk.Button(button_frame2, text="Probar Imagen", command=self.probar_imagen).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame2, text="Cargar Mejor Modelo", command=self.cargar_mejor_modelo).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame2, text="Evaluar Modelo", command=self.evaluar_modelo).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame2, text="Fine-Tuning Avanzado", command=self.fine_tuning_avanzado).grid(row=0, column=3, padx=2)
        ttk.Button(button_frame2, text="Limpiar", command=self.limpiar_resultados).grid(row=0, column=4, padx=2)
        
        # Barra de progreso para mostrar avance
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, sticky=tk.EW, pady=(5, 0))
        
        # Etiqueta para mostrar estado actual
        self.status_var = tk.StringVar(value="Listo")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        # Área de texto para logs
        self.text_area = scrolledtext.ScrolledText(left_frame, width=60, height=35)
        self.text_area.grid(row=5, column=0, pady=5)
        
        # Canvas para gráficas
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.grid(row=0, column=0)
        
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, pady=5)
        
        # Variables de estado
        self.dataset_path = None
        self.model = None
        self.class_names = []
        self.best_model_path = 'mejor_modelo.h5'
        self.current_thread = None
        self.is_processing = False
        
        # Iniciar el actualizador de UI
        self.configure_ui_updater()
        
        # Redirección de stdout
        self.stdout_original = sys.stdout
        self.redirect_stdout()
        
        log("Interfaz gráfica inicializada correctamente")
        log("Esperando acción del usuario...")
        
    def configure_ui_updater(self):
        """Configura el actualizador de UI para procesar mensajes de la cola"""
        def update_ui():
            try:
                # Procesar hasta 100 mensajes por actualización para evitar bloqueos
                for _ in range(100):
                    if log_queue.empty():
                        break
                    mensaje = log_queue.get_nowait()
                    self.text_area.insert(tk.END, mensaje + '\n')
                    self.text_area.see(tk.END)
                    log_queue.task_done()
            except queue.Empty:
                pass
            finally:
                # Programar la próxima actualización
                self.root.after(100, update_ui)
        
        # Iniciar el actualizador
        self.root.after(100, update_ui)

    def redirect_stdout(self):
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, str):
                log_queue.put(str.strip())

            def flush(self):
                pass

        sys.stdout = StdoutRedirector(self.text_area)
        
    def run_in_thread(self, func, success_message=None):
        """Ejecuta una función en un hilo separado para no bloquear la UI"""
        if self.is_processing:
            log("Hay un proceso en ejecución. Espere a que termine.", "WARN")
            return False
            
        def thread_func():
            self.is_processing = True
            self.status_var.set("Procesando...")
            try:
                func()
                if success_message:
                    log(success_message, "SUCCESS")
            except Exception as e:
                log(f"Error en el proceso: {str(e)}", "ERROR")
                import traceback
                log(traceback.format_exc(), "ERROR")
            finally:
                self.is_processing = False
                self.status_var.set("Listo")
                self.progress_var.set(0)
        
        # Crear y comenzar el hilo
        self.current_thread = threading.Thread(target=thread_func)
        self.current_thread.daemon = True
        self.current_thread.start()
        return True

    def cargar_dataset(self):
        log("Iniciando carga de dataset...")
        self.dataset_path = filedialog.askdirectory(title="Seleccionar carpeta del dataset")
        if self.dataset_path:
            log(f"Ruta del dataset seleccionada: {self.dataset_path}")
            log("Escaneando clases disponibles...")
            
            self.class_names = [d for d in os.listdir(self.dataset_path) 
                              if os.path.isdir(os.path.join(self.dataset_path, d))]
            log(f"Dataset cargado. Clases encontradas: {self.class_names}")
            
            # Mostrar algunas estadísticas
            total_images = 0
            log("Contando imágenes por clase:")
            for clase in self.class_names:
                clase_path = os.path.join(self.dataset_path, clase)
                n_images = len([f for f in os.listdir(clase_path) 
                              if os.path.isfile(os.path.join(clase_path, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                log(f"Clase {clase}: {n_images} imágenes")
                total_images += n_images
            log(f"Total de imágenes en el dataset: {total_images}")
            log("Dataset cargado correctamente. Listo para entrenar.", "SUCCESS")
        else:
            log("Operación cancelada: No se seleccionó ningún directorio", "WARN")

    def crear_modelo(self):
        log("Creando arquitectura del modelo...")
        log("Cargando MobileNetV2 pre-entrenado como modelo base...")
        
        # Usar MobileNetV2 como base
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        
        log("Configurando fine-tuning: congelando solo capas iniciales...")
        # Congelar solo las primeras capas (75%) y descongelar las capas superiores
        for layer in base_model.layers[:int(len(base_model.layers) * 0.75)]:
            layer.trainable = False
        for layer in base_model.layers[int(len(base_model.layers) * 0.75):]:
            layer.trainable = True
            
        log(f"Configuración de capas: {len(base_model.layers) - int(len(base_model.layers) * 0.75)} capas superiores entrenables")
        
        log("Añadiendo capas personalizadas para clasificación...")
        # Añadir capas personalizadas más complejas con dropout para regularización
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(self.class_names), activation='softmax')(x)
        
        # Crear modelo final
        model = Model(inputs=base_model.input, outputs=predictions)
        
        log("Compilando modelo con tasa de aprendizaje reducida...")
        # Compilar con tasa de aprendizaje reducida para mayor estabilidad
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        log(f"Modelo creado exitosamente con {len(self.class_names)} clases de salida", "SUCCESS")
        return model

    def entrenar_modelo(self):
        # Verificar si podemos iniciar
        if not self.dataset_path:
            log("Error: Primero debes cargar el dataset", "ERROR")
            return
        
        # Lanzar el entrenamiento en un hilo
        self.run_in_thread(self._train_model_thread, "Entrenamiento completado con éxito")
    
    def _train_model_thread(self):
        """Función que ejecuta el entrenamiento en un hilo separado"""
        log("Iniciando proceso de entrenamiento...")
        log("Configurando generadores de datos con aumento de datos avanzado...")
        
        # Generadores de datos con data augmentation más agresivo
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,  # Conectores no suelen aparecer invertidos verticalmente
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,  # Añadir transformación de cizallamiento
            fill_mode='nearest'
        )
        
        log("Creando generador de entrenamiento...")
        # Generadores de entrenamiento y validación
        train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42  # Fijar semilla para reproducibilidad
        )
        
        log("Creando generador de validación...")
        validation_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=42  # Fijar semilla para reproducibilidad
        )
        
        # Calcular pesos de clase para manejar desbalances
        log("Calculando pesos de clase para manejar desbalances...")
        class_counts = train_generator.classes
        total_samples = len(class_counts)
        n_classes = len(self.class_names)
        
        # Contar instancias por clase
        class_weights = {}
        for i in range(n_classes):
            count = np.sum(class_counts == i)
            weight = total_samples / (n_classes * count) if count > 0 else 1.0
            class_weights[i] = weight
            log(f"Clase {self.class_names[i]}: {count} muestras, peso = {weight:.2f}")
        
        log("Configurando callbacks para el entrenamiento...")
        # Callbacks para mejor entrenamiento
        callbacks = [
            # Guardar el mejor modelo
            ModelCheckpoint(
                self.best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Detener si no hay mejora
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,  # Aumentado para permitir más intentos (ajustado para 200 épocas)
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir learning rate si se estanca
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # Reducción más agresiva
                patience=15,  # Más paciencia antes de reducir (ajustado para 200 épocas)
                min_lr=1e-7,  # Mínimo más bajo
                verbose=1
            ),
            # Callback para mostrar progreso en tiempo real
            LogProgressCallback(self.EPOCHS),
            # Callback para actualizar la barra de progreso
            ProgressBarCallback(self.progress_var, self.EPOCHS)
        ]
        
        # Crear y entrenar modelo
        log("Creando arquitectura del modelo...")
        self.model = self.crear_modelo()
        
        log("\n--- CONFIGURACIÓN DEL ENTRENAMIENTO ---")
        log(f"Tamaño de imágenes: {self.IMG_SIZE}x{self.IMG_SIZE}")
        log(f"Tamaño de batch: {self.BATCH_SIZE}")
        log(f"Número máximo de épocas: {self.EPOCHS}")
        log(f"Clases a clasificar: {len(self.class_names)}")
        log(f"Imágenes de entrenamiento: {train_generator.samples}")
        log(f"Imágenes de validación: {validation_generator.samples}")
        log("Aumentado de datos: Rotación, desplazamiento, zoom, brillo, cizallamiento")
        log(f"Usando pesos de clase para compensar desbalances")
        log("Validación cruzada integrada: Se ejecutará automáticamente al finalizar para validar robustez del modelo")
        log("----------------------------------------")
        
        log("\nIniciando entrenamiento principal...")
        log("El entrenamiento se detendrá automáticamente si no hay mejora en 25 épocas")
        log("Se guardará automáticamente la versión del modelo con mejor desempeño")
        
        # Registrar tiempo de inicio
        tiempo_inicio = datetime.datetime.now()
        log(f"Inicio de entrenamiento: {tiempo_inicio.strftime('%H:%M:%S')}")
        
        # Entrenar con pesos de clase
        history = self.model.fit(
            train_generator,
            epochs=self.EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Registrar tiempo de finalización
        tiempo_fin = datetime.datetime.now()
        duracion = tiempo_fin - tiempo_inicio
        log(f"Fin de entrenamiento principal: {tiempo_fin.strftime('%H:%M:%S')}")
        log(f"Duración total: {duracion}")
        
        # Cargar el mejor modelo guardado
        if os.path.exists(self.best_model_path):
            log("\nCargando el mejor modelo guardado...")
            self.model = load_model(self.best_model_path)
            
            # Copiar el mejor modelo a un archivo con nombre fijo para el despliegue web
            try:
                log("Copiando el mejor modelo para despliegue web...")
                copy2(self.best_model_path, 'mejor_modelo_ft.h5')
                
                # Verificar las fechas de los archivos
                tiempo_modelo = os.path.getmtime(self.best_model_path)
                tiempo_modelo_ft = os.path.getmtime('mejor_modelo_ft.h5')
                
                # Convertir timestamps a fechas legibles
                fecha_modelo = datetime.datetime.fromtimestamp(tiempo_modelo).strftime('%Y-%m-%d %H:%M:%S')
                fecha_modelo_ft = datetime.datetime.fromtimestamp(tiempo_modelo_ft).strftime('%Y-%m-%d %H:%M:%S')
                
                log(f"Modelo copiado exitosamente como mejor_modelo_ft.h5", "SUCCESS")
                log(f"Fecha de {self.best_model_path}: {fecha_modelo}")
                log(f"Fecha de mejor_modelo_ft.h5: {fecha_modelo_ft}")
                
                # Verificar tamaños de archivo para confirmar que son idénticos
                size_modelo = os.path.getsize(self.best_model_path)
                size_modelo_ft = os.path.getsize('mejor_modelo_ft.h5')
                
                if size_modelo == size_modelo_ft:
                    log(f"Verificación completa: Ambos archivos tienen el mismo tamaño ({size_modelo/1024/1024:.2f} MB)", "SUCCESS")
                else:
                    log(f"¡Advertencia! Los archivos tienen tamaños diferentes: {size_modelo/1024/1024:.2f} MB vs {size_modelo_ft/1024/1024:.2f} MB", "WARN")
            except Exception as e:
                log(f"Error al copiar el modelo: {str(e)}", "ERROR")
            
            # Evaluar el modelo y mostrar matriz de confusión
            log("Evaluando desempeño del mejor modelo...")
            self._evaluate_model_thread()
        
        # Graficar resultados del entrenamiento
        log("Generando gráficas de entrenamiento...")
        self.plot_training_history(history)
        
        # Iniciar validación cruzada automáticamente
        log("\n=== INICIANDO VALIDACIÓN CRUZADA AUTOMÁTICA ===")
        log("Este proceso evalúa la robustez del modelo a través de múltiples divisiones de datos")
        log("La validación cruzada es esencial para verificar que el modelo generaliza correctamente")
        
        self._cross_validation_thread()
        
        log("\n¡Entrenamiento completo con validación cruzada terminado!", "SUCCESS")
        log(f"El mejor modelo ha sido guardado en: {self.best_model_path}", "SUCCESS")

    def evaluar_modelo(self):
        """Inicia la evaluación del modelo en un hilo separado."""
        if not self.model or not self.dataset_path:
            log("Error: Necesitas tener un modelo cargado y un dataset", "ERROR")
            return
            
        self.run_in_thread(self._evaluate_model_thread, "Evaluación completada con éxito")
            
    def _evaluate_model_thread(self):
        """Ejecuta la evaluación del modelo en un hilo separado."""
        log("\nIniciando evaluación detallada del modelo...")
        
        # Crear generador para evaluación
        log("Preparando imágenes para evaluación...")
        datagen = ImageDataGenerator(rescale=1./255)
        eval_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # Importante: no mezclar para mantener el orden
        )
        
        # Realizar predicciones
        total_batches = int(np.ceil(eval_generator.samples / eval_generator.batch_size))
        log(f"Realizando predicciones sobre {eval_generator.samples} imágenes...")
        
        # Función para predecir con retroalimentación de progreso
        predictions_list = []
        for i in range(total_batches):
            # Actualizar barra de progreso
            progress = (i + 1) / total_batches * 100
            self.progress_var.set(progress)
            self.status_var.set(f"Evaluando... {i+1}/{total_batches} batches")
            
            # Obtener batch y predecir
            batch_x, _ = eval_generator[i]
            batch_preds = self.model.predict(batch_x, verbose=0)
            predictions_list.append(batch_preds)
            
            # Mostrar progreso cada 5 batches o en el último
            if (i + 1) % 5 == 0 or i == total_batches - 1:
                log(f"Progreso: {i+1}/{total_batches} batches ({(i+1)/total_batches*100:.1f}%)", "PROGRESS")
                
        # Concatenar todas las predicciones
        predictions = np.vstack(predictions_list)
        
        # Recalcular el generador para obtener etiquetas en orden correcto
        eval_generator.reset()
        y_true = eval_generator.classes
        y_pred = np.argmax(predictions[:len(y_true)], axis=1)
        
        # Calcular y mostrar matriz de confusión
        log("Calculando matriz de confusión...")
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcular métricas por clase
        log("Generando reporte de clasificación detallado...")
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        report_str = classification_report(y_true, y_pred, target_names=self.class_names)
        print("\nReporte de Clasificación:")
        print(report_str)
        
        # Visualizar matriz de confusión
        log("Visualizando matriz de confusión...")
        self.plot_confusion_matrix(cm)
        
        # Calcular exactitud total
        accuracy = np.trace(cm) / np.sum(cm)
        log(f"Exactitud total del modelo: {accuracy:.2%}")
        
        # Identificar categorías problemáticas
        problematic_classes = []
        for i, class_name in enumerate(self.class_names):
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            log(f"Clase {class_name}: Precisión={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}, Muestras={support}")
            
            if precision < 0.98 or recall < 0.98:  # Menos del 98% es considerado como problema
                problematic_classes.append((class_name, precision, recall))
                log(f"⚠️ Posible problema en clase {class_name}: Confusión detectada", "WARN")
                
                # Encontrar con qué clases se confunde más
                for j, other_class in enumerate(self.class_names):
                    if i != j and cm[i, j] > 0:
                        log(f"   - Se confunde con {other_class}: {cm[i, j]} muestras ({cm[i, j]/sum(cm[i]):.1%})")
        
        # Guardar el reporte y matriz de confusión
        try:
            log("Guardando resultados de evaluación...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = "evaluaciones"
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
                
            # Guardar matriz de confusión como CSV
            cm_file = os.path.join(eval_dir, f"confusion_matrix_{timestamp}.csv")
            np.savetxt(cm_file, cm, delimiter=',', fmt='%d')
            
            # Guardar reporte de clasificación
            report_file = os.path.join(eval_dir, f"classification_report_{timestamp}.txt")
            with open(report_file, 'w') as f:
                f.write(f"Reporte de Clasificación - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Exactitud Total: {accuracy:.2%}\n\n")
                f.write(report_str)
                
                if problematic_classes:
                    f.write("\n\nClases con posibles problemas:\n")
                    for cls, prec, rec in problematic_classes:
                        f.write(f"• {cls}: Precisión={prec:.2%}, Recall={rec:.2%}\n")
                
            log(f"Resultados guardados en: {eval_dir}/", "SUCCESS")
        except Exception as e:
            log(f"Error al guardar resultados: {str(e)}", "ERROR")
            
        # Recomendaciones para mejorar
        if accuracy < 1.0:
            log("\nRecomendaciones para mejorar el modelo:")
            if problematic_classes:
                log("1. Aumentar las muestras de las siguientes clases con problemas:")
                for cls, _, _ in problematic_classes:
                    log(f"   - {cls}")
                log("2. Revisar calidad de imágenes para descartar incorrectamente etiquetadas")
                log("3. Intentar con un modelo más potente (EfficientNet, ResNet)")
            else:
                log("1. Aumentar número de épocas o usar fine-tuning más agresivo")
                log("2. Probar con diferentes arquitecturas")
        
        log("Evaluación completada", "SUCCESS")

    def plot_confusion_matrix(self, cm):
        log("Generando visualización de matriz de confusión...")
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Calcular accuracy total
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Crear mapa de calor
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        # Configurar etiquetas
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        ax.set_title(f'Matriz de Confusión\nExactitud Total: {accuracy:.2%}')
        
        # Rotar etiquetas para mejor visualización
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
        
        # Ajustar layout
        self.fig.tight_layout()
        self.canvas.draw()
        log("Visualización de matriz de confusión completada")

    def cargar_mejor_modelo(self):
        log("Iniciando carga de modelo...")
        # Permitir al usuario seleccionar el archivo de modelo
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo a cargar",
            filetypes=(("Modelos H5", "*.h5"), ("Todos los archivos", "*.*")),
            initialdir="."  # Comenzar en el directorio actual
        )
        
        if filename:
            log(f"Modelo seleccionado: {filename}")
            try:
                log("Cargando modelo...")
                self.model = load_model(filename)
                log("Modelo cargado exitosamente", "SUCCESS")
                
                # Evaluar el modelo automáticamente al cargarlo
                log("Iniciando evaluación automática del modelo cargado...")
                self.evaluar_modelo()
            except Exception as e:
                log(f"Error al cargar el modelo: {str(e)}", "ERROR")
                import traceback
                log(traceback.format_exc(), "ERROR")
        else:
            log("Operación cancelada: No se seleccionó ningún modelo", "WARN")

    def plot_training_history(self, history):
        log("Generando gráficas de rendimiento del entrenamiento...")
        self.fig.clear()
        
        # Gráfica de precisión
        log("Generando gráfica de precisión...")
        ax1 = self.fig.add_subplot(211)
        ax1.plot(history.history['accuracy'], label='Entrenamiento')
        ax1.plot(history.history['val_accuracy'], label='Validación')
        ax1.set_title('Precisión del Modelo')
        ax1.set_ylabel('Precisión')
        ax1.set_xlabel('Época')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfica de pérdida
        log("Generando gráfica de pérdida...")
        ax2 = self.fig.add_subplot(212)
        ax2.plot(history.history['loss'], label='Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Validación')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_ylabel('Pérdida')
        ax2.set_xlabel('Época')
        ax2.legend()
        ax2.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
        log("Gráficas de rendimiento generadas correctamente", "SUCCESS")

    def probar_imagen(self):
        if not self.model:
            log("Error: Primero debes entrenar el modelo o cargar un modelo guardado", "ERROR")
            return
        
        log("Iniciando prueba con imagen individual...")
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen de prueba",
            filetypes=(("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        
        if filename:
            log(f"Imagen seleccionada: {filename}")
            
            # Cargar y preprocesar imagen
            log("Preprocesando imagen...")
            img = load_img(filename, target_size=(self.IMG_SIZE, self.IMG_SIZE))
            
            # Convertir a escala de grises
            log("Convirtiendo imagen a escala de grises para coincidencia con el dataset de entrenamiento...")
            img_array = img_to_array(img)
            # Convertir a escala de grises utilizando el método de luminosidad ponderada
            # Fórmula: 0.299*R + 0.587*G + 0.114*B
            gray_img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            gray_img_array = gray_img_array.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
            # Replicar el canal gris a 3 canales para mantener compatibilidad con el modelo
            gray_img_array = np.repeat(gray_img_array, 3, axis=2)
            
            # Guardar imagen en escala de grises temporalmente para visualización
            img_gray_pil = Image.fromarray(gray_img_array.astype('uint8'))
            
            # Normalizar
            x = gray_img_array / 255.0
            x = np.expand_dims(x, axis=0)
            
            # Predicción
            log("Realizando predicción...")
            predictions = self.model.predict(x)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            log(f"\nResultados para {os.path.basename(filename)}:")
            log(f"Clase predicha: {self.class_names[predicted_class]}")
            log(f"Confianza: {confidence*100:.2f}%")
            
            # Mostrar las top 3 predicciones
            top_3 = np.argsort(predictions[0])[-3:][::-1]
            log("\nTop 3 predicciones:")
            for idx in top_3:
                log(f"{self.class_names[idx]}: {predictions[0][idx]*100:.2f}%")
            
            # Mostrar la imagen con la predicción
            log("Visualizando imagen con predicción...")
            self.plot_prediction(img_gray_pil, self.class_names[predicted_class], confidence)
            log("Prueba de imagen completada", "SUCCESS")
        else:
            log("Operación cancelada: No se seleccionó ninguna imagen", "WARN")

    def plot_prediction(self, img, predicted_class, confidence):
        log("Mostrando imagen con resultado de predicción...")
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Predicción: {predicted_class}\nConfianza: {confidence*100:.2f}%')
        self.canvas.draw()

    def limpiar_resultados(self):
        log("Limpiando área de resultados y gráficas...")
        self.text_area.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas.draw()
        log("Interfaz limpiada correctamente")
        log("Esperando nueva acción del usuario...")

    def equilibrar_clases(self):
        """Identifica y equilibra clases con problemas, duplicando imágenes problemáticas."""
        if not self.dataset_path or not self.class_names:
            log("Error: Primero debes cargar el dataset", "ERROR")
            return
            
        self.run_in_thread(self._balance_classes_thread, "Dataset equilibrado correctamente")
            
    def _balance_classes_thread(self):
        """Ejecuta el proceso de equilibrado de clases en un hilo separado."""
        log("Analizando balance de clases...")
        
        # Contar imágenes por clase
        clase_conteos = {}
        for clase in self.class_names:
            clase_path = os.path.join(self.dataset_path, clase)
            imagenes = [f for f in os.listdir(clase_path) 
                       if os.path.isfile(os.path.join(clase_path, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            clase_conteos[clase] = len(imagenes)
        
        # Identificar la clase con más imágenes
        max_imagenes = max(clase_conteos.values())
        
        # Verificar si existen las carpetas de clases balanceadas
        balanced_dataset = os.path.join(self.dataset_path, "_balanced")
        if not os.path.exists(balanced_dataset):
            os.makedirs(balanced_dataset)
            log(f"Creando directorio para dataset balanceado: {balanced_dataset}")
            
            # Actualizar barra de progreso para el proceso completo
            total_classes = len(self.class_names)
            
            # Crear directorios de clases
            for i, clase in enumerate(self.class_names):
                # Actualizar progreso
                progress = (i / total_classes) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Equilibrando clase {i+1}/{total_classes}")
                
                clase_balanced_path = os.path.join(balanced_dataset, clase)
                if not os.path.exists(clase_balanced_path):
                    os.makedirs(clase_balanced_path)
                    
                # Copiar todas las imágenes originales
                clase_path = os.path.join(self.dataset_path, clase)
                imagenes = [f for f in os.listdir(clase_path) 
                           if os.path.isfile(os.path.join(clase_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                log(f"Procesando clase {clase}: {len(imagenes)} imágenes originales")
                
                for j, img_file in enumerate(imagenes):
                    img_path = os.path.join(clase_path, img_file)
                    new_img_path = os.path.join(clase_balanced_path, img_file)
                    copy2(img_path, new_img_path)
                    
                    # Actualizar subprogreso dentro de esta clase
                    if len(imagenes) > 0:
                        sub_progress = (i + (j / len(imagenes)) / total_classes) * 100
                        self.progress_var.set(sub_progress)
                
                # Si hay desbalance, duplicar imágenes hasta alcanzar el máximo
                num_duplicar = max_imagenes - len(imagenes)
                if num_duplicar > 0:
                    log(f"Equilibrando clase {clase}: añadiendo {num_duplicar} duplicados aumentados")
                    
                    # Utilizar data augmentation para crear duplicados ligeramente diferentes
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
                    
                    # Configurar generador para aumentar datos
                    aug_gen = ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        brightness_range=[0.9, 1.1],
                        fill_mode='nearest'
                    )
                    
                    # Duplicar imágenes usando augmentation
                    imagenes_ciclo = imagenes.copy()
                    for j in range(num_duplicar):
                        # Mostrar progreso
                        if num_duplicar > 10 and j % (num_duplicar // 10) == 0:
                            log(f"Generando imagen {j+1}/{num_duplicar} para clase {clase}...", "PROGRESS")
                            
                            # Actualizar subprogreso
                            sub_progress = (i + 0.5 + (j / num_duplicar / 2)) / total_classes * 100
                            self.progress_var.set(sub_progress)
                            
                        # Seleccionar una imagen al azar
                        img_file = imagenes_ciclo[j % len(imagenes_ciclo)]
                        img_path = os.path.join(clase_path, img_file)
                        
                        # Cargar y preprocesar la imagen
                        img = load_img(img_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
                        x = img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        
                        # Generar una imagen aumentada
                        batch = aug_gen.flow(x, batch_size=1)
                        aug_img = next(batch)[0].astype('uint8')
                        
                        # Guardar la imagen aumentada
                        aug_file = f"{os.path.splitext(img_file)[0]}_aug_{j}{os.path.splitext(img_file)[1]}"
                        aug_path = os.path.join(clase_balanced_path, aug_file)
                        aug_img_pil = array_to_img(aug_img)
                        aug_img_pil.save(aug_path)
            
            # Actualizar la ruta del dataset al balanceado
            self.dataset_path = balanced_dataset
            log(f"Dataset equilibrado creado en: {balanced_dataset}", "SUCCESS")
            log("Utilizando dataset equilibrado para el entrenamiento")
        else:
            log(f"Utilizando dataset balanceado existente: {balanced_dataset}")
            self.dataset_path = balanced_dataset

    def fine_tuning_avanzado(self):
        """Realiza un fine-tuning más intensivo sobre el modelo ya entrenado para las clases problemáticas."""
        if not self.model:
            log("Error: Primero debes entrenar o cargar un modelo", "ERROR")
            return
            
        self.run_in_thread(self._fine_tuning_thread, "Fine-tuning avanzado completado con éxito")
            
    def _fine_tuning_thread(self):
        """Ejecuta el proceso de fine-tuning en un hilo separado."""
        log("Iniciando proceso de fine-tuning avanzado...")
        log("Este proceso mejorará la precisión en clases problemáticas")
        
        # Primero evaluar el modelo para identificar clases problemáticas
        log("Evaluando modelo actual para identificar problemas...")
        
        # Crear generador para evaluación
        datagen = ImageDataGenerator(rescale=1./255)
        eval_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Realizar predicciones
        log("Realizando predicciones para identificar clases problemáticas...")
        total_batches = int(np.ceil(eval_generator.samples / eval_generator.batch_size))
        
        # Función para predecir con retroalimentación de progreso
        predictions_list = []
        for i in range(total_batches):
            # Actualizar barra de progreso
            progress = (i + 1) / total_batches * 25  # 25% del progreso total
            self.progress_var.set(progress)
            self.status_var.set(f"Evaluando... {i+1}/{total_batches} batches")
            
            # Obtener batch y predecir
            batch_x, _ = eval_generator[i]
            batch_preds = self.model.predict(batch_x, verbose=0)
            predictions_list.append(batch_preds)
            
            # Mostrar progreso periódicamente
            if (i + 1) % 5 == 0 or i == total_batches - 1:
                log(f"Evaluación inicial: {i+1}/{total_batches} batches ({(i+1)/total_batches*100:.1f}%)", "PROGRESS")
        
        # Concatenar predicciones y calcular métricas
        predictions = np.vstack(predictions_list)
        y_pred = np.argmax(predictions[:len(eval_generator.classes)], axis=1)
        y_true = eval_generator.classes
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Identificar categorías problemáticas
        problematic_classes = []
        for i, class_name in enumerate(self.class_names):
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            
            if precision < 1.0 or recall < 1.0:  # Menos que perfecto
                problematic_classes.append((i, class_name, precision, recall))
                log(f"Clase problemática identificada: {class_name} (Precisión: {precision:.2%}, Recall: {recall:.2%})", "WARN")
        
        if not problematic_classes:
            log("No se detectaron clases problemáticas. ¡El modelo ya tiene 100% de precisión!", "SUCCESS")
            return
            
        # Configurar fine-tuning más agresivo
        log("Preparando modelo para fine-tuning intensivo...")
        
        # Descongelar todas las capas del modelo base para fine-tuning completo
        for layer in self.model.layers:
            layer.trainable = True
            
        # Compilar modelo con tasa de aprendizaje muy baja
        log("Recompilando modelo con tasa de aprendizaje muy reducida...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        # Crear un generador de datos más específico para las clases problemáticas
        log("Configurando aumentación de datos específica para clases problemáticas...")
        
        # Generador con augmentation intensificado
        datagen_ft = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.3,
            brightness_range=[0.7, 1.3],
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Generadores para fine-tuning
        train_generator = datagen_ft.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE // 2,  # Batch size reducido para mejor generalización
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen_ft.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE // 2,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Calcular pesos para enfocarse más en clases problemáticas
        class_weights = {}
        for i in range(len(self.class_names)):
            # Peso base
            class_weights[i] = 1.0
            
        # Aumentar peso de clases problemáticas
        for idx, cls, prec, rec in problematic_classes:
            # Cuanto peor sea la precisión/recall, mayor será el peso
            factor = 2.0 * (2.0 - prec - rec)  # Fórmula que da más peso a clases con peor desempeño
            class_weights[idx] = max(factor, 2.0)  # Mínimo el doble de peso
            log(f"Aumentando peso de clase {cls} a {class_weights[idx]:.2f}x")
            
        # Configurar callbacks para fine-tuning
        ft_callbacks = [
            ModelCheckpoint(
                'mejor_modelo_ft.h5',  # Guardar en un archivo diferente
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # Aumentado para permitir más intentos con las 50 épocas
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,  # Aumentado para las 50 épocas
                min_lr=1e-8,
                verbose=1
            ),
            # Callback para mostrar progreso en tiempo real
            LogProgressCallback(self.FT_EPOCHS),
            # Callback para actualizar la barra de progreso
            ProgressBarCallback(self.progress_var, self.FT_EPOCHS)
        ]
        
        # Iniciar fine-tuning
        log("\nIniciando fine-tuning intensivo...")
        log(f"Épocas de fine-tuning: {self.FT_EPOCHS}")
        log("Uso de pesos de clase personalizados para enfocarse en clases problemáticas")
        
        # Registrar tiempo de inicio
        tiempo_inicio_ft = datetime.datetime.now()
        log(f"Inicio de fine-tuning: {tiempo_inicio_ft.strftime('%H:%M:%S')}")
        
        # Realizar fine-tuning
        history_ft = self.model.fit(
            train_generator,
            epochs=self.FT_EPOCHS,  # Ajustado a 50 épocas para fine-tuning
            validation_data=validation_generator,
            callbacks=ft_callbacks,
            class_weight=class_weights
        )
        
        tiempo_fin = datetime.datetime.now()
        log(f"Fine-tuning completado en {tiempo_fin - tiempo_inicio_ft}")
        
        # Cargar el mejor modelo de fine-tuning
        if os.path.exists('mejor_modelo_ft.h5'):
            log("Cargando el mejor modelo de fine-tuning...")
            self.model = load_model('mejor_modelo_ft.h5')
            
            # Evaluar si es mejor que el original
            log("Evaluando si el fine-tuning mejoró el modelo original...")
            self._evaluate_model_thread()
            
            # Guardar el modelo en ambos archivos para mantener consistencia
            log("Guardando modelo mejorado en ambos archivos...")
            # Primero guardar como modelo principal
            self.model.save(self.best_model_path)
            log(f"Modelo guardado en: {self.best_model_path}", "SUCCESS")
            
            # Luego verificar que ambos archivos sean idénticos copiando explícitamente
            try:
                # Usar la función copy2 que preserva metadatos
                copy2(self.best_model_path, 'mejor_modelo_ft.h5')
                log("Sincronización de archivos completada. Ambos archivos contienen el mismo modelo.", "SUCCESS")
                
                # Verificar las fechas de los archivos
                tiempo_modelo = os.path.getmtime(self.best_model_path)
                tiempo_modelo_ft = os.path.getmtime('mejor_modelo_ft.h5')
                
                # Convertir timestamps a fechas legibles
                fecha_modelo = datetime.datetime.fromtimestamp(tiempo_modelo).strftime('%Y-%m-%d %H:%M:%S')
                fecha_modelo_ft = datetime.datetime.fromtimestamp(tiempo_modelo_ft).strftime('%Y-%m-%d %H:%M:%S')
                
                log(f"Fecha de {self.best_model_path}: {fecha_modelo}")
                log(f"Fecha de mejor_modelo_ft.h5: {fecha_modelo_ft}")
                
                # Verificar tamaños de archivo para confirmar que son idénticos
                size_modelo = os.path.getsize(self.best_model_path)
                size_modelo_ft = os.path.getsize('mejor_modelo_ft.h5')
                
                if size_modelo == size_modelo_ft:
                    log(f"Verificación completa: Ambos archivos tienen el mismo tamaño ({size_modelo/1024/1024:.2f} MB)", "SUCCESS")
                else:
                    log(f"¡Advertencia! Los archivos tienen tamaños diferentes: {size_modelo/1024/1024:.2f} MB vs {size_modelo_ft/1024/1024:.2f} MB", "WARN")
            except Exception as e:
                log(f"Error al sincronizar archivos: {str(e)}", "ERROR")
        
        log("Proceso de fine-tuning avanzado completado", "SUCCESS")

    def validacion_cruzada(self):
        """Realiza validación cruzada para evaluar el rendimiento del modelo de manera más robusta."""
        if not self.dataset_path:
            log("Error: Primero debes cargar el dataset", "ERROR")
            return
            
        self.run_in_thread(self._cross_validation_thread, "Validación cruzada completada con éxito")
            
    def _cross_validation_thread(self):
        """Ejecuta el proceso de validación cruzada en un hilo separado."""
        from sklearn.model_selection import KFold
        import matplotlib.pyplot as plt
        import numpy as np
        
        log("Iniciando proceso de validación cruzada...")
        
        # Parámetros de la validación cruzada
        k_folds = 5  # Número de particiones
        log(f"Configurando validación cruzada con {k_folds} folds...")
        
        # Preparar conjunto de datos
        log("Escaneando imágenes en el dataset...")
        class_dirs = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # Recopilar todas las rutas de imágenes y etiquetas
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.dataset_path, class_name)
            class_images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                           if os.path.isfile(os.path.join(class_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            
            log(f"Clase {class_name}: {len(class_images)} imágenes")
        
        # Convertir a arrays para poder indexar más fácilmente
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Inicializar KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Métricas para almacenar resultados
        fold_accuracies = []
        fold_losses = []
        
        # Iterar sobre los folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
            log(f"\n--- Fold {fold+1}/{k_folds} ---", "PROGRESS")
            
            # Actualizar barra de progreso
            self.progress_var.set((fold * 100) / k_folds)
            
            train_paths = image_paths[train_idx]
            train_labels = labels[train_idx]
            val_paths = image_paths[val_idx]
            val_labels = labels[val_idx]
            
            log(f"Imágenes de entrenamiento: {len(train_paths)}")
            log(f"Imágenes de validación: {len(val_paths)}")
            
            # Crear directorios temporales para este fold
            temp_dir = os.path.join("temp_cv", f"fold_{fold+1}")
            temp_train_dir = os.path.join(temp_dir, "train")
            temp_val_dir = os.path.join(temp_dir, "val")
            
            # Limpiar directorios si existen
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Crear estructura de directorios
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(temp_train_dir, exist_ok=True)
            os.makedirs(temp_val_dir, exist_ok=True)
            
            # Crear subdirectorios para cada clase
            for class_name in class_dirs:
                os.makedirs(os.path.join(temp_train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(temp_val_dir, class_name), exist_ok=True)
            
            # Copiar imágenes a directorios temporales
            log("Preparando imágenes para este fold...")
            
            # Función para copiar una imagen con su etiqueta al directorio temporal
            def copy_image_to_temp(img_path, label, is_train):
                class_name = class_dirs[label]
                dest_dir = os.path.join(temp_train_dir if is_train else temp_val_dir, class_name)
                dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
            
            # Copiar imágenes de entrenamiento
            for i, (img_path, label) in enumerate(zip(train_paths, train_labels)):
                copy_image_to_temp(img_path, label, True)
                
                # Actualizar progreso
                if i % 100 == 0:
                    self.progress_var.set((fold * 100 + (i / len(train_paths) * 50)) / k_folds)
            
            # Copiar imágenes de validación
            for i, (img_path, label) in enumerate(zip(val_paths, val_labels)):
                copy_image_to_temp(img_path, label, False)
                
                # Actualizar progreso
                if i % 100 == 0:
                    self.progress_var.set((fold * 100 + 50 + (i / len(val_paths) * 20)) / k_folds)
            
            # Crear y entrenar modelo para este fold
            log("Creando modelo para este fold...")
            
            # Crear modelo base con MobileNetV2
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                   input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
            
            # Congelar primeras capas
            for layer in base_model.layers[:int(len(base_model.layers) * 0.75)]:
                layer.trainable = False
            
            # Añadir capas personalizadas
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.4)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(len(class_dirs), activation='softmax')(x)
            
            # Modelo final
            fold_model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compilar
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            fold_model.compile(optimizer=optimizer,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
            
            # Preparar generadores
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                shear_range=0.1,
                fill_mode='nearest'
            )
            
            train_generator = datagen.flow_from_directory(
                temp_train_dir,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                batch_size=self.BATCH_SIZE,
                class_mode='categorical',
                shuffle=True
            )
            
            val_generator = datagen.flow_from_directory(
                temp_val_dir,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                batch_size=self.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
            
            # Callback para mostrar progreso
            class FoldProgressCallback(Callback):
                def __init__(self, fold, k_folds, progress_var):
                    super().__init__()
                    self.fold = fold
                    self.k_folds = k_folds
                    self.progress_var = progress_var
                
                def on_epoch_end(self, epoch, logs=None):
                    # Calcular progreso total (70% - 95% para entrenamiento)
                    fold_base = (self.fold * 100) / self.k_folds
                    epoch_progress = (epoch + 1) / 10  # Entrenamos por 10 épocas por fold
                    total_progress = fold_base + 70 + (epoch_progress * 25)
                    self.progress_var.set(min(total_progress, 95))  # Mantener bajo 95% para evaluación final
                    
            # Entrenar modelo por pocas épocas (10 épocas por fold es suficiente para CV)
            log("Entrenando modelo para este fold (10 épocas)...")
            fold_history = fold_model.fit(
                train_generator,
                epochs=10,  # Menos épocas para validación cruzada
                validation_data=val_generator,
                callbacks=[FoldProgressCallback(fold, k_folds, self.progress_var)]
            )
            
            # Evaluar modelo
            log("Evaluando modelo en conjunto de validación...")
            results = fold_model.evaluate(val_generator)
            
            # Guardar resultados
            fold_accuracies.append(results[1])  # Accuracy
            fold_losses.append(results[0])      # Loss
            
            log(f"Resultados Fold {fold+1}: Accuracy = {results[1]:.4f}, Loss = {results[0]:.4f}")
            
            # Limpiar para liberar memoria
            tf.keras.backend.clear_session()
            
        # Calcular resultados promedio
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        
        log("\n=== Resultados Validación Cruzada ===", "SUCCESS")
        log(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        log(f"Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        
        # Mostrar resultados individuales por fold
        log("\nResultados por fold:")
        for i, (acc, loss) in enumerate(zip(fold_accuracies, fold_losses)):
            log(f"Fold {i+1}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
        
        # Graficar resultados
        self.fig.clear()
        ax1 = self.fig.add_subplot(111)
        
        # Barras para accuracy
        bars = ax1.bar(range(1, k_folds+1), fold_accuracies, 0.4, label='Accuracy', color='blue', alpha=0.7)
        
        # Añadir valor sobre cada barra
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # Línea para promedio
        ax1.axhline(y=mean_accuracy, color='blue', linestyle='--', 
                   label=f'Promedio: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
        
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Resultados de Validación Cruzada')
        ax1.set_xticks(range(1, k_folds+1))
        ax1.set_ylim(0.9, 1.0)  # Ajustar según tus resultados
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Limpiar directorios temporales
        try:
            log("Limpiando archivos temporales...")
            if os.path.exists("temp_cv"):
                shutil.rmtree("temp_cv")
        except Exception as e:
            log(f"Error al limpiar archivos temporales: {str(e)}", "WARN")
        
        # Completar barra de progreso
        self.progress_var.set(100)
        log("\nValidación cruzada completada con éxito.", "SUCCESS")
        log(f"Accuracy promedio: {mean_accuracy:.2%}")
        
        # Interpretar resultados
        if std_accuracy > 0.03:  # Más del 3% de desviación estándar
            log("\nObservación: Existe variabilidad significativa entre los folds, lo que podría indicar que el modelo es sensible a la división de datos.", "WARN")
            log("Recomendación: Considere aumentar el conjunto de datos o mejorar la diversidad de las imágenes.")
        else:
            log("\nObservación: Baja variabilidad entre folds, indicando que el modelo es robusto a diferentes divisiones de datos.", "SUCCESS")
        
        if mean_accuracy < 0.95:
            log("El rendimiento promedio es inferior al 95%. Considere ajustar hiperparámetros o la arquitectura del modelo.", "WARN")
        else:
            log("Excelente rendimiento promedio. El modelo parece generalizar bien.", "SUCCESS")

def main():
    # Mostrar información del sistema
    import platform
    print(f"Iniciando aplicación a las {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=== INFORMACIÓN DEL SISTEMA ===")
    print(f"Sistema Operativo: {platform.system()} {platform.version()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print("=== VERSIONES DE LIBRERÍAS ===")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"PIL: {Image.__version__}")
    
    # Verificar si hay GPU disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("=== GPU DETECTADA ===")
        for gpu in gpus:
            print(f" - {gpu.name}")
        
        # Intentar configurar memoria de GPU para crecimiento dinámico
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memoria GPU configurada para crecimiento dinámico.")
        except Exception as e:
            print(f"Error al configurar memoria GPU: {e}")
    else:
        print("=== NO SE DETECTARON GPUs ===")
        print("El entrenamiento se realizará en CPU (será más lento).")
    
    # Iniciar la interfaz gráfica
    print("\nIniciando interfaz gráfica...")
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: (print("Cerrando aplicación..."), root.destroy()))
    app = ConectorClassifierGUI(root)
    
    # Configurar manejo de excepciones para mantener la aplicación en ejecución
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Manejo normal para Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Para otras excepciones, registrar en el log
        import traceback
        log("¡ERROR CRÍTICO NO MANEJADO!", "ERROR")
        log(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), "ERROR")
    
    # Establecer manejador de excepciones
    sys.excepthook = handle_exception
    
    # Iniciar el bucle de la interfaz
    print("Aplicación iniciada correctamente. Esperando acciones del usuario...")
    root.mainloop()
    print("Aplicación cerrada.")

if __name__ == "__main__":
    main()