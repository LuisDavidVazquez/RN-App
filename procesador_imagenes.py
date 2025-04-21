import os
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")
        self.root.geometry("800x600")
        
        # Variables
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.target_size = (224, 224)
        
        # Variables de control
        self.resize_var = tk.BooleanVar(value=True)
        self.rotate_var = tk.BooleanVar(value=False)
        self.grayscale_var = tk.BooleanVar(value=False)
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame para selección de carpetas
        folder_frame = ttk.LabelFrame(main_frame, text="Selección de Carpetas", padding="10")
        folder_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Input folder
        ttk.Label(folder_frame, text="Carpeta de origen:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(folder_frame, text="Seleccionar", command=self.select_input_folder).grid(row=0, column=2)
        
        # Output folder
        ttk.Label(folder_frame, text="Carpeta de destino:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(folder_frame, text="Seleccionar", command=self.select_output_folder).grid(row=1, column=2)
        
        # Frame para opciones de procesamiento
        options_frame = ttk.LabelFrame(main_frame, text="Opciones de Procesamiento", padding="10")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Checkboxes para opciones
        ttk.Checkbutton(options_frame, text="Redimensionar a 224x224", variable=self.resize_var).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(options_frame, text="Rotar 90° a la derecha", variable=self.rotate_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(options_frame, text="Convertir a escala de grises", variable=self.grayscale_var).grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Botón de procesamiento
        ttk.Button(main_frame, text="Procesar Imágenes", command=self.process_images).grid(row=2, column=0, pady=20)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=3, column=0, pady=10)
        
        # Etiqueta de estado
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=4, column=0, pady=10)
        
    def select_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)
            
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)
            
    def process_images(self):
        input_path = self.input_folder.get()
        output_path = self.output_folder.get()
        
        if not input_path or not output_path:
            messagebox.showerror("Error", "Por favor selecciona las carpetas de origen y destino")
            return
            
        if not (self.resize_var.get() or self.rotate_var.get() or self.grayscale_var.get()):
            messagebox.showerror("Error", "Por favor selecciona al menos una opción de procesamiento")
            return
            
        # Obtener lista de imágenes y subcarpetas
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    images.append(os.path.join(root, file))
        
        if not images:
            messagebox.showerror("Error", "No se encontraron imágenes en la carpeta de origen")
            return
            
        # Configurar barra de progreso
        self.progress['maximum'] = len(images)
        self.progress['value'] = 0
        
        # Procesar cada imagen
        for i, image_path in enumerate(images):
            try:
                # Actualizar estado
                image_name = os.path.basename(image_path)
                relative_path = os.path.relpath(os.path.dirname(image_path), input_path)
                output_subfolder = os.path.join(output_path, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                self.status_label.config(text=f"Procesando: {image_name}")
                self.root.update()
                
                # Abrir imagen
                img = Image.open(image_path)
                
                # Aplicar transformaciones según las opciones seleccionadas
                if self.resize_var.get():
                    # Calcular nuevo tamaño manteniendo la relación de aspecto
                    ratio = min(self.target_size[0] / img.size[0], self.target_size[1] / img.size[1])
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    
                    # Redimensionar imagen
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Crear nueva imagen con fondo blanco
                    new_img = Image.new("RGB", self.target_size, (255, 255, 255))
                    
                    # Pegar la imagen redimensionada en el centro
                    paste_x = (self.target_size[0] - new_size[0]) // 2
                    paste_y = (self.target_size[1] - new_size[1]) // 2
                    new_img.paste(img, (paste_x, paste_y))
                    img = new_img
                
                if self.rotate_var.get():
                    img = img.rotate(-90, expand=True)
                
                if self.grayscale_var.get():
                    img = img.convert('L')
                
                # Guardar imagen en la subcarpeta correspondiente
                output_image_path = os.path.join(output_subfolder, image_name)
                img.save(output_image_path, quality=95)
                
                # Actualizar barra de progreso
                self.progress['value'] = i + 1
                self.root.update()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar {image_name}: {str(e)}")
                return
                
        self.status_label.config(text="¡Procesamiento completado!")
        messagebox.showinfo("Éxito", "Todas las imágenes han sido procesadas exitosamente")

def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main() 