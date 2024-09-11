import csv
import random

def generar_csv_con_dependencias(nombre_archivo, num_filas):
    encabezado = ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
    
    with open(nombre_archivo, mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        
        # Escribir el encabezado
        writer.writerow(encabezado)
        
        # Generar y escribir las filas con relaciones
        for _ in range(num_filas):
            cloudy = random.randint(0, 1)
            sprinkler = random.randint(0, 1) if cloudy == 1 else 0
            rain = random.randint(0, 1) if cloudy == 1 else 0
            wet_grass = 1 if (sprinkler == 1 or rain == 1) else 0
            fila = [cloudy, sprinkler, rain, wet_grass]
            writer.writerow(fila)

# Genera un archivo CSV con 1000 filas de ejemplo con relaciones entre las variables
generar_csv_con_dependencias('dataset.csv', 10000)
