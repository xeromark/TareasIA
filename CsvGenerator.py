import csv
import random

def generar_csv_con_dependencias(nombre_archivo, num_filas):
    encabezado = ['Nublado', 'Aspersor', 'Lluvia', 'Hierba_mojada']
    
    with open(nombre_archivo, mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        
        writer.writerow(encabezado)
        
        # Generador
        for i in range(num_filas):
            Nublado = random.randint(0, 1)
            Aspersor = 0 if Nublado == 1 else random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1)
            Lluvia = random.randint(0, 1) if Nublado == 1 else 0
            Hierva_mojada = 1 if (Aspersor == 1 or Lluvia == 1) else 0

            fila = [Nublado, Aspersor, Lluvia, Hierva_mojada]
            writer.writerow(fila)

# Genera un archivo CSV con 10000 filas de ejemplo con relaciones entre las variables
generar_csv_con_dependencias('dataset.csv', 10001)
