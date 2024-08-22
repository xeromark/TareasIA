import random
import time
inicio_tiempo = time.time()

# Lista de nombres de comida
comidas = [
    "Pizza", "Hamburguesa", "Ensalada", "Sushi", "Taco", "Pasta", "Sandwich", "Burrito", "Ramen", "Kebab",
    "Pollo frito", "Ceviche", "Paella", "Lasagna", "Churrasco", "Arepa", "Empanada", "Falafel", "Dim sum", "Baguette",
    "Goulash", "Shawarma", "Croissant", "Gnocchi", "Quesadilla", "Tamales", "Pho", "Baklava", "Biryani", "Gyros",
    "Chow mein", "Moussaka", "Ratatouille", "Bruschetta", "Maki", "Tempura", "Teriyaki", "Curry", "Tandoori", "Hummus",
    "Kimchi", "Bibimbap", "Banh mi", "Pierogi", "Fajitas", "Mole", "Bratwurst", "Kofta", "Samosa", "Pad thai",
    "Sauerbraten", "Rosti", "Bouillabaisse", "Tiramisu", "Cannoli", "Gelato", "Poutine", "Chili con carne", "Guacamole",
    "Tortilla espaniola", "Gumbo", "Jambalaya", "Waffles", "Crepes", "Fondue", "Tapas", "Antipasto", "Borscht",
    "Clam chowder", "Lobster roll", "Fish and chips", "Shepherd's pie", "Bangers and mash", "Haggis", "Yorkshire pudding",
    "Black pudding", "Ploughman's lunch", "Beef Wellington", "Eton mess", "Sticky toffee pudding", "Creme brulee",
    "Croque monsieur", "Raclette", "Steak tartare", "Cassoulet", "Rillettes", "Couscous", "Tagine", "Jollof rice",
    "Fufu", "Ugali", "Bunny chow", "Bobotie", "Malva pudding", "Chakalaka", "Pastel de choclo", "Cazuela", "Curanto",    
    "Sopa de tortilla", "Huevos rancheros", "Pozole", "Chilaquiles", "Tlayudas", "Tacos al pastor", "Tamales oaxaquenios", 
    "Barbacoa", "Menudo", "Aguachile", "Churros", "Conchas", "Buniuelos", "Flan", "Pan de muerto", "Mole poblano", 
    "Carnitas", "Carne asada", "Enchiladas", "Flautas", "Picadillo", "Pico de gallo", "Salsa verde", "Salsa roja", 
    "Guajolote", "Pipian", "Entomatadas", "Sopes", "Gorditas", "Huaraches", "Pambazo", "Tlacoyo", "Molletes", 
    "Chapulines", "Chiles en nogada", "Chiles rellenos", "Rajas con crema", "Caldo de camaron", "Caldo de res", 
    "Caldo tlalpenio", "Caldo de pollo", "Birria", "Chimichanga", "Nachos", "Frito pie", "Margarita", "Tequila", 
    "Mezcal", "Pulque", "Atole", "Champurrado", "Aguamiel", "Horchata", "Jamaica", "Tamarindo", "Tejuino", "Licuado", 
    "Tepache", "Chocomilk", "Rompope", "Cafe de olla", "Chocolate abuelita", "Pozol", "Sangrita", "Pinia colada", 
    "Michelada", "Chelada", "Clamato", "Agua de cebada", "Agua de coco", "Agua de horchata", "Agua de jamaica", 
    "Agua de limon", "Agua de tamarindo", "Agua de sandia", "Agua de pepino", "Agua de pinia", "Agua de melon", 
    "Agua de fresa", "Agua de mango", "Agua de guayaba", "Agua de papaya", "Agua de platano", "Agua de durazno", 
    "Agua de ciruela", "Agua de chia", "Agua de avena", "Agua de linaza", "Agua de nopal", "Agua de alfalfa", 
    "Agua de perejil", "Agua de apio", "Agua de zanahoria", "Agua de betabel", "Agua de espinaca", "Agua de pepino", 
    "Agua de jicama", "Agua de tuna", "Agua de limon con chia", "Agua de pepino con limon"

]

# Nombre del archivo de salida
archivo_salida = 'pedidos'

# Numero de entradas que quieres generar
num_entradas = 10001

# Abre el archivo en modo escritura
with open(archivo_salida, 'w') as archivo:
    for _ in range(num_entradas):
        comida = random.choice(comidas)
        precio = random.randint(1000, 20000)
        archivo.write(f"{comida}|{precio}\n")



fin_tiempo = time.time()
# Calcula el tiempo transcurrido
tiempo_transcurrido = fin_tiempo - inicio_tiempo

print(f"Archivo '{archivo_salida}' generado con exito en '{tiempo_transcurrido}' segundos.")
