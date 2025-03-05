from src.Biome import Biome
from src.Graphics import Graphics

def f(x,a,b): # modèle de fonction de régression pour le temps
    return a*x**2 + b*x

def g(x, a, b): # modèle de fonction de régression pour le nombre d'étapes
    return a*x+b

if __name__ == "__main__":
    block_funcs = [lambda size: (size//2, size//2), lambda size: (0, 0), lambda size: (size//4, size//4)] # fonctions renvoyant les coordonnées du premier bloc
    # en fonction de la taille du biome

    b = Biome(20) # un biome carré 20x20 blocs
    b.run(draw=True) # simulation pour un biome qui générera un GIF dans le dossier /gifs grâce à l'argument draw

    Graphics.time_graph((5,40,1), series=2, block_funcs=block_funcs, regression_func=f, output_path="time.png") # affiche le graphe du temps CPU pris pour générer la simulation en fonction de la taille du biome
    Graphics.step_graph((5,40,1), series=2, block_funcs=block_funcs, regression_func=g, output_path="step.png") # affiche le graphe du nombre d'étapes de la simulation en fonction du biome
    # dans ces exemples, les tailles vont de 5 à 40 blocs avec un pas de 1 et il y a deux séries donc deux simulations pour chaque taille