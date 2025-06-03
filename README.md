# GRASSIM
## Un simulateur en Python de la propagation de l'herbe dans un lopin de terre réalisé dans le cadre d'un projet de Première NSI

## Introduction
**Grassim** (*Grass simulator*) est un petit script Python qui permet de simuler la propagation de l'herbe dans un carré de terre appelé biome, selon un mécanisme similaire à celui utilisé dans le jeu Minecraft.

Le cœur de ce dépôt a été réalisé fin 2023 pour un projet de Première NSI. Il a été adapté à une publication Github (fichier `README.md`) en janvier 2025.

![](https://i.imgur.com/bvpYvDF.gif)

### Algorithme utilisé pour la propagation
Nous partons d'un biome carré, représenté par une liste 2D, composé au départ d'un seul bloc d'herbe.

```
Couvrir un unique bloc d'herbe
Tant que tous les blocs ne sont pas couverts d'herbe Faire
    Pour chaque bloc du biome Faire
        l <-- liste des blocs voisins du bloc
        g <-- nombre des blocs dans l couverts d'herbe
        p <-- g/(2*longueur(l))

        k <-- nombre tiré au hasard entre 0 et 1
        Si k < p Faire
            le bloc se couvre d'herbe
        Fin Si
    Fin Pour
Fin Tant que
```

Chaque passage dans la boucle « Tant que » est appelé *étape* de la simulation.

## Fonctionnalités
### Classe `Biome`
Le dépôt contient une classe `Biome` qui permet de simuler la propagation de l'herbe sur un seul biome de taille arbitraire.

* `Biome(size: int, first_bloc: tuple[int, int] = None, max_steps: int = 10000)` : instance de la classe `Biome` avec les arguments `size` (taille en blocs), `first_bloc` (optionnel ; coordonnées du premier bloc d'herbe, par défaut le milieu du biome), `max_steps` (nombre maximum d'étapes de l'algorithme avant abandon).
* `Biome.run(draw: bool = False)` : permet de lancer la simulation. Si `draw` est `True`, un GIF représentant la simulation sera exporté dans le dossier `/gifs`.

### Classe `Graphics` 
* `Graphics()` : instance de la classe contenant les méthodes utilisées pour l'affichage de statistiques en rapport avec la simulation.
* `Graphics.step_graph(x_params: tuple[int,int,int], block_funcs: list[Callable] = None, series: int = 1, regression_func: Callable = None, output_path: str = None, show: bool = True)` : affiche le graphe du nombre d'étapes de la simulation en fonction de la taille du biome en fonction et de l'« indice de position du bloc 0 » (sous forme d'échelle de couleur), valeur comprise entre 0 et 1, qui évalue sa proximité avec le centre du biome (plus de détails sur les calculs de cette métrique dans les *docstrings*).
* `Graphics.time_graph(x_params: tuple[int,int,int], block_funcs: list[Callable] = None, series: int = 1, regression_func: Callable = None, output_path: str = None, show: bool = True)` : affiche le graphe du temps CPU qu'a pris la simulation en fonction des mêmes paramètres que précédemment.

Ces deux fonctions prennent pour arguments :
* `x_params` : `tuple` similaire aux entrées de la fonction `range` (taille minimale, taille maximale, pas).
* `serie` : nombre de simulations par taille et par `lambda` fonction (cf. *infra*).
* `block_funcs` : une liste de `lambda` fonctions prenant en paramètre la taille et renvoyant les coordonnées du premier bloc. Une série supplémentaire sera réalisée pour chaque fonction.
* `regression_func` : un modèle de fonction dont les coefficients seront déterminés par une régression polynomiale (voir exemple dans `main.py`).
* `output_path` : chaîne de caractères représentant le chemin où le graphe sera enregistré (pas d'enregistrement si pas renseigné).
* `show` : booléen à `True` pour afficher le graphe, `False` sinon.


![](https://i.imgur.com/I18la3L.png)
![](https://i.imgur.com/cB4PQqr.png)

## Mise en place
Après avoir cloné le dépôt et ouvert un terminal à la racine, exécutez la commande suivante :
```
pip install -r requirements.txt
```

Une version récente de Python est requise (`>= 3.9`).
