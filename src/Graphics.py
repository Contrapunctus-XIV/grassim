import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from sklearn.metrics import r2_score
from src.Biome import Biome
from typing import Callable
from enum import IntEnum

class SimulationType(IntEnum):
    STEPS = 0
    CPU_TIME = 1

class Graphics:
    """
    Générer des graphiques sur des simulations de Biome.

    Syntaxe
    --------------
    Initialisation

    graphics = Graphics()

    voir docs des méthodes Graphics.graph et Graphics.time_graph
    """
        
    @staticmethod
    def _compute_dist(bloc1: tuple[int], size: int) -> float:
        """
        Calcule la distance du bloc donné en entrée par rapport au centre du biome selon la méthode détaillée
        dans la docstring de Graphics.graph
        """
        dist = max(abs(bloc1[0]-size//2), abs(bloc1[1]-size//2))
        if size%2==0 or size==1:
            return dist*2/size
        return dist*2/(size-1)
    
    @staticmethod
    def _check_x_params(x_params: tuple[int, int, int]) -> None:
        """
        S'assure que les paramètres x (tailles, pas) donnés pour les graphes sont corrects.
        """
        x_min, x_max, steps = x_params
        assert x_min<=x_max and x_min>0 and x_max>0 and steps>0 and type(x_min)==int and type(x_max)==int and type(steps)==int

    @staticmethod
    def _get_values(simulation_type: SimulationType, x_values: list[int], prefix: str, get_block: Callable) -> tuple[list[float]]:
        """
        Renvoie les valeurs calculées pour les simulations selon le type (get_block, temps ou nombre d'étapes) et met à jour la barre tqdm.
        """
        distances = [Graphics._compute_dist(get_block(size), size) for size in x_values]
        tqdm_range = tqdm(x_values, leave=False)
        tqdm_range.set_description(prefix)
        y_values = [Biome(size, get_block(size), max_steps=500).run()[simulation_type] for size in tqdm_range]
            
        return y_values, distances

    @staticmethod
    def _plot(simulation_type: SimulationType, x_values, y_values) -> None:
        """
        Prépare le graphique du type de simulation donné en entrée, en marquant les points de coordonnées (x_values, y_values) donnés en entrée.
        """
        ax = plt.figure().gca()
        ax.xaxis.get_major_locator().set_params(integer=True)

        for ser in y_values:
            for y in ser:
                plt.scatter(x=x_values, y=y[0], c=y[1], cmap='viridis', s=30, vmin=0, vmax=1)

        plt.xlabel('Dimension')

        if simulation_type == SimulationType.CPU_TIME:
            plt.title('Temps en fonction de la dimension')
            plt.ylabel('Temps (sec)')

        else:
            plt.title('Nombre d\'étapes en fonction de la dimension')
            plt.ylabel('Nbre d\'étapes')

        plt.colorbar(label='Indice de position du bloc 0')

    @staticmethod
    def _regression(x_values: list[int], y_values: list[float], func: Callable) -> tuple[list, float]:
        """
        Réalise une régression sur un unique jeu de données donné en entrée et renvoie les coefs et r^2 obtenus.
        """
        coefs = curve_fit(func, x_values, y_values)[0]
        r_squared = r2_score(y_values, func(np.array(x_values), *coefs))

        return coefs, r_squared

    @staticmethod
    def _global_regression(x_values, y_values, series, nb_lambdas, func) -> None:
        """
        Réalise l'ensemble des régressions linéaires sur toutes les séries en faisant une moyenne des valeurs obtenues.
        Affiche le résultat.
        """
        x_func = np.linspace(min(x_values), max(x_values), 200)

        x_nb = len(y_values[0][0][0])
        y_l = [[[y_values[i][j][0][k] for i in range(series)] for k in range(x_nb)] for j in range(nb_lambdas)]
        y_av = [[sum(y_l[i][j])/series for j in range(x_nb)] for i in range(nb_lambdas)]
            
        coefs_list = [Graphics._regression(x_values, tab, func) for tab in y_av]

        for coefs in coefs_list:
            print(f'coefs = {coefs[0]}, r^2 = {coefs[1]}')
            plt.plot(x_func, func(x_func, *coefs[0]))

    @staticmethod
    def _create_tqdm_prefix(serie: tuple[int], block_func: tuple[int] = None) -> str:
        """
        Renvoie un nouveau préfixe tqdm selon les valeurs passées en entrée.

        Arguments
        ---------
        serie: tuple[int]
            tuple contenant deux entiers : le numéro de la série actuelle et le nombre total de séries à calculer
        [block_func: tuple[int]]
            tuple optionnel contenant deux entiers : le numéro de la fonction de calcul du premier bloc actuel et le nombre total de functions à prendre en compte

        """
        current_serie, tot_series = serie
        prefix = f'Serie {current_serie+1}/{tot_series}' if block_func is None else f'Serie {current_serie+1}/{tot_series}, lambda {block_func[0]+1}/{block_func[1]}'
        
        return prefix
    
    @staticmethod
    def time_graph(x_params: tuple[int,int,int], block_funcs: list[Callable] = None, series: int = 1, regression_func: Callable = None, output_path: str = None, show: bool = True) -> None:
        """Graphics.time_graph(x_params[, series[, block_funcs[, regression_func]]])

        Trace le graphique du temps CPU d'une simulation en fonction de la dimension.
        x : dimension
        y : temps CPU (sec)
        z (couleur des points): indice de position du bloc 0 (évalue la distance du bloc de départ par rapport au centre du biome)

        L'indice de position est calculé ainsi, pour un bloc de coordonnées (x,y) et une dimension size:
            max(|x-size//2|, |y-size//2|)*2/size        si size est un entier pair ou égal à 1
            max(|x-size//2|, |y-size//2|)*2/(size-1)    si size est un entier impair
        L'indice de position prend des valeurs comprises entre 0.0 et 1.0 (inclus) :
            un indice de 0.0 signifiant que le premier bloc de gazon est situé au centre du biome
            un indice de 1.0 signifiant que le premier bloc de gazon est situé dans un coin du biome
            
        Fonctionnement de la régression linéaire (si regression_func renseigné) :
            Calcule la régression pour chaque moyenne de valeurs y de chaque lambda selon le modèle
            de la fonction passée en entrée.

        Arguments
        ---------
        x_params : tuple[int,int,int]
            un tuple d'int similaire aux entrées de la fonction range
            (dimension_debut, dimension_fin, pas)
            ces trois grandeurs doivent être des entiers strictement positifs
            dimension_debut et dimension_fin sont incluses
        [block_funcs : list[Callable]]
            une liste de functions renvoyant les coordonnées du premier bloc de gazon en fonction de la dimension du biome
        [series : int]
            nombre de séries de simulations
            tout le tableau de lambdas est parcouru pour chaque série
        [regression_func : Callable]
            une fonction servant de modèle pour la régression
            ne pas renseigner pour ne pas calculer de régression
        [output_path : str]
            chemin auquel le graphe sera enregistré
            ne pas renseigner pour ne pas sauvegarder le graphe
        [show : bool]
            True pour afficher le graphe, False sinon.
        """

        Graphics._check_x_params(x_params)

        if block_funcs is None:
            block_funcs = [lambda size: (size//2,)*2]

        x_min, x_max, steps = x_params
        x_values = list(range(x_min, x_max+1, steps))
        nb_lambdas = len(block_funcs)
        y_values = [[Graphics._get_values(SimulationType.CPU_TIME, x_values=x_values, get_block=block_funcs[i], prefix=Graphics._create_tqdm_prefix((ser, series), block_func=(i, nb_lambdas))) for i in range(nb_lambdas)] for ser in range(series)]

        Graphics._plot(SimulationType.CPU_TIME, x_values, y_values)

        if regression_func is not None:
            Graphics._global_regression(x_values, y_values, series, nb_lambdas, regression_func)
            
        if output_path is not None:
            plt.savefig(output_path)
            
        if show:
            plt.show()
        
    @staticmethod
    def step_graph(x_params: tuple[int,int,int], block_funcs: list[Callable] = None, series: int = 1, regression_func: Callable = None, output_path: str = None, show: bool = True) -> None:
        """
        Graphics.graph(x_params[, series[, block_funcs[, regression_func[, output_path]]]])

        Trace le graphique du nombre d'étapes d'une simulation en fonction de la dimension.
        x : dimension
        y : nombre d'étapes d'une simulation
        z (couleur des points): indice de position du bloc 0 (évalue la distance du bloc de départ par rapport au centre du biome)
        seriessize-1)    si size est un entier impair
        L'indice de position prend des valeurs comprises entre 0.0 et 1.0 (inclus) :
            un indice de 0.0 signifiant que le premier bloc de gazon est situé au centre du biome
            un indice de 1.0 signifiant que le premier bloc de gazon est situé dans un coin du biome
            
        Fonctionnement de la régression linéaire (si regression_func renseigné) :
            Calcule la régression pour chaque moyenne de valeurs y de chaque lambda selon le modèle
            de la fonction passée en entrée.

        Arguments
        ---------
        x_params : tuple[int,int,int]
            un tuple d'int similaire aux entrées de la fonction range
            (dimension_debut, dimension_fin, pas)
            ces trois grandeurs doivent être des entiers strictement positifs
            dimension_debut et dimension_fin sont incluses
        [block_funcs : list[Callable]]
            une liste de functions renvoyant les coordonnées du premier bloc de gazon en fonction de la dimension du biome
        [series : int]
            nombre de séries de simulations
            tout le tableau de lambdas est parcouru pour chaque série
        [regression_func : Callable]
            une fonction servant de modèle pour la régression
            ne pas renseigner pour ne pas calculer de régression
        [output_path : str]
            chemin auquel le graphe sera enregistré
            ne pas renseigner pour ne pas sauvegarder le graphe
        [show : bool]
            True pour afficher le graphe, False sinon.
        """

        Graphics._check_x_params(x_params)

        if block_funcs is None:
            block_funcs = [lambda size: (size//2,)*2]

        x_min, x_max, steps = x_params
        x_values = list(range(x_min, x_max+1, steps))
        nb_lambdas = len(block_funcs)
        y_values = [[Graphics._get_values(SimulationType.STEPS, x_values=x_values, get_block=block_funcs[i], prefix=Graphics._create_tqdm_prefix((ser, series), block_func=(i, nb_lambdas))) for i in range(nb_lambdas)] for ser in range(series)]

        Graphics._plot(SimulationType.STEPS, x_values, y_values)

        if regression_func is not None:
            Graphics._global_regression(x_values, y_values, series, nb_lambdas, regression_func)
        
        if output_path is not None:
            plt.savefig(output_path)

        if show:
            plt.show()