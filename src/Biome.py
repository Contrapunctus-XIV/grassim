import random
from time import perf_counter, time
from enum import IntFlag
from pathlib import Path
from PIL import Image

class Cell(IntFlag):
    BORDER = -1
    DIRT = 0
    GRASS = 1

DIRNAME = Path(__file__).parent.parent.resolve()
SPR_DIRT = Image.open(DIRNAME/"sprites/dirt.png")
SPR_GRASS = Image.open(DIRNAME/"sprites/grass.png")
CELL_SIZE = 16

class Biome:
    def __init__(self, size: int, first_bloc: tuple[int,int] = None, max_steps: int = 10_000):
        """
        Biome standard simulant la propagation du gazon sans intervention d'entité extérieure.

        Syntaxe
        --------------
        Initialisation

        Biome.Biome(size, first_bloc[, max_steps])
        size : int
            dimensions du biome
        fist_bloc : tuple[int,int]
            coordonnées du premier bloc de gazon sous forme de tuple d'int
            si pas renseigné, choix du bloc le plus au centre
        [max_steps : int]
            nombre maximal d'étapes pour une simulation (par défaut 10 000)
        ---------------
        Générer la simulation

        Biome.simulation (voir doc ci-dessous)
        """
        if first_bloc is None:
            first_bloc = (size//2, size//2)

        x, y = first_bloc
        assert 0 <= x < size and 0 <= y < size, "First grass block is not in the biome or size is not valid."

        self.grid: list[list[Cell]] = []
        self.size: int = size
        self.first_bloc: tuple[int] = first_bloc
        self.max_steps: int = max_steps
        self.frames: list = []

    def _draw_step(self) -> Image:
        """
        Ajoute l'image de l'étape actuelle de la simulation à la liste des frames.
        """
        img_size = self.size * CELL_SIZE
        image = Image.new("RGB", (img_size, img_size))

        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                coords = ((x-1) * CELL_SIZE, (y-1) * CELL_SIZE)
                if self.grid[x][y] == Cell.DIRT:
                    image.paste(SPR_DIRT, coords)
                else:
                    image.paste(SPR_GRASS, coords)

        self.frames.append(image)
        return image
    
    def _gif(self, output_path: str = None, duration: int = 100):
        """
        Rassemble les frames pour crée le GIF.
        """
        if output_path is None:
            output_path = DIRNAME/f"gifs/sim_{str(f'{time():.2f}').replace(".", "_")}.gif"

        assert len(self.frames) > 0, "No frames were drown during the simulation."
        base_img: Image = self.frames[0]
        base_img.save(output_path,
                save_all=True,
                append_images=[self.frames[i] for i in range(1, len(self.frames))],
                duration=duration,
                loop=0)
        
        return output_path
            
    def _update(self) -> list[list[Cell]]:
        """
        Met à jour le biome.
        """
        updated_grid = [row[:] for row in self.grid]

        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                if self.grid[x][y] == Cell.DIRT:
                    neighbours = [
                        self.grid[x - 1][y - 1], self.grid[x - 1][y], self.grid[x - 1][y + 1],
                        self.grid[x][y - 1], self.grid[x][y + 1],
                        self.grid[x + 1][y - 1], self.grid[x + 1][y], self.grid[x + 1][y + 1]
                    ]

                    block_neighbours, grass_neighbours = 8, 0

                    for neighbour in neighbours:
                        if neighbour == Cell.GRASS:
                            grass_neighbours += 1
                        elif neighbour == Cell.BORDER:
                            block_neighbours -= 1
                            
                    p = grass_neighbours / (2 * block_neighbours)

                    if random.random() < p:
                        self.grass_blocks += 1
                        updated_grid[x][y] = 1

        self.grid = updated_grid

        return self.grid

    def run(self, draw: bool = False) -> tuple[int, float, str]:
        """
        Biome.simulation

        Génère la simulation. Peut-être appelée plusieurs fois pour un même objet Biome.

        Args
        ----------------
        draw: bool
            True si un GIF doit être produit, False sinon.

        Valeurs renvoyées
        -----------------
        tuple : (steps, simulation_time)
            steps : int
                nombre d'étapes de la simulation
            simulation_time : float
                temps qu'a pris la simulation (mesuré avec time.perf_counter)
        """
        t_start = perf_counter()

        self.frames = []
        self.grid = [[0 if 1 <= i <= self.size and 1 <= j <= self.size else -1 for j in range(self.size + 2)] for i in range(self.size + 2)]
        self.grid[self.first_bloc[0]][self.first_bloc[1]] = 1
        self.grass_blocks = 1

        steps = 0
        
        while self.grass_blocks < self.size ** 2 and steps <= self.max_steps:
            self._update()
            if draw:
                self._draw_step()
            steps += 1
            
        simulation_time = perf_counter() - t_start

        gif_path = None
        if draw:
            gif_path = self._gif()
            
        return steps, simulation_time, gif_path