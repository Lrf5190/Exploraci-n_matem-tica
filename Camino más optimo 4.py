import pygame
import sys
import time
import math
from queue import PriorityQueue

# Inicializar pygame
pygame.init()

# Configuración de colores
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
CYAN    = (0, 255, 255)
GREY    = (255,175,0)
EDGE_COLOR = (200, 200, 200)  # Color por defecto para enlaces

# Configuración del grafo
WIDTH, HEIGHT = 800, 800
ROWS = 50         # Densidad del grafo (distribución en forma de cuadrícula)
GAP = WIDTH // ROWS  # Distancia entre nodos (para calcular su posición)
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Camino Más Corto con Dijkstra (Grafo)")

# Variables globales
start = None
end = None
visited_nodes = 0
path_length = 0
search_time = 0

# Clase Nodo: cada nodo representa un vértice del grafo
class Node:
    def __init__(self, row, col, gap):
        self.row = row
        self.col = col
        # La posición central del nodo se calcula para que se dispongan en forma de cuadrícula
        self.x = col * gap + gap // 2
        self.y = row * gap + gap // 2
        self.gap = gap
        # Radio reducido para ver las conexiones entre nodos
        self.radius = gap // 4
        self.color = WHITE
        self.neighbors = []  # Lista de nodos vecinos

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = CYAN

    def make_end(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = RED

    def make_path(self):
        self.color = BLUE

    def make_visited(self):
        self.color = GREY

    def draw(self, win):
        # Dibuja el nodo como un círculo (relleno y contorno)
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(win, BLACK, (self.x, self.y), self.radius, 1)

    def get_center(self):
        return (self.x, self.y)

    def update_neighbors(self, nodes, total_rows):
        # Si el nodo es obstáculo, no tendrá vecinos
        if self.is_barrier():
            self.neighbors = []
            return

        self.neighbors = []
        # Direcciones: 4 cardinales y 4 diagonales
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            new_row = self.row + d[0]
            new_col = self.col + d[1]
            if 0 <= new_row < total_rows and 0 <= new_col < total_rows:
                neighbor = nodes[(new_row, new_col)]
                if not neighbor.is_barrier():
                    self.neighbors.append(neighbor)

# Crear el grafo: un diccionario donde las llaves son (row, col) y el valor es un nodo
def create_graph(rows, gap):
    nodes = {}
    for i in range(rows):
        for j in range(rows):
            nodes[(i, j)] = Node(i, j, gap)
    return nodes

# Dibuja los enlaces (líneas) entre nodos y los nodos (círculos)
def draw(win, nodes, total_rows, gap):
    win.fill(WHITE)
    # Dibujar enlaces: se recorre cada nodo y se dibuja la conexión con cada vecino (sin duplicar)
    for node in nodes.values():
        for neighbor in node.neighbors:
            # Dibujar el enlace solo una vez (usando el orden de las coordenadas)
            if (node.row, node.col) < (neighbor.row, neighbor.col):
                # Si ambos nodos son parte del camino óptimo, se utiliza el color del camino
                if node.color == BLUE and neighbor.color == BLUE:
                    line_color = BLUE
                else:
                    line_color = EDGE_COLOR
                pygame.draw.line(win, line_color, node.get_center(), neighbor.get_center(), 2)
    # Dibujar cada nodo
    for node in nodes.values():
        node.draw(win)
    pygame.display.update()

# Función para obtener el nodo clicado (buscando el que tenga su centro cerca del cursor)
def get_clicked_node(pos, nodes):
    mx, my = pos
    for node in nodes.values():
        distance = math.sqrt((node.x - mx) ** 2 + (node.y - my) ** 2)
        if distance < node.radius:
            return node
    return None

# Algoritmo de Dijkstra
def dijkstra_algorithm(draw, nodes, start, end):
    global visited_nodes, path_length, search_time
    visited_nodes = 0
    path_length = 0
    search_time = 0

    start_time = time.time()
    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    distances = {node: float("inf") for node in nodes.values()}
    distances[start] = 0
    prev_node = {node: None for node in nodes.values()}

    while not queue.empty():
        current_distance, _, current_node = queue.get()

        if current_node == end:
            reconstruct_path(prev_node, end, draw)
            search_time = time.time() - start_time
            return True

        for neighbor in current_node.neighbors:
            temp_distance = current_distance + 1  # Costo unitario para cada movimiento
            if temp_distance < distances[neighbor]:
                distances[neighbor] = temp_distance
                prev_node[neighbor] = current_node
                count += 1
                queue.put((temp_distance, count, neighbor))
                if neighbor != start and neighbor != end:
                    neighbor.make_visited()
                visited_nodes += 1
        draw()
    search_time = time.time() - start_time
    return False

# Reconstrucción del camino óptimo
def reconstruct_path(prev_node, current, draw):
    global path_length
    while prev_node[current]:
        current = prev_node[current]
        if current != start:
            current.make_path()
        path_length += 1
        draw()

def main():
    global start, end
    nodes = create_graph(ROWS, GAP)
    run = True
    clock = pygame.time.Clock()

    while run:
        # Actualizar vecinos de cada nodo para reflejar cambios (por ejemplo, si se ha marcado un obstáculo)
        for node in nodes.values():
            node.update_neighbors(nodes, ROWS)
        draw(WINDOW, nodes, ROWS, GAP)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

            # Click izquierdo: asigna nodo de inicio, fin u obstáculo
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes)
                if node:
                    if not start and node != end:
                        start = node
                        start.make_start()
                    elif not end and node != start:
                        end = node
                        end.make_end()
                    elif node != start and node != end:
                        node.make_barrier()
                        # Al marcar un obstáculo se actualizan inmediatamente los vecinos
                        for n in nodes.values():
                            n.update_neighbors(nodes, ROWS)

            # Click derecho: resetea el nodo
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes)
                if node:
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None
                    for n in nodes.values():
                        n.update_neighbors(nodes, ROWS)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    # Actualizamos vecinos antes de iniciar el algoritmo
                    for node in nodes.values():
                        node.update_neighbors(nodes, ROWS)
                    dijkstra_algorithm(lambda: draw(WINDOW, nodes, ROWS, GAP), nodes, start, end)
                    print("-----------------------------------------------")
                    print(f"Tiempo de búsqueda: {search_time:.4f} segundos")
                    print(f"Nodos analizados: {visited_nodes}")
                    print(f"Longitud del camino: {path_length} nodos")
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    nodes = create_graph(ROWS, GAP)

        clock.tick(120)
    pygame.quit()

if __name__ == "__main__":
    main()
