import pygame
import math
import time
from queue import PriorityQueue

# Configuraciones generales
WIDTH = 800
ROWS = 15  # Cantidad de nodos en cada dirección
BACKGROUND_COLOR = (255, 255, 255)

# Definición de colores
START_COLOR = (0, 125, 255)
END_COLOR = (0, 255, 0)
OBSTACLE_COLOR = (255, 0, 0)
OPEN_COLOR = (255, 125, 0)
CLOSED_COLOR = (255, 175, 0)
PATH_COLOR = (0, 0, 255)
EDGE_COLOR = (200, 200, 200)
OUTLINE_COLOR = (0, 0, 0)  # Color del contorno de los nodos

# Inicializar pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Camino más corto con A* (Grafo)")

class Node:
    def __init__(self, row, col, gap):
        self.row = row
        self.col = col
        # Posición central del nodo para dibujarlo
        self.x = col * gap + gap // 2
        self.y = row * gap + gap // 2
        self.gap = gap
        # Se reduce el tamaño para ver mejor los enlaces
        self.radius = gap // 4  
        self.color = BACKGROUND_COLOR
        self.neighbors = []  # Lista de tuplas: (nodo_vecino, costo)

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == CLOSED_COLOR

    def is_open(self):
        return self.color == OPEN_COLOR

    def is_obstacle(self):
        return self.color == OBSTACLE_COLOR

    def is_start(self):
        return self.color == START_COLOR

    def is_end(self):
        return self.color == END_COLOR

    def reset(self):
        self.color = BACKGROUND_COLOR

    def make_start(self):
        self.color = START_COLOR

    def make_closed(self):
        self.color = CLOSED_COLOR

    def make_open(self):
        self.color = OPEN_COLOR

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR

    def make_end(self):
        self.color = END_COLOR

    def make_path(self):
        self.color = PATH_COLOR

    def draw(self, win):
        # Se dibuja el nodo siempre (círculo relleno y contorno)
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(win, OUTLINE_COLOR, (self.x, self.y), self.radius, 1)

    def update_neighbors(self, nodes_dict, total_rows):
        # Si el nodo es obstáculo, no tiene vecinos
        if self.is_obstacle():
            self.neighbors = []
            return
        self.neighbors = []
        # Lista de movimientos: (d_row, d_col)
        directions = [
            (-1,  0), (1,  0), (0, -1), (0, 1),      # Arriba, abajo, izquierda, derecha
            (-1, -1), (-1, 1), (1, -1), (1, 1)         # Diagonales
        ]
        for d in directions:
            new_row, new_col = self.row + d[0], self.col + d[1]
            if 0 <= new_row < total_rows and 0 <= new_col < total_rows:
                neighbor = nodes_dict[(new_row, new_col)]
                # Se ignoran vecinos que sean obstáculo
                if not neighbor.is_obstacle():
                    # Costo 1 para movimientos rectos y √2 para diagonales
                    cost = 1 if d[0] == 0 or d[1] == 0 else math.sqrt(2)
                    self.neighbors.append((neighbor, cost))

def heuristic(node1, node2):
    # Distancia euclidiana entre centros de nodos
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def reconstruct_path(came_from, current, draw):
    path_length = 0
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
        path_length += 1
    return path_length

def algorithm(draw, nodes_dict, total_rows, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    # Inicialización de scores para cada nodo
    g_score = {node: float("inf") for node in nodes_dict.values()}
    g_score[start] = 0
    f_score = {node: float("inf") for node in nodes_dict.values()}
    f_score[start] = heuristic(start, end)

    open_set_hash = {start}
    start_time = time.time()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw)
            elapsed_time = time.time() - start_time
            return elapsed_time, count, path_length

        for neighbor, move_cost in current.neighbors:
            tentative_g_score = g_score[current] + move_cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return None, count, 0

def make_graph(rows, width):
    """
    Crea un diccionario de nodos distribuidos en forma de cuadrícula.
    Las llaves son (row, col) y el valor es el objeto Node.
    """
    nodes = {}
    gap = width // rows
    for i in range(rows):
        for j in range(rows):
            node = Node(i, j, gap)
            nodes[(i, j)] = node
    return nodes, gap

def draw(win, nodes_dict, gap, rows):
    win.fill(BACKGROUND_COLOR)
    # Dibujar los enlaces (cada enlace se dibuja una sola vez)
    for node in nodes_dict.values():
        for neighbor, _ in node.neighbors:
            if (node.row, node.col) < (neighbor.row, neighbor.col):
                # Si ambos nodos son parte del camino óptimo, usar PATH_COLOR para el enlace
                if node.color == PATH_COLOR and neighbor.color == PATH_COLOR:
                    link_color = PATH_COLOR
                else:
                    link_color = EDGE_COLOR
                pygame.draw.line(win, link_color, (node.x, node.y), (neighbor.x, neighbor.y), 1)
    # Dibujar los nodos
    for node in nodes_dict.values():
        node.draw(win)
    pygame.display.update()

def get_clicked_node(pos, nodes_dict):
    """
    Dado la posición del mouse, devuelve el nodo cuyo centro se encuentre
    lo suficientemente cerca de dicha posición.
    """
    mx, my = pos
    for node in nodes_dict.values():
        distance = math.sqrt((node.x - mx) ** 2 + (node.y - my) ** 2)
        if distance < node.radius:
            return node
    return None

def main(win, width):
    nodes_dict, gap = make_graph(ROWS, width)
    start, end = None, None

    run = True
    while run:
        # Actualizar los vecinos de cada nodo para reflejar cambios (como obstáculos)
        for node in nodes_dict.values():
            node.update_neighbors(nodes_dict, ROWS)
        draw(win, nodes_dict, gap, ROWS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Clic izquierdo: asignar inicio, fin o marcar obstáculo
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes_dict)
                if node:
                    if not start and node != end:
                        start = node
                        start.make_start()
                    elif not end and node != start:
                        end = node
                        end.make_end()
                    elif node != start and node != end:
                        node.make_obstacle()
                        # Actualizamos inmediatamente los vecinos al marcar un obstáculo
                        for n in nodes_dict.values():
                            n.update_neighbors(nodes_dict, ROWS)

            # Clic derecho: resetear el nodo
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes_dict)
                if node:
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None
                    # Actualizamos vecinos luego de resetear
                    for n in nodes_dict.values():
                        n.update_neighbors(nodes_dict, ROWS)

            if event.type == pygame.KEYDOWN:
                # Al presionar SPACE se inicia la búsqueda con A*
                if event.key == pygame.K_SPACE and start and end:
                    elapsed_time, analyzed_nodes, path_length = algorithm(
                        lambda: draw(win, nodes_dict, gap, ROWS), nodes_dict, ROWS, start, end
                    )
                    if elapsed_time is not None:
                        print("-" * 100)
                        print(f"Tiempo de búsqueda: {elapsed_time:.4f} segundos")
                        print(f"Nodos analizados: {analyzed_nodes}")
                        print(f"Longitud del camino: {path_length} nodos")
                # Con la tecla 'c' se limpia el grafo
                if event.key == pygame.K_c:
                    start, end = None, None
                    nodes_dict, gap = make_graph(ROWS, width)

    pygame.quit()

main(win, WIDTH)
