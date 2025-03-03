import pygame
import math
import time

# Configuraciones generales
WIDTH = 800
ROWS = 50
BACKGROUND_COLOR = (255, 255, 255)
EDGE_COLOR = (200, 200, 200)
OUTLINE_COLOR = (0, 0, 0)

# Definición de colores
START_COLOR = (0, 125, 255)
END_COLOR = (0, 255, 0)
OBSTACLE_COLOR = (255, 0, 0)
OPEN_COLOR = (255, 125, 0)
CLOSED_COLOR = (255, 175, 0)
PATH_COLOR = (0, 0, 255)

# Inicializar pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Camino más corto Algoritmo 2 (Grafo)")

class Node:
    def __init__(self, row, col, gap):
        self.row = row
        self.col = col
        # Posición central para dibujar el círculo
        self.x = col * gap + gap // 2
        self.y = row * gap + gap // 2
        self.gap = gap
        # Radio pequeño para que se vean los enlaces
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
        # Dibuja el nodo como un círculo (relleno y contorno)
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(win, OUTLINE_COLOR, (self.x, self.y), self.radius, 1)

    def get_center(self):
        # Retorna la posición central del nodo (para dibujar enlaces)
        return (self.x, self.y)

    def update_neighbors(self, nodes_dict, total_rows):
        # Si el nodo es obstáculo, no tendrá vecinos
        if self.is_obstacle():
            self.neighbors = []
            return
        self.neighbors = []
        # Direcciones: vertical, horizontal y diagonales
        directions = [
            (-1,  0), (1,  0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for d in directions:
            new_row = self.row + d[0]
            new_col = self.col + d[1]
            if 0 <= new_row < total_rows and 0 <= new_col < total_rows:
                neighbor = nodes_dict[(new_row, new_col)]
                if not neighbor.is_obstacle():
                    # Costo 1 para movimientos rectos, √2 para diagonales
                    cost = 1 if d[0] == 0 or d[1] == 0 else math.sqrt(2)
                    self.neighbors.append((neighbor, cost))

def heuristic(node1, node2):
    # Distancia euclidiana entre los centros de los nodos
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def reconstruct_path(came_from, current, draw):
    path_length = 0
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
        path_length += 1
    return path_length

def algorithm(draw, nodes_dict, start, end):
    analyzed_nodes = 0
    open_set = [(start, 0)]
    came_from = {}
    closed_set = set()
    start_time = time.time()

    while open_set:
        # Ordenar por costo (heurística + costo de movimiento)
        open_set.sort(key=lambda x: x[1])
        current, _ = open_set.pop(0)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw)
            elapsed_time = time.time() - start_time
            return elapsed_time, analyzed_nodes, path_length

        closed_set.add(current)
        for neighbor, move_cost in current.neighbors:
            if neighbor in closed_set:
                continue

            total_cost = move_cost + heuristic(neighbor, end)
            if neighbor not in [item[0] for item in open_set]:
                came_from[neighbor] = current
                analyzed_nodes += 1
                open_set.append((neighbor, total_cost))
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return None, analyzed_nodes, 0

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

def draw(win, nodes_dict, rows, gap):
    win.fill(BACKGROUND_COLOR)
    # Dibujar los enlaces entre nodos
    for node in nodes_dict.values():
        for neighbor, _ in node.neighbors:
            # Para evitar duplicar, dibujamos el enlace solo si (row, col) de node es menor que la de neighbor
            if (node.row, node.col) < (neighbor.row, neighbor.col):
                # Si ambos nodos forman parte del camino óptimo, el enlace se dibuja con PATH_COLOR
                if node.color == PATH_COLOR and neighbor.color == PATH_COLOR:
                    link_color = PATH_COLOR
                else:
                    link_color = EDGE_COLOR
                pygame.draw.line(win, link_color, node.get_center(), neighbor.get_center(), 2)
    # Dibujar los nodos
    for node in nodes_dict.values():
        node.draw(win)
    pygame.display.update()

def get_clicked_node(pos, nodes_dict):
    mx, my = pos
    for node in nodes_dict.values():
        distance = math.sqrt((node.x - mx) ** 2 + (node.y - my) ** 2)
        if distance < node.radius:
            return node
    return None

def main(win, width):
    clock = pygame.time.Clock()
    nodes_dict, gap = make_graph(ROWS, width)
    start, end = None, None

    run = True
    while run:
        # Actualizar vecinos de cada nodo para reflejar cambios (como obstáculos)
        for node in nodes_dict.values():
            node.update_neighbors(nodes_dict, ROWS)
        draw(win, nodes_dict, ROWS, gap)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Clic izquierdo: asignar nodo de inicio, fin u obstáculo
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
                        # Al marcar obstáculo, actualizamos vecinos inmediatamente
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
                    for n in nodes_dict.values():
                        n.update_neighbors(nodes_dict, ROWS)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    elapsed_time, analyzed_nodes, path_length = algorithm(
                        lambda: draw(win, nodes_dict, ROWS, gap), nodes_dict, start, end
                    )
                    if elapsed_time is not None:
                        print("-" * 100)
                        print(f"Tiempo de búsqueda: {elapsed_time:.4f} segundos")
                        print(f"Nodos analizados: {analyzed_nodes}")
                        print(f"Longitud del camino: {path_length} nodos")
                if event.key == pygame.K_c:
                    start, end = None, None
                    nodes_dict, gap = make_graph(ROWS, width)

        clock.tick(120)
    pygame.quit()

main(win, WIDTH)
