import pygame
import sys
import math
import time
from queue import PriorityQueue

pygame.init()
WIDTH, HEIGHT = 800, 800
ROWS = 10           # Número de nodos en cada dirección
GAP = WIDTH // ROWS # Distancia entre nodos (para calcular posiciones)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Algoritmo 1 (Grafo)")

# Definición de colores
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
BLUE    = (0, 0, 255)      # Camino óptimo
RED     = (255, 0, 0)      # Obstáculo
CYAN    = (0, 125, 255)    # Inicio
GREEN   = (0, 255, 0)      # Fin
ORANGE  = (255, 125, 0)    # Abierto (en búsqueda)
GREY    = (255, 175, 0)    # Cerrado
EDGE_DEFAULT = (200, 200, 200)  # Color de enlace por defecto

# Estados (usando colores directamente)
EMPTY    = WHITE
START    = CYAN
END      = GREEN
OBSTACLE = RED
PATH     = BLUE
CLOSED   = GREY
OPEN     = ORANGE

# Variables globales para la búsqueda
start_node = None
end_node = None
visited_nodes = 0
path_length = 0
search_time = 0

class Node:
    def __init__(self, row, col, gap):
        self.row = row
        self.col = col
        # Posición central del nodo (para dibujar el círculo y los enlaces)
        self.x = col * gap + gap // 2
        self.y = row * gap + gap // 2
        self.gap = gap
        # Radio pequeño para que se visualicen los enlaces entre nodos
        self.radius = gap // 4
        self.color = EMPTY
        self.neighbors = []

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if other is None:
            return False
        return (self.row, self.col) == (other.row, other.col)

    def get_pos(self):
        return (self.row, self.col)

    def is_barrier(self):
        return self.color == OBSTACLE

    def reset(self):
        self.color = EMPTY

    def make_start(self):
        self.color = START

    def make_end(self):
        self.color = END

    def make_barrier(self):
        self.color = OBSTACLE

    def make_path(self):
        self.color = PATH

    def make_visited(self):
        self.color = CLOSED

    def make_open(self):
        self.color = OPEN

    def draw(self, win):
        # Dibuja el nodo como un círculo relleno con contorno negro
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(win, BLACK, (self.x, self.y), self.radius, 1)

    def get_center(self):
        return (self.x, self.y)

    def update_neighbors(self, nodes, total_rows):
        # Si es obstáculo, no tiene vecinos
        if self.is_barrier():
            self.neighbors = []
            return
        self.neighbors = []
        # Se consideran las 8 direcciones: cardinales y diagonales
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            new_row = self.row + d[0]
            new_col = self.col + d[1]
            if 0 <= new_row < total_rows and 0 <= new_col < total_rows:
                neighbor = nodes[(new_row, new_col)]
                if not neighbor.is_barrier():
                    self.neighbors.append(neighbor)

def create_graph(rows, gap):
    nodes = {}
    for i in range(rows):
        for j in range(rows):
            nodes[(i, j)] = Node(i, j, gap)
    return nodes

def draw_graph(win, nodes, total_rows, gap):
    win.fill(WHITE)
    # Dibujar enlaces entre nodos (solo una vez para cada par)
    for node in nodes.values():
        for neighbor in node.neighbors:
            if (node.row, node.col) < (neighbor.row, neighbor.col):
                # Si ambos nodos forman parte del camino óptimo, se usa el color PATH
                if node.color == PATH and neighbor.color == PATH:
                    line_color = PATH
                else:
                    line_color = EDGE_DEFAULT
                pygame.draw.line(win, line_color, node.get_center(), neighbor.get_center(), 2)
    # Dibujar los nodos
    for node in nodes.values():
        node.draw(win)
    pygame.display.update()

def get_clicked_node(pos, nodes):
    mx, my = pos
    for node in nodes.values():
        if math.sqrt((node.x - mx)**2 + (node.y - my)**2) < node.radius:
            return node
    return None

def heuristic(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def cost_between(node1, node2):
    dx = abs(node1.row - node2.row)
    dy = abs(node1.col - node2.col)
    if dx + dy == 1:
        return 1
    elif dx == 1 and dy == 1:
        return math.sqrt(2)
    else:
        return float('inf')

# MODIFICADO: Se cambia la condición del bucle para detenerse al llegar al nodo de inicio.
def reconstruct_path(came_from, current, draw):
    global path_length
    path_length = 0
    # Mientras no lleguemos al nodo de inicio, recorremos el camino
    while current != start_node:
        # Si por alguna razón current no tiene predecesor, salimos para evitar bucle infinito.
        if current not in came_from:
            break
        current = came_from[current]
        if current != start_node:
            current.make_path()
        path_length += 1
        draw()

def find_shortest_path(start, end, nodes, total_rows, gap, draw):
    global visited_nodes, search_time
    visited_nodes = 0
    search_time = 0
    start_time = time.time()
    open_set = {start: 0}  # Diccionario: nodo -> costo (f = g + h)
    came_from = {}
    counted_nodes = set()

    while open_set:
        current = min(open_set, key=open_set.get)
        if current == end:
            reconstruct_path(came_from, end, lambda: draw_graph(WIN, nodes, total_rows, gap))
            end_time = time.time()
            search_time = end_time - start_time
            print("---------------------------")
            print("¡Camino encontrado!")
            print(f"Tiempo: {search_time:.4f} segundos")
            print(f"Nodos analizados: {visited_nodes}")
            print(f"Longitud del camino: {path_length} nodos")
            end.make_end()
            return True

        open_set.pop(current)
        if current != start and current not in counted_nodes:
            visited_nodes += 1
            counted_nodes.add(current)
        for neighbor in current.neighbors:
            temp_cost = open_set.get(current, 0) + cost_between(current, neighbor)
            total_cost = temp_cost + heuristic(neighbor, end)
            if neighbor not in open_set and neighbor not in counted_nodes:
                open_set[neighbor] = total_cost
                came_from[neighbor] = current
                if neighbor != start and neighbor != end:
                    neighbor.make_open()
                if neighbor not in counted_nodes:
                    visited_nodes += 1
                    counted_nodes.add(neighbor)
        current.make_visited()
        draw_graph(WIN, nodes, total_rows, gap)
    end_time = time.time()
    search_time = end_time - start_time
    print("No se encontró camino.")
    print(f"Tiempo: {search_time:.4f} segundos")
    print(f"Nodos analizados: {visited_nodes}")
    return False

def reset_graph(nodes, total_rows):
    global start_node, end_node, visited_nodes, path_length, search_time
    for node in nodes.values():
        node.reset()
    start_node = None
    end_node = None
    visited_nodes = 0
    path_length = 0
    search_time = 0

def main():
    global start_node, end_node
    nodes = create_graph(ROWS, GAP)
    run = True
    clock = pygame.time.Clock()
    while run:
        # Actualizar vecinos de cada nodo (para que si se cambia un estado, se ajusten los enlaces)
        for node in nodes.values():
            node.update_neighbors(nodes, ROWS)
        draw_graph(WIN, nodes, ROWS, GAP)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
            # Click izquierdo: asigna inicio, fin u obstáculo
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes)
                if node:
                    if not start_node and node != end_node:
                        start_node = node
                        start_node.make_start()
                    elif not end_node and node != start_node:
                        end_node = node
                        end_node.make_end()
                    elif node != start_node and node != end_node:
                        node.make_barrier()
                        for n in nodes.values():
                            n.update_neighbors(nodes, ROWS)
            # Click derecho: resetea el nodo
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                node = get_clicked_node(pos, nodes)
                if node:
                    node.reset()
                    if node == start_node:
                        start_node = None
                    elif node == end_node:
                        end_node = None
                    for n in nodes.values():
                        n.update_neighbors(nodes, ROWS)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_node and end_node:
                    for n in nodes.values():
                        n.update_neighbors(nodes, ROWS)
                    find_shortest_path(start_node, end_node, nodes, ROWS, GAP, lambda: draw_graph(WIN, nodes, ROWS, GAP))
                if event.key == pygame.K_c:
                    reset_graph(nodes, ROWS)
        clock.tick(120)
    pygame.quit()

if __name__ == "__main__":
    main()
