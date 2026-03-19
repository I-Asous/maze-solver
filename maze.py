import cl
import pygame
import random
from cl import ChannelSet, StimDesign

stim_design = StimDesign(160, -1.0, 160, 1.0)

CHANNEL_MAP = {
    0: ChannelSet(10, 11),
    1: ChannelSet(20, 21),
    2: ChannelSet(30, 31),
    3: ChannelSet(40, 41),
}

DIRECTIONS = {
    0: (-1, 0),  # up
    1: ( 1, 0),  # down
    2: ( 0,-1),  # left
    3: ( 0, 1),  # right
}

def generate_maze(rows, cols):
    #Generating a random maze using DFS, guaranteeing a path from (0,0) to (rows-1, cols-1)
    maze = [[1] * cols for _ in range(rows)]

    def carve(r, c):
        maze[r][c] = 0
        directions = [(0,2),(0,-2),(2,0),(-2,0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 1:
                maze[r + dr//2][c + dc//2] = 0
                carve(nr, nc)

    carve(0, 0)
    maze[0][0] = 0
    maze[rows-1][cols-1] = 0
    return maze

def is_valid(maze, r, c):
    rows = len(maze)
    cols = len(maze[0])
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0

def draw(screen, font, clock, maze, pos, trail, spike_counts, log, cell_size):
    rows = len(maze)
    cols = len(maze[0])
    panel_h = 190
    screen.fill((15, 15, 25))

    # Drawing the maze
    for r in range(rows):
        for c in range(cols):
            color = (40, 40, 60) if maze[r][c] == 0 else (80, 30, 30)
            pygame.draw.rect(screen, color,
                (c*cell_size+2, r*cell_size+2, cell_size-4, cell_size-4),
                border_radius=6)

    # Drawing trail
    for tr, tc in trail:
        pygame.draw.rect(screen, (30, 80, 120),
            (tc*cell_size+cell_size//4, tr*cell_size+cell_size//4,
             cell_size//2, cell_size//2), border_radius=4)

    # Drawing the goal
    gr, gc = rows-1, cols-1
    pygame.draw.rect(screen, (50, 200, 100),
        (gc*cell_size+cell_size//5, gr*cell_size+cell_size//5,
         cell_size*3//5, cell_size*3//5), border_radius=6)

    # Draw the agent
    pr, pc = pos
    pygame.draw.circle(screen, (255, 200, 50),
        (pc*cell_size + cell_size//2, pr*cell_size + cell_size//2),
        cell_size//3)

    # Spike panel
    maze_pixel_h = rows * cell_size
    panel_y = maze_pixel_h
    pygame.draw.rect(screen, (20, 20, 35), (0, panel_y, cols*cell_size, panel_h))
    label = font.render("Neural Spike Activity (ch 0-63)", True, (180, 180, 200))
    screen.blit(label, (10, panel_y + 5))

    max_s = max(spike_counts.values()) or 1
    bar_area_w = cols * cell_size
    for ch, count in spike_counts.items():
        bar_h = int((count / max_s) * 60)
        x = int(ch * (bar_area_w / 64))
        pygame.draw.rect(screen, (80, 160, 255),
            (x, panel_y + 75 - bar_h, max(2, int(bar_area_w/64)-1), bar_h))

    dir_labels = {0: "UP", 1: "DN", 2: "LT", 3: "RT"}
    dir_colors = {0: (255,100,100), 1: (100,255,100), 2: (100,100,255), 3: (255,255,100)}
    for d, label_str in dir_labels.items():
        txt = font.render(f"{label_str}={spike_counts.get(10+d*10,0)}", True, dir_colors[d])
        screen.blit(txt, (10 + d * (bar_area_w//4), panel_y + 85))

    for i, entry in enumerate(log[-3:]):
        t = font.render(entry, True, (150, 200, 150))
        screen.blit(t, (10, panel_y + 120 + i * 20))

    pygame.display.flip()
    clock.tick(30)

def pick_direction(spike_counts):
    scores = {d: sum(spike_counts.get(10 + d*10 + offset, 0) for offset in [0, 1]) for d in range(4)}
    return max(scores, key=scores.get)

def get_maze_size():
    print("\n=== Neural Maze Solver ===")
    while True:
        try:
            rows = int(input("Enter maze rows (min 3, max 20): "))
            cols = int(input("Enter maze cols (min 3, max 20): "))
            if 3 <= rows <= 20 and 3 <= cols <= 20:
                return rows, cols
            print("Values must be between 3 and 20.")
        except ValueError:
            print("Please enter valid integers.")

def main():
    rows, cols = get_maze_size()
    maze = generate_maze(rows, cols)

    START = (0, 0)
    GOAL  = (rows-1, cols-1)

    # Fit cell size to screen (max 800px wide, 600px maze height)
    cell_size = min(800 // cols, 600 // rows)
    cell_size = max(cell_size, 20)  # minimum readable size

    screen_w = cols * cell_size
    screen_h = rows * cell_size + 190

    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Neural Maze Solver ({rows}x{cols})")
    font  = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    spike_counts = {i: 0 for i in range(64)}
    pos        = list(START)
    trail      = [tuple(pos)]
    log        = []
    tick_count = 0

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=120):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            for spike in tick.analysis.spikes:
                spike_counts[spike.channel % 64] += 1

            if tick_count % 10 == 0 and tuple(pos) != GOAL:
                direction = pick_direction(spike_counts)
                dr, dc = DIRECTIONS[direction]
                nr, nc = pos[0] + dr, pos[1] + dc

                if is_valid(maze, nr, nc):
                    pos = [nr, nc]
                    trail.append(tuple(pos))
                    log.append(f"Tick {tick_count}: moved {'↑↓←→'[direction]} → {tuple(pos)}")
                else:
                    neurons.stim(CHANNEL_MAP[direction], stim_design)
                    log.append(f"Tick {tick_count}: wall! stimulating dir {direction}")

                spike_counts = {i: 0 for i in range(64)}

            draw(screen, font, clock, maze, pos, trail, spike_counts, log, cell_size)
            tick_count += 1

            if tuple(pos) == GOAL:
                log.append("We have MADE IT")
                draw(screen, font, clock, maze, pos, trail, spike_counts, log, cell_size)
                pygame.time.wait(3000)
                break

    pygame.quit()

if __name__ == '__main__':
    main()