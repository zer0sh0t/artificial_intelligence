'''
press left mouse button to fill the cell with life
hit enter to start/stop the evolution
hit space to clear the grid
'''

import pygame
import random

w = 1500 # width
h = 800 # height
res = 10 # resolution
rand_init = False # random initialization
pygame.init()
win = pygame.display.set_mode((w, h))
pygame.display.set_caption('game_of_life')

class Cell():
    def __init__(self, state, x, y, res):
        self.state = state
        self.rect = (x, y, res, res)
        self.purple = (148, 0, 211)
        self.black = (0, 0, 0)
        
    def draw(self, win):
        if self.state == 0:
            pygame.draw.rect(win, self.black, self.rect)
        elif self.state == 1:
            pygame.draw.rect(win, self.purple, self.rect)
            
class Grid():
    def __init__(self, win, rows, cols, res, rand_init):
        self.win = win
        self.rows = rows
        self.cols = cols
        
        if rand_init:
            self.curr_grid = [[Cell(random.randint(0, 1), i * res, j * res, res) for j in range(cols)] for i in range(rows)]
        else:    
            self.curr_grid = [[Cell(0, i * res, j * res, res) for j in range(cols)] for i in range(rows)]
        self.next_states = [[None for _ in range(cols)] for _ in range(rows)]
        
    def draw(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.curr_grid[i][j].draw(self.win)
       
    def count_neighbors(self, x, y):
        num_neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0: # if this is not the current cell
                    row = (x + i + self.rows) % self.rows # wrap-around
                    col = (y + j + self.cols) % self.cols
                    num_neighbors += self.curr_grid[row][col].state
                    
                    if num_neighbors > 3:
                        return num_neighbors
        return num_neighbors
                
    def update(self):
        for i in range(self.rows):
            for j in range(self.cols):
                curr_state = self.curr_grid[i][j].state
                num_neighbors = self.count_neighbors(i, j)
                
                if curr_state == 0 and num_neighbors == 3:
                    self.next_states[i][j] = 1
                elif curr_state == 1 and (num_neighbors < 2 or num_neighbors > 3):
                    self.next_states[i][j] = 0
                else:
                    self.next_states[i][j] = curr_state
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.curr_grid[i][j].state = self.next_states[i][j]
                
    def clear(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.curr_grid[i][j].state = 0
            
def main(win, h, w, res, rand_init):
    clock = pygame.time.Clock()
    clock.tick(60)
    rows = w // res
    cols = h // res
    grid = Grid(win, rows, cols, res, rand_init)
    update_flag = False    
    gen = 0
    font = pygame.font.SysFont('ubuntumono', 25)

    while True:
        text = font.render(f'gen:{gen}', True, (148, 0, 211), (0, 0, 0))
        win.blit(text, (12, 7))
        pygame.display.flip()
        
        grid.draw()
        if update_flag:
            grid.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                x, y = round(pos[0] / res), round(pos[1] / res)
                if 0 <= x < rows and 0 <= y < cols:
                    grid.curr_grid[x][y].state = 1
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grid.clear()
                    gen = 0
                    update_flag = False
                    
                elif event.key == pygame.K_RETURN:
                    if update_flag == False:
                        update_flag = True
                    else:
                        update_flag = False
                        
        if update_flag:
            gen += 1
        pygame.display.update()
                
if __name__ == '__main__':
    main(win, h, w, res, rand_init)