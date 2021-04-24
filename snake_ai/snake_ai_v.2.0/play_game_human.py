import random
import pygame
from collections import namedtuple

pygame.init()
font = pygame.font.Font('assets/arial.ttf', 25)
Point = namedtuple('Point', ('x', 'y'))

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
BLOCK_SIZE = 20
SPEED = 20


class SnakeGame():
    def __init__(self, w=640, h=480):
        pygame.display.set_caption("Snake")

        self.w, self.h = w, h
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.direction = "r"

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head]

        self.score = 0
        self.food = None
        self.place_food()

    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self.place_food()

    def play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = "l"
                elif event.key == pygame.K_RIGHT:
                    self.direction = "r"
                elif event.key == pygame.K_UP:
                    self.direction = "u"
                elif event.key == pygame.K_DOWN:
                    self.direction = "d"

        self.move()
        self.snake.insert(0, self.head)

        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        return game_over, self.score

    def is_collision(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True

        if self.head in self.snake[1:]:
            return True

        return False

    def update_ui(self):
        self.display.fill(BLACK)
        for p in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(
                p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self):
        x = self.head.x
        y = self.head.y

        if self.direction == "r":
            x += BLOCK_SIZE
        elif self.direction == "l":
            x -= BLOCK_SIZE
        elif self.direction == "d":
            y += BLOCK_SIZE
        elif self.direction == "u":
            y -= BLOCK_SIZE

        self.head = Point(x, y)


game = SnakeGame()
game_over = False
while not game_over:
    game_over, score = game.play()

print(f"score: {score}")
