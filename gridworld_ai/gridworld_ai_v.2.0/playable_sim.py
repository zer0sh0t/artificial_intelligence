from PIL import Image
import cv2
import numpy as np

SIZE = 10
EPISODES = 25000
SHOW_EVERY = 2000
MAX_STEPS = 200
ACTION_SPACE = 9
MOVE_PENALTY = 1
ENEMY_PENALTY = 25
FOOD_REWARD = 300
epsilon = 0.7
HALF_EPISODE = EPISODES//2
EPSILON_DECAY = epsilon/(HALF_EPISODE - 1)
LEARNING_RATE = 0.3
GAMMA = 0.95
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
BLOCK_N = 4
d = {1: (255, 0, 0, 1), 2: (0, 255, 0, 1), 3: (
    0, 0, 255, 1), 4: (255, 100, 255, 1)}

coordinates = [(8, 0), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (0, 1), (0, 2), (0, 3), (0, 4),
               (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
               (5, 5), (6, 5), (7, 7), (7, 5), (7, 4),
               (7, 3), (5, 3), (7, 5), (4, 3), (3, 3),
               (2, 3), (5, 2), (9, 4), (9, 5), (9, 6),
               (9, 7), (8, 7), (7, 7), (6, 7), (5, 7),
               (4, 7), (3, 7), (1, 7), (3, 8), (5, 9), (6, 9)]

available = []
for i in range(SIZE):
    for j in range(SIZE):
        c1 = i
        c2 = j
        c = (c1, c2)
        if c not in coordinates:
            available.append(c)

available = np.array(available)

blocks = []
for i in range(40):
    j = f'block{i}'
    blocks.append(j)


class Blob():
    def __init__(self, name, x=False, y=False):
        self.name = name

        if self.name in blocks:
            self.x = x
            self.y = y

        if self.name == "player" or self.name == "food" or self.name == "enemy":
            num = np.random.randint(0, 40)
            self.x = available[num][0]
            self.y = available[num][1]
        else:
            pass

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def step(self, choice):
        if choice == 0:
            self.move(x=1, y=0)
        if choice == 1:
            self.move(x=-1, y=0)
        if choice == 2:
            self.move(x=0, y=1)
        if choice == 3:
            self.move(x=0, y=-1)
        if choice == 4:
            self.move(x=1, y=1)
        if choice == 5:
            self.move(x=-1, y=1)
        if choice == 6:
            self.move(x=1, y=-1)
        if choice == 7:
            self.move(x=-1, y=-1)
        if choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if self.name == "player":
            a = self.x
            b = self.y
            self.x += x
            self.y += y
            if (self.x, self.y) in coordinates:
                self.x = a
                self.y = b

        elif self.name == "enemy":
            self.x += np.random.randint(-1, 2)
            self.y += np.random.randint(-1, 2)

        else:
            a = self.x
            b = self.y
            self.x += np.random.randint(-1, 2)
            self.y += np.random.randint(-1, 2)
            if (self.x, self.y) in coordinates:
                self.x = a
                self.y = b

        if self.x <= 0:
            self.x = 0
        elif self.x >= SIZE-1:
            self.x = SIZE-1

        if self.y <= 0:
            self.y = 0
        elif self.y >= SIZE-1:
            self.y = SIZE-1


blobs = []
for i, block in enumerate(blocks):
    j = Blob(block, coordinates[i][0], coordinates[i][1])
    blobs.append(j)

player = Blob("player")
food = Blob("food")
enemy = Blob("enemy")


# while True:
for i in range(100):
    # action = np.random.randint(0, ACTION_SPACE)
    key = cv2.waitKey(0)

    if key == ord('w'):
        action = 1
    elif key == ord('a'):
        action = 3
    elif key == ord('s'):
        action = 0
    elif key == ord('d'):
        action = 2
    elif key == ord('q'):
        break
    else:
        action = 8

    player.step(action)

    env = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    for blob in blobs:
        env[blob.x][blob.y] = d[BLOCK_N]

    food.move()
    enemy.move()
    env[player.x][player.y] = d[PLAYER_N]
    env[food.x][food.y] = d[FOOD_N]
    env[enemy.x][enemy.y] = d[ENEMY_N]

    img = Image.fromarray(env, "RGBA")
    img = img.resize((500, 500))
    cv2.imshow("pacman", np.array(img))

    if player.x == food.x and player.y == food.y:
        print("food")
        break
    elif player.x == enemy.x and player.y == enemy.y:
        print("enemy")
        break
    # cv2.waitKey(10)

print("steps ended")
