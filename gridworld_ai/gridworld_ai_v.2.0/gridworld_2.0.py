import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

# ENEMY_FOOD_MOVE = False  # Enemy and the Food moves
load_Q = True
save_Q = False
print("Q-Table Status: ")
print(f"Load:{load_Q} | Save:{save_Q}")

SIZE = 10
EPISODES = 25000
SHOW_EVERY = 2000
MAX_STEPS = 200
ACTION_SPACE = 8
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

coordinates = [(7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (0, 1), (0, 2), (0, 3), (0, 4),
               (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
               (5, 5), (6, 5), (7, 7), (7, 5), (7, 4),
               (7, 3), (5, 3), (7, 5), (4, 3), (3, 3),
               (2, 3), (5, 2), (9, 4), (9, 5), (9, 6),
               (9, 7), (8, 7), (7, 7), (6, 7), (5, 7),
               (4, 7), (3, 7), (1, 7), (3, 8), (5, 9), (6, 9), (8, 1)]

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
for i in range(41):
    j = f'block{i}'
    blocks.append(j)

if load_Q == True:
    iQ = "Q-Tables/Q.pickle"  # initial Q
else:
    iQ = None


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
        # else:
        #     if not x and not y:
        #         a = self.x
        #         self.x += np.random.randint(-1, 2)
        #     else:
        #         self.x += x

        #     if not y:
        #         b = self.y
        #         self.y += np.random.randint(-1, 2)
        #     else:
        #         self.y += y

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

if iQ == None:
    Q = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    Q[((x1, y1), (x2, y2))] = [np.random.uniform(-ACTION_SPACE, 0)
                                               for i in range(ACTION_SPACE)]
else:
    with open(iQ, 'rb') as f:
        Q = pickle.load(f)
        print("Q-Table LOADED!!!")

print("<------------------------------------------>")
print(f"Shape of the Q-Table is : {len(Q), ACTION_SPACE}")
print("<------------------------------------------>")

epsiode_rewards = []
food_counter = 0
enemy_counter = 0
for episode in range(EPISODES):
    player = Blob("player")
    food = Blob("food")
    enemy = Blob("enemy")

    if episode == 0:
        food_counter_ep = 0
        enemy_counter_ep = 0

    if episode % SHOW_EVERY == 0:
        print(
            f"episode:{episode} | Epsilon:{epsilon} | Mean Rewards:{np.mean(epsiode_rewards[-SHOW_EVERY:])} | Food Counter:{food_counter_ep} | Enemy Counter:{enemy_counter_ep}")
        RENDER = True
        food_counter_ep = 0
        enemy_counter_ep = 0
    else:
        RENDER = False

    episode_reward = 0
    for i in range(MAX_STEPS):
        state = (player - food, player - enemy)

        if np.random.random() < epsilon:
            action = np.random.randint(0, ACTION_SPACE)
        else:
            action = np.argmax(Q[state])

        player.step(action)
        food.move()
        enemy.move()

        # if ENEMY_FOOD_MOVE == True:
        #     enemy.move()
        #     food.move()
        # else:
        #     pass

        new_state = (player - food, player - enemy)

        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
            Q[state][action] = FOOD_REWARD
            food_counter += 1
            food_counter_ep += 1
            if episode % SHOW_EVERY == 0:
                print(f"The Agent ate the FOOD in {episode} episode")
        elif player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
            Q[state][action] = -ENEMY_PENALTY
            enemy_counter += 1
            enemy_counter_ep += 1
            if episode % SHOW_EVERY == 0:
                print(f"The Agent hit the ENEMY in {episode} episode")
        else:
            reward = -MOVE_PENALTY
            Q[state][action] = Q[state][action] + LEARNING_RATE * \
                (reward + GAMMA * np.max(Q[new_state]) - Q[state][action])

        if RENDER:
            env = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

            for blob in blobs:
                env[blob.x][blob.y] = d[BLOCK_N]

            env[player.x][player.y] = d[PLAYER_N]
            env[food.x][food.y] = d[FOOD_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]

            img = Image.fromarray(env, "RGBA")
            img = img.resize((500, 500))
            cv2.imshow("AI Agent - eat the food", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(3000) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(40) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    epsiode_rewards.append(episode_reward)
    epsilon -= EPSILON_DECAY

if save_Q:
    with open(f"Q-Tables/ .pickle", "wb") as f:
        pickle.dump(Q, f)
        print("Q-Table SAVED!!!")
else:
    print("Q-Table NOT SAVED!!!")

print("TRAINING DONE!!!")
print("<----------------------------------------------------------------------------------------------------->")
print("                                    REPORT:                                               ")
print(
    f"Mean Reward:{np.mean(epsiode_rewards)} | Food Counter:{food_counter} | Enemy Counter:{enemy_counter}")
print("<----------------------------------------------------------------------------------------------------->")
