"""
there are 3 types of configurations in the game:
you vs ai (type "1" for this)
you vs normal computer without ai (type "2" for this)
ai vs ai (type "3" for this)
"""

import pygame
import random
import math
import time

global configuration
configuration = input(
    "\nSelect Configuration:\n (player vs ai - 1) | (player vs normal computer - 2) | (ai vs ai - 3)\n (ai vs ai gameplay runs too fast for us to see so it's slowed down by more than 10000000%)\n ")

if int(configuration) != 3:
    print("\nplayer - O\nAI/Computer - X")


global counter
counter = 0

global p_turn, c_turn
who_starts = random.randint(0, 1)
if who_starts == 0:
    p_turn = True
    c_turn = False
elif who_starts == 1:
    p_turn = False
    c_turn = True

global win_configs
win_configs = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8],
               [2, 4, 6], [0, 3, 6], [1, 4, 7], [2, 5, 8]]

win = pygame.display.set_mode((600, 600))
win.fill((255, 255, 255))
pygame.display.set_caption("Tic Tac Toe AI")
black = (0, 0, 0)

board = [" " for _ in range(9)]
coordinates_p = [(100, 100), (300, 100), (500, 100), (100, 300),
                 (300, 300), (500, 300), (100, 500), (300, 500), (500, 500)]

sp_c = []
ep_c = []
for c in coordinates_p:
    c_x, c_y = c
    sp_c.append((c_x-50, c_y-50))
for c in coordinates_p:
    c_x, c_y = c
    ep_c.append((c_x+50, c_y+50))


def draw_board():
    pygame.draw.line(win, black, (200, 0), (200, 599), 2)
    pygame.draw.line(win, black, (400, 0), (400, 599), 2)
    pygame.draw.line(win, black, (0, 200), (599, 200), 2)
    pygame.draw.line(win, black, (0, 400), (599, 400), 2)


def draw_x(where, win, colors):
    global p_turn, c_turn, counter
    if board[where] == " ":
        start_pos = sp_c[where]
        end_pos = ep_c[where]
        sp_x, sp_y = start_pos
        ep_x, ep_y = end_pos
        board[where] = "x"

        pygame.draw.line(win, black, start_pos, end_pos, 2)
        pygame.draw.line(win, black, (sp_x, ep_y), (ep_x, sp_y), 2)

        counter += 1
        c_turn = False
        p_turn = True


def draw_circle(m, win, color):
    global p_turn, c_turn, counter
    if board[m] == " ":
        center = coordinates_p[m]
        board[m] = "o"

        pygame.draw.circle(win, color, center, 70, 2)

        counter += 1
        c_turn = True
        p_turn = False


def mouse_index():
    x, y = pygame.mouse.get_pos()
    if 0 < x < 200 and 0 < y < 200:
        m = 0
    elif 200 < x < 400 and 0 < y < 200:
        m = 1
    elif 400 < x < 600 and 0 < y < 200:
        m = 2
    elif 0 < x < 200 and 200 < y < 400:
        m = 3
    elif 200 < x < 400 and 200 < y < 400:
        m = 4
    elif 400 < x < 600 and 200 < y < 400:
        m = 5
    elif 0 < x < 200 and 400 < y < 600:
        m = 6
    elif 200 < x < 400 and 400 < y < 600:
        m = 7
    elif 400 < x < 600 and 400 < y < 600:
        m = 8

    return m


def show_board():
    print("")
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("-----------")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("-----------")
    print(f"{board[6]} | {board[7]} | {board[8]}")
    print("")


def player_turn():
    m = mouse_index()
    draw_circle(m, win, black)
    check_winner(True)


def computer_turn():
    go = True
    while go:
        where = random.randint(0, 8)
        if board[where] == " ":
            go = False

    draw_x(where, win, black)
    check_winner(True)


def ai_turn():
    best_score = -math.inf
    for i in range(len(board)):
        if board[i] == " ":
            board[i] = "x"
            score = minimax(0, False, "ai")
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    draw_x(best_move, win, black)
    check_winner(True)


def player_ai_turn():
    best_score = -math.inf
    for i in range(len(board)):
        if board[i] == " ":
            board[i] = "o"
            score = minimax(0, False, "player_ai")
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    draw_circle(best_move, win, black)
    check_winner(True)


def minimax(depth, is_maximizing, configuration):
    global p_won, c_won
    check_winner(False)

    if c_won:
        if configuration == "ai":
            score = 1
        elif configuration == "player_ai":
            score = -1
        return score

    elif p_won:
        if configuration == "ai":
            score = -1
        elif configuration == "player_ai":
            score = 1
        return score

    if board.count(" ") == 0:
        score = 0
        return score

    if is_maximizing:
        best_score = -math.inf
        for i in range(len(board)):
            if board[i] == " ":
                if configuration == "ai":
                    board[i] = "x"
                    score = minimax(depth+1, False, "ai")
                elif configuration == "player_ai":
                    board[i] = "o"
                    score = minimax(depth+1, False, "player_ai")

                board[i] = " "
                best_score = max(score, best_score)

        return best_score
    else:
        best_score = math.inf
        for i in range(len(board)):
            if board[i] == " ":
                if configuration == "ai":
                    board[i] = "o"
                    score = minimax(depth+1, True, "ai")
                elif configuration == "player_ai":
                    board[i] = "x"
                    score = minimax(depth+1, True, "player_ai")

                board[i] = " "
                best_score = min(score, best_score)

        return best_score


def check_winner(print_stat):
    global win_configs, run, tie, p_won, c_won, configuration
    o_p = []
    x_p = []
    p_won = False
    c_won = False
    tie = False

    for i in range(len(board)):
        if board[i] == "o":
            o_p.append(i)
        elif board[i] == "x":
            x_p.append(i)

    set2 = set(o_p)
    set3 = set(x_p)
    for config in win_configs:
        set1 = set(config)
        if set1.issubset(set2):
            p_won = True
        elif set1.issubset(set3):
            c_won = True

    if print_stat:
        if p_won:
            print("Player Won!!")

            show_board()
            run = False
        elif c_won:
            if int(configuration) == 2:
                print("Computer Won!!")
            elif int(configuration) == 1 or 3:
                print("AI Won!!")

            show_board()
            run = False
        else:
            tie = True


def run_game():
    global p_turn, c_turn, counter, run, tie, configuration
    clock = pygame.time.Clock()
    run = True
    while run:
        if int(configuration) == 3:
            clock.tick(0.3)

        if counter == 9:
            check_winner(True)
            if tie == True:
                show_board()
                print("It's a tie")
                run = False

        for event in pygame.event.get():
            if event == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]:
                if int(configuration) == 1 or 2:
                    if p_turn and run == True:
                        player_turn()

        if int(configuration) == 3:
            if p_turn and counter != 9 and run == True:
                player_ai_turn()

        if c_turn and counter != 9 and run == True:
            if int(configuration) == 2 or counter == 0:
                computer_turn()
            else:
                ai_turn()

        draw_board()

        pygame.display.update()


run_game()

