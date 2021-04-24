import time
import numpy as np

board0 = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7]
]


board1 = [
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 8, 0, 0, 0, 7, 0, 9, 0],
    [6, 0, 2, 0, 0, 0, 5, 0, 0],
    [0, 7, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 5, 0, 0, 0, 6, 0, 3],
    [0, 9, 0, 4, 0, 0, 0, 7, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0]
]

board2 = []


def empty_pos(b):
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == 0:
                return (i, j)

    return None


def is_valid(b, num, pos):
    for i in range(len(b[0])):
        if b[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(b)):
        if b[i][pos[1]] == num and pos[0] != i:
            return False

    x = pos[1] // 3
    y = pos[0] // 3

    for i in range(y*3, y*3 + 3):
        for j in range(x*3, x*3 + 3):
            if b[i][j] == num and (i, j) != pos:
                return False

    return True


def backtrack(b):
    empty = empty_pos(b)
    if not empty:
        return True
    else:
        r, c = empty

    for i in range(1, 10):
        if is_valid(b, i, (r, c)):
            b[r][c] = i

            if backtrack(b):
                return True

            b[r][c] = 0

    return False


def print_board(b):
    for i in range(len(b)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(b[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end=" ")

            if j == 8:
                print(b[i][j])
            else:
                print(str(b[i][j]) + " ", end="")


def user_input():
    print("enter the entries in a single line(separated by space, enter 0 if empty): ")
    entries = list(map(int, input().split()))
    matrix = np.array(entries).reshape(9, 9)
    return matrix


board2 = user_input()
print("")
print_board(board2)
print("")

start = time.time()
backtrack(board2)
end = time.time()
print(f"time taken to solve: {end - start} seconds")

print("")
print_board(board2)
print("")

#0 0 0 9 0 0 0 6 7 0 9 0 0 0 0 2 0 8 4 6 0 0 7 8 0 0 0 3 2 0 0 9 4 0 7 0 7 0 0 6 0 3 0 0 2 0 1 0 7 8 0 0 4 3 0 0 0 8 5 0 0 1 6 5 0 1 0 0 0 0 9 0 6 7 0 0 0 9 0 0 0

# board0 for user_input
# 7 8 0 4 0 0 1 2 0 6 0 0 0 7 5 0 0 9 0 0 0 6 0 1 0 7 8 0 0 7 0 4 0 2 6 0 0 0 1 0 5 0 9 3 0 9 0 4 0 6 0 0 0 5 0 7 0 3 0 0 0 1 2 1 2 0 0 0 7 4 0 0 0 4 9 2 0 6 0 0 7
