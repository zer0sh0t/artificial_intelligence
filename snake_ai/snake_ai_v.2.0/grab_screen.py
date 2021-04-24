import pyautogui
import win32gui

# windows_list = []
# toplist = []
# def enum_win(hwnd, result):
#     win_text = win32gui.GetWindowText(hwnd)
#     windows_list.append((hwnd, win_text))
# win32gui.EnumWindows(enum_win, toplist)

def screenshot(window_title=None):
    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            return im
        else:
            print('Window not found!')
    else:
        im = pyautogui.screenshot()
        return im

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # print(windows_list)
    # im = screenshot("none")
    im = screenshot("Snake")
    imgs = []
    while im is not None:
        im = screenshot("Snake")
        im = Image.fromarray(np.uint8(im)).convert("RGB")
        im = im.resize((400, 400))
        im = im.crop((0, 30, 400, 400))
        imgs.append(im)
        if len(imgs) > 5:
            break
        # print(im.shape)
        # plt.imshow(im)
        # plt.show()

    print(imgs[0].size)
    for i in range(5):
        plt.imshow(imgs[i], cmap="gray")
        plt.show()