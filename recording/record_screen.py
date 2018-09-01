# dirty testing of capturing game video with inputs
import win32api
import win32console
import win32gui
import pythoncom, pyHook
import argparse, pdb, time
import threading

import cv2
import mss
import numpy


keys = {
    'F9': 120,
    'W': 87,
    'A': 65,
    'S': 83,
    'D': 68,
    'M1': 1001
}

font = cv2.FONT_HERSHEY_SIMPLEX

wasd_keys = [keys['W'], keys['A'], keys['S'], keys['D']]

def parse_start_key(key):

    return keys[key.upper()]

parser = argparse.ArgumentParser("WASD Keylogger")
parser.add_argument("-s", "--start_stop", default="F9")
parser.add_argument("-g", "--game", default="WARFRAME")

args = parser.parse_args()
start_key = parse_start_key(args.start_stop)

running = False
started = False
capture_thread = None

state = {
    keys['W']: False,
    keys['A']: False,
    keys['S']: False,
    keys['D']: False,
    keys['M1']: False
}

def capture_game():
    global running

    game_hwnd = win32gui.FindWindow(None, args.game)

    game_w_rect = win32gui.GetWindowRect(game_hwnd)

    monitor = {
        'left': game_w_rect[0],
        'top': game_w_rect[1],
        'width': game_w_rect[2],
        'height': game_w_rect[3]
    }

    print('start')
    with mss.mss() as sct:
        while running:

            # Get raw pixels from the screen, save it to a Numpy array
            im = numpy.array(sct.grab(monitor))

            cv2.resize(im, dsize=(600, 800), interpolation=cv2.INTER_CUBIC)
            # Display the picture

            # Display the picture in grayscale
            # cv2.imshow('OpenCV/Numpy grayscale',
            #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            text = " ".join(['1' if state[x] else '0' for x in state.keys()])
            cv2.putText(im, text,(30,30), font, 2,(255,255,255),1,cv2.LINE_AA)

            cv2.imshow("test", im)

            cv2.waitKey(25)
            # time.sleep(0.025)

def handle_key(state, key_id, down):
    state[key_id] = down

    #print("{}: {}".format(
    #    time.perf_counter(),
    #    " ".join(['1' if state[x] else '0' for x in state.keys()])))

    return state


def OnKeyboardUpEvent(event):
    global running, state, capture_thread

    if event.KeyID == start_key:
        if running:
            runnig = False
            win32api.PostQuitMessage()
        else:
            running = True
            capture_thread = threading.Thread(target=capture_game, daemon=True)
            capture_thread.start()
            time.perf_counter()

    if running and event.KeyID in wasd_keys:
        state = handle_key(state, event.KeyID, False)

    return True


def OnKeyboardDownEvent(event):
    global state

    if running and event.KeyID in wasd_keys:
        state = handle_key(state, event.KeyID, True)

    return True

def OnMouseDown(event):
    global state

    if running:
        state = handle_key(state, keys['M1'], True)

    return True

def OnMouseUp(event):
    global state

    if running:
        state = handle_key(state, keys['M1'], True)

    return True


win = win32console.GetConsoleWindow()
# win32gui.ShowWindow(win, 0)
# create a hook manager object
hm = pyHook.HookManager()

hm.KeyDown = OnKeyboardDownEvent
hm.KeyUp = OnKeyboardUpEvent
hm.MouseLeftDown = OnMouseDown
hm.MouseLeftUp = OnMouseUp

# set the hook
hm.HookKeyboard()
hm.HookMouse()

# wait forever
pythoncom.PumpMessages()

running = False

print("end")
if capture_thread:
    capture_thread.join(1000)
