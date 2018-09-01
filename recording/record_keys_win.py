# Python code for keylogger
# to be used in windows
import win32api
import win32console
import win32gui
import pythoncom, pyHook
import argparse, pdb, time


keys = {
    'F9': 120,
    'W': 87,
    'A': 65,
    'S': 83,
    'D': 68,
    'M1': 1001
}

wasd_keys = [keys['W'], keys['A'], keys['S'], keys['D']]

def parse_start_key(key):

    return keys[key.upper()]

parser = argparse.ArgumentParser("WASD Keylogger")
parser.add_argument("-s", "--start_stop", default="F9")

args = parser.parse_args()
start_key = parse_start_key(args.start_stop)

running = False

state = {
    keys['W']: False,
    keys['A']: False,
    keys['S']: False,
    keys['D']: False,
    keys['M1']: False
}



def handle_key(state, key_id, down):
    state[key_id] = down

    print("{}: {}".format(
        time.perf_counter(),
        " ".join(['1' if state[x] else '0' for x in state.keys()])))

    return state

def OnKeyboardUpEvent(event):
    global running, state
 
    if event.KeyID == start_key:
        if running:
            quit()
        else:
            running = True
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
