import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Set up argument parsing
parser = argparse.ArgumentParser(description="Animate drawing text on a pixel canvas.")
parser.add_argument("--text", type=str, default="Hello", help="The text to animate drawing")
parser.add_argument("--speed", type=int, default=1, help="How many blocks to draw per frame (higher = faster)")
args = parser.parse_args()

text_to_draw = args.text
speed = args.speed

shape = (32, 32, 3)
arr = np.zeros(shape, dtype=int)

plt.ion()
fig, ax = plt.subplots(figsize=(4, 4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

im = ax.imshow(arr % 256, interpolation='nearest', vmin=0, vmax=255)

FONT = {
    'A': [".111.", "1...1", "1...1", "11111", "1...1", "1...1", "1...1"],
    'B': ["1111.", "1...1", "1...1", "1111.", "1...1", "1...1", "1111."],
    'C': [".111.", "1...1", "1....", "1....", "1....", "1...1", ".111."],
    'D': ["1111.", "1...1", "1...1", "1...1", "1...1", "1...1", "1111."],
    'E': ["11111", "1....", "1....", "111..", "1....", "1....", "11111"],
    'F': ["11111", "1....", "1....", "111..", "1....", "1....", "1...."],
    'G': [".111.", "1...1", "1....", "1..11", "1...1", "1...1", ".111."],
    'H': ["1...1", "1...1", "1...1", "11111", "1...1", "1...1", "1...1"],
    'I': [".111.", "..1..", "..1..", "..1..", "..1..", "..1..", ".111."],
    'J': ["..111", "...1.", "...1.", "...1.", "...1.", "1..1.", ".11.."],
    'K': ["1...1", "1..1.", "1.1..", "11...", "1.1..", "1..1.", "1...1"],
    'L': ["1....", "1....", "1....", "1....", "1....", "1....", "11111"],
    'M': ["1...1", "11.11", "1.1.1", "1.1.1", "1...1", "1...1", "1...1"],
    'N': ["1...1", "11..1", "1.1.1", "1..11", "1...1", "1...1", "1...1"],
    'O': [".111.", "1...1", "1...1", "1...1", "1...1", "1...1", ".111."],
    'P': ["1111.", "1...1", "1...1", "1111.", "1....", "1....", "1...."],
    'Q': [".111.", "1...1", "1...1", "1...1", "1.1.1", "1..1.", ".11.1"],
    'R': ["1111.", "1...1", "1...1", "1111.", "1.1..", "1..1.", "1...1"],
    'S': [".1111", "1....", "1....", ".111.", "....1", "....1", "1111."],
    'T': ["11111", "..1..", "..1..", "..1..", "..1..", "..1..", "..1.."],
    'U': ["1...1", "1...1", "1...1", "1...1", "1...1", "1...1", ".111."],
    'V': ["1...1", "1...1", "1...1", "1...1", ".1.1.", ".1.1.", "..1.."],
    'W': ["1...1", "1...1", "1...1", "1.1.1", "1.1.1", "11.11", "1...1"],
    'X': ["1...1", "1...1", ".1.1.", "..1..", ".1.1.", "1...1", "1...1"],
    'Y': ["1...1", "1...1", ".1.1.", "..1..", "..1..", "..1..", "..1.."],
    'Z': ["11111", "....1", "...1.", "..1..", ".1...", "1....", "11111"],
    ' ': [".....", ".....", ".....", ".....", ".....", ".....", "....."]
}

def paint_large_letter(char, scale=4, color_val=220):
    matrix = FONT.get(char.upper())
    if not matrix:
        return
    
    rows = len(matrix)       
    cols = len(matrix[0])    
    
    start_row = (shape[0] - (rows * scale)) // 2 
    start_col = (shape[1] - (cols * scale)) // 2 
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1':
                # Paint the entire scaled block first...
                for sr in range(scale):
                    for sc in range(scale):
                        pr = start_row + (r * scale) + sr
                        pc = start_col + (c * scale) + sc
                        arr[pr, pc] += color_val
                # ...THEN yield. This cuts yields down from 560 to 35 per letter!
                yield

delay = 0.001

try:
    for letter in text_to_draw:
        arr.fill(0)
        im.set_data(arr)
        plt.pause(0.1)
        
        task = paint_large_letter(letter, scale=4, color_val=200)
        
        if task is None:
            continue
            
        while True:
            try:
                # Batch processing: pull multiple steps from the generator 
                # before we force Matplotlib to redraw the screen
                for _ in range(speed):
                    next(task)
                
                im.set_data(arr % 256)
                plt.pause(delay)
            except StopIteration:
                # Ensure the final state is drawn
                im.set_data(arr % 256)
                plt.pause(delay)
                break 
                
        plt.pause(0.5)

except KeyboardInterrupt:
    print("Animation stopped.")

plt.ioff()
plt.close('all')
