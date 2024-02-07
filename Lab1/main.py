import tkinter as tk
from animator import Animator


root = tk.Tk()
root.title("Lab 1")

button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, anchor='w')

canvas = tk.Canvas(root, bg='white')
canvas.pack(fill=tk.BOTH, expand=True)


animator = Animator(canvas)

move_left_button = tk.Button(button_frame, text="Move Left",
                             command=lambda: animator.move_left())
move_left_button.pack(side=tk.TOP, expand=True)

move_right_button = tk.Button(button_frame, text="Move Right",
                             command=lambda: animator.move_right())
move_right_button.pack(side=tk.TOP, expand=True)

move_up_button = tk.Button(button_frame, text="Move Up",
                             command=lambda: animator.move_up())
move_up_button.pack(side=tk.TOP, expand=True)

move_down_button = tk.Button(button_frame, text="Move Down",
                             command=lambda: animator.move_down())
move_down_button.pack(side=tk.TOP, expand=True)

zoom_in_button = tk.Button(button_frame, text="Zoom In",
                             command=lambda: animator.zoom(1.2))
zoom_in_button.pack(side=tk.TOP, expand=True)

zoom_out_button = tk.Button(button_frame, text="Zoom Out",
                             command=lambda: animator.zoom(0.8))
zoom_out_button.pack(side=tk.TOP, expand=True)

rotate_button = tk.Button(button_frame, text="Rotate",
                          command=lambda: animator.rotate(1, 4, 0))
rotate_button.pack(side=tk.TOP, expand=True)

reset_button = tk.Button(button_frame, text="Reset",
                         command=lambda: animator.reset())
reset_button.pack(side=tk.TOP, expand=True)
root.mainloop()
