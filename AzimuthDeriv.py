import tkinter as tk
from tkinter import simpledialog
import time
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stroke Derivative Viewer")

        self.show_smooth_stroke = False  # Toggle for showing original stroke overlaid

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, width=150)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.clear_button = tk.Button(self.left_frame, text="Clear Canvas", command=self.clear_canvas, height=2, width=20)
        self.clear_button.pack(side=tk.TOP, pady=5)

        self.save_button = tk.Button(self.left_frame, text="Save Derivative", command=self.save_derivative, height=2, width=20)
        self.save_button.pack(side=tk.TOP, pady=5)

        self.smooth_button = tk.Button(self.left_frame, text="Smooth Derivative", command=self.smooth_derivative, height=2, width=20)
        self.smooth_button.pack(side=tk.TOP, pady=5)

        self.toggle_smooth_stroke_button = tk.Button(self.left_frame, text="Toggle Smooth Stroke", command=self.toggle_smooth_stroke, height=2, width=20, relief=tk.RAISED)
        self.toggle_smooth_stroke_button.pack(side=tk.TOP, pady=5)

        self.fourier_button = tk.Button(self.left_frame, text="Show Fourier Transform", command=self.show_fourier, height=2, width=20)
        self.fourier_button.pack(side=tk.TOP, pady=5)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(self.right_frame, bd=2, relief="solid")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = tk.Frame(self.right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.stroke = []
        self.derivative_data = None

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        self.colorbar = None
        

    def toggle_smooth_stroke(self):
        self.show_smooth_stroke = not self.show_smooth_stroke
        if self.show_smooth_stroke:
            self.toggle_smooth_stroke_button.config(relief=tk.SUNKEN)
        else:
            self.toggle_smooth_stroke_button.config(relief=tk.RAISED)
        print("Showing smoothed stroke:" if self.show_smooth_stroke else "Showing raw stroke")

    def start_draw(self, event):
        self.stroke = [(event.x, event.y, time.time())]
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=2)
        self.stroke.append((event.x, event.y, time.time()))
        self.last_x, self.last_y = event.x, event.y

    def end_draw(self, event):
        self.stroke.append((event.x, event.y, time.time()))
        self.process_stroke()

    def simplify_path(self, stroke, distance_threshold=5):
        simplified_stroke = [stroke[0]]
        for i in range(1, len(stroke)):
            prev_x, prev_y, _ = simplified_stroke[-1]
            curr_x, curr_y, curr_time = stroke[i]
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            if distance >= distance_threshold:
                simplified_stroke.append((curr_x, curr_y, curr_time))
        return simplified_stroke

    def smooth_stroke(self, stroke, window_size=5):
        x = np.array([p[0] for p in stroke])
        y = np.array([p[1] for p in stroke])
        x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        t_smooth = np.linspace(0, max([p[2] for p in stroke]), len(x_smooth))
        return x_smooth, y_smooth, t_smooth

    def compute_speed(self, x, y, t):
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-5)
        return speed

    def process_stroke(self):
        if len(self.stroke) < 5:
            print("Too few points to process.")
            return

        simplified_stroke = self.simplify_path(self.stroke, distance_threshold=5)
        x, y, t = [np.array([p[i] for p in simplified_stroke]) for i in range(3)]

        try:
            tck, u = splprep([x, y], s=20, k=3)
            unew = np.linspace(0, 1, 1000)
            out = splev(unew, tck)
            deriv = splev(unew, tck, der=1)
            speed = self.compute_speed(out[0], out[1], unew)

            self.axs[0].cla()
            self.axs[1].cla()

            norm = mcolors.Normalize(vmin=np.min(speed), vmax=np.max(speed))
            cmap = plt.colormaps['RdYlGn']
            points = np.array([out[0], out[1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(speed)
            lc.set_linewidth(2)
            self.axs[0].add_collection(lc)
            self.axs[0].set_title("Stroke (Speed Heatmap)")
            self.axs[0].invert_yaxis()
            self.axs[0].axis("equal")
            if self.colorbar:
                self.colorbar.ax.remove()
                self.colorbar = None
            self.colorbar = self.fig.colorbar(lc, ax=self.axs[0], label="Speed (px/s)")

            if self.show_smooth_stroke:
                orig_x = np.array([p[0] for p in self.stroke])
                orig_y = np.array([p[1] for p in self.stroke])
                self.axs[0].plot(orig_x, orig_y, color='blue', alpha=0.25, linewidth=1, label='Original Stroke')

            self.derivative_data = deriv
            self.axs[1].plot(deriv[0], deriv[1], label='dx/dt vs dy/dt', color='red')
            self.axs[1].set_title("Derivative")
            self.axs[1].invert_yaxis()
            self.axs[1].axis("equal")
            self.axs[1].legend()

            self.plot_canvas.draw()

        except Exception as e:
            print("Error processing stroke:", e)

    def smooth_derivative(self):
        if self.derivative_data is None:
            print("No derivative data to smooth.")
            return

        smoothed_derivative = gaussian_filter1d(self.derivative_data[1], sigma=3)
        self.axs[1].clear()
        self.axs[1].plot(self.derivative_data[0], smoothed_derivative, label='Smoothed dx/dt vs dy/dt', color='green')
        self.axs[1].set_title("Smoothed Derivative")
        self.axs[1].invert_yaxis()
        self.axs[1].axis("equal")
        self.axs[1].legend()
        self.plot_canvas.draw()

    def save_derivative(self):
        filename = simpledialog.askstring("Save As", "Enter filename for derivative plot (without extension):")
        if filename:
            full_filename = f"{filename}.png"
            self.axs[1].get_figure().savefig(full_filename, bbox_inches='tight', transparent=True)
            print(f"Saved derivative plot to {full_filename}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.stroke = []
        self.derivative_data = None
        self.axs[0].clear()
        self.axs[1].clear()
        self.plot_canvas.draw()

    def show_fourier(self):
        if len(self.stroke) < 5:
            print("Too few points for Fourier transform.")
            return

        x = np.array([p[0] for p in self.stroke])
        y = np.array([p[1] for p in self.stroke])
        x_fft = np.fft.fft(x - np.mean(x))
        y_fft = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(len(x))

        window = tk.Toplevel(self.root)
        window.title("Fourier Transform")
        fig, ax = plt.subplots(2, 1, figsize=(6, 4))
        ax[0].plot(freqs, np.abs(x_fft), label='|X(f)|', color='blue')
        ax[0].set_title("Fourier Transform of X")
        ax[1].plot(freqs, np.abs(y_fft), label='|Y(f)|', color='orange')
        ax[1].set_title("Fourier Transform of Y")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
