import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plots = []
hlines = []
line_colors = []

colors = mpf.make_marketcolors(up="green",down="red", wick='inherit', edge='inherit', volume='in')
mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=colors)

def add_marker(data, marker_size, symbol, color, panel=0):
    plots.append(mpf.make_addplot(data, type='scatter', panel=panel, markersize=marker_size, marker=symbol, color=color))

def add_plot(data, color, panel=0, ylabel="", secondary_y=False):
    plots.append(mpf.make_addplot(data, color=color, panel=panel, ylabel=ylabel, secondary_y=secondary_y))
    return data

def add_p_to_p_lines(data, max_length, color, panel=0, secondary_y=False):
    line = [np.nan] * max_length
    for i in data:
        p1 = i[0]
        p2 = i[1]
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1]
        for j in range(p1[0], p2[0] + 1):
            line[j] = (j - p1[0]) * m + b
    plot = pd.Series(line)
    add_plot(plot, color=color, panel=panel, secondary_y=secondary_y)

def add_hlines(data, color):
    global hlines, line_colors
    hlines = hlines + data
    line_colors = line_colors + [color] * len(data)

def plot_data(stock, data, type, volume=False, graph_plots=True):
    if graph_plots: mpf.plot(data, type=type, style=mpf_style, title=stock, addplot=plots, hlines=dict(hlines=hlines, colors=line_colors), volume=volume)
    else: mpf.plot(data, type=type, style=mpf_style, title=stock, hlines=dict(hlines=hlines, colors=line_colors), volume=volume)

def plot_bar_graph(data, data_x, data_y):
    plt.bar(range(len(data)), data_x, tick_label=data_y)
    plt.show()
