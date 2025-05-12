# external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(filename, gens, runs, maxg=1000):
    data = pd.read_csv(f"./outputs/{filename}.csv")

    columns = data.columns[2:]

    avg = {col: [] for col in columns}
    std = {col: [] for col in columns}

    for gen in range(gens):
        df = data.iloc[[i * gens + gen for i in range(runs)]]
        for col in columns:
            avg[col].append(df[col].mean())
            std[col].append(df[col].std())

    fig, (ax1, ax2) = plt.subplots(2)
    colors = ["red", "green", "blue", "orange"]
    for i, col in enumerate(columns[:4]):
        ax1.plot(avg[col][:maxg], color=colors[i], label=col)
        ax1.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + np.array(std[col][:maxg]),
            alpha=0.3,
            color=colors[i],
        )
        ax1.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - 2 * np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + 2 * np.array(std[col][:maxg]),
            alpha=0.1,
            color=colors[i],
        )
    ax1.set_ylabel("Frequency of Strategy")
    ax1.legend(loc="lower right")

    colors = ["green", "red"]
    for i, col in enumerate(columns[4:]):
        ax2.plot(avg[col][:maxg], color=colors[i], label=col)
        ax2.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + np.array(std[col][:maxg]),
            alpha=0.3,
            color=colors[i],
        )
        ax2.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - 2 * np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + 2 * np.array(std[col][:maxg]),
            alpha=0.1,
            color=colors[i],
        )
    ax2.set_xlabel("Generations")
    ax2.set_ylabel("Frequency of Strategy")
    ax2.legend(loc="lower right")

    plt.show()
