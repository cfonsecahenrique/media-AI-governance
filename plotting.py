# external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

double_std: bool = False


def plot_time_series(filename, parameters, runs, maxg=1000):
    sim_params = parameters[0]
    payoffs = parameters[1]

    data = pd.read_csv(f"./outputs/{filename}.csv")
    gens = parameters[0]["generations"]
    columns = data.columns[1:]
    avg = {col: [] for col in columns}
    std = {col: [] for col in columns}

    for gen in range(gens):
        df = data.iloc[[i * gens + gen for i in range(runs)]]
        for col in columns:
            avg[col].append(df[col].mean())
            std[col].append(df[col].std())

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8))
    colors = ["red", "green", "orange", "blue"]
    for i, col in enumerate(columns[1:5]):
        ax1.plot(avg[col][:maxg], color=colors[i], label=col)
        ax1.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + np.array(std[col][:maxg]),
            alpha=0.2,
            color=colors[i],
        )
        if double_std:
            ax1.fill_between(
                range(len(avg[col][:maxg])),
                np.array(avg[col][:maxg]) - 2 * np.array(std[col][:maxg]),
                np.array(avg[col][:maxg]) + 2 * np.array(std[col][:maxg]),
                alpha=0.1,
                color=colors[i],
            )
    ax1.set_ylabel("Frequency of Strategy")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1)

    colors = ["red", "green"]
    for i, col in enumerate(columns[5:]):
        ax2.plot(avg[col][:maxg], color=colors[i], label=col)
        ax2.fill_between(
            range(len(avg[col][:maxg])),
            np.array(avg[col][:maxg]) - np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + np.array(std[col][:maxg]),
            alpha=0.2,
            color=colors[i],
        )
        if double_std:
            ax2.fill_between(
                range(len(avg[col][:maxg])),
                np.array(avg[col][:maxg]) - 2 * np.array(std[col][:maxg]),
                np.array(avg[col][:maxg]) + 2 * np.array(std[col][:maxg]),
                alpha=0.1,
                color=colors[i],
            )
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Generations")
    ax2.set_ylabel("Frequency of Strategy")
    ax2.legend(loc="lower right")

    col = columns[0]  # 'acr' column
    ax3.plot(range(1, maxg), avg[col][1:maxg], label=col)
    ax3.fill_between(
        range(len(avg[col][:maxg])),
        np.array(avg[col][:maxg]) - np.array(std[col][:maxg]),
        np.array(avg[col][:maxg]) + np.array(std[col][:maxg]),
        alpha=0.2,
    )
    if double_std:
        ax3.fill_between(
            range(1, len(avg[col][1:maxg])),
            np.array(avg[col][:maxg]) - 2 * np.array(std[col][:maxg]),
            np.array(avg[col][:maxg]) + 2 * np.array(std[col][:maxg]),
            alpha=0.1,
        )
    ax3.axvline(
        x=sim_params["convergence period"], linestyle="--", color="gray", linewidth=1
    )
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Generations")
    ax3.set_ylabel("Average Cooperation Ratio")
    ax3.legend(loc="lower right")

    # ==== CAPTION with PARAMETERS ====
    caption_above = (
        f"Z: {sim_params['user population size']}  "
        f"Zc: {sim_params['creator population size']}  "
        f"U(mutations): {100*(sim_params['user mutation probability']/sim_params['user population size'])}%  "
        f"C(mutations): {100*(sim_params['creator mutation probability']/sim_params['creator population size'])}%"
    )
    caption_below = (
        f"media quality (q): {payoffs['media quality']}  "
        f"user benefit (b_U): {payoffs['user benefit']}  "
        f"user cost (c_U): {payoffs['user cost']}  "
        f"investigation cost (c_i): {payoffs['cost investigation']}  "
        f"creator benefit (b_p): {payoffs['creator benefit']}  "
        f"creator cost (c_p): {payoffs['creator cost']}"
    )
    fig.text(0.5, 0.01, caption_below, ha="center", fontsize=9, wrap=True)
    fig.suptitle(caption_above, fontsize=12, y=0.95)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])  # Leave space for caption
    plt.show()


def plot_heatmap(
    filenames, dir, vars, v1_range, v2_range, data_len, precision=101, save_fig=False
):
    translator = {
        "q": "Media Quality",
        "bU": "User Benefit",
        "cU": "User Cost",
        "cI": "Cost of Investigation",
        "bP": "Creator Benefit",
        "cP": "Creator Cost",
    }

    n_bins1 = len(v1_range)
    n_bins2 = len(v2_range)
    v1_range = np.round(v1_range, 2)
    v2_range = np.round(v2_range, 2)
    c_heatmap = np.zeros((n_bins1, n_bins2))

    path = f"./outputs/{dir}"

    for i, filename in enumerate(filenames):
        data = pd.read_csv(f"{path}/{filename}.csv")
        count, _ = np.histogram(data["acr"], bins=precision)
        avg_cooperation_dist = count / data_len
        avg_cooperation = sum(
            [k / precision * avg_cooperation_dist[k] for k in range(precision)]
        )
        c_heatmap[i // n_bins2, i % n_bins2] = avg_cooperation

    plt.title("Average Cooperation Rate")
    plt.imshow(c_heatmap, cmap="RdYlGn")
    plt.xticks(ticks=[i for i in range(n_bins2)], labels=v2_range)
    plt.yticks(ticks=[i for i in range(n_bins1)], labels=reversed(v1_range))
    plt.xlabel(translator[vars[1]])
    plt.ylabel(translator[vars[0]])
    plt.colorbar()
    if save_fig:
        plt.savefig(f"{path}.png")
    plt.show()
