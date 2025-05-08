import sys
import matplotlib.pyplot as plt
import pandas as pd


def run():
    if sys.argv[1]:
        file_name: str = "outputs/" + str(sys.argv[1])
    else:
        raise ValueError(
            "No filename provided. Please run as 'python plotting.py <filename>'"
        )

    df = pd.read_csv(file_name).drop("generation", axis=1)
    df.plot()
    plt.show()


if __name__ == "__main__":
    run()
