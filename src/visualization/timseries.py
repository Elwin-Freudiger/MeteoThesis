import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/clean/station_precipitation.csv")
    df = df[df['station']=="TAE"]
    
    plt.plot(df[['precip']])
    plt.savefig("report/figures/timeserie.png")


if __name__ == "__main__":
    main()