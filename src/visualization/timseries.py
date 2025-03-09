import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/filtered/pression_filter.csv")
    df = df[df['station']=="TAE"]
    
    plt.plot(df[['pression']])
    plt.show()
    plt.savefig("report/figures/pression_ts.png")


if __name__ == "__main__":
    main()