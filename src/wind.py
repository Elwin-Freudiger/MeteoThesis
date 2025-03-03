##file to transform wind to North East Value

#import libraries
import pandas as pd
import numpy as np



def main():
    #load files
    speed_df = pd.read_csv("data/processed/wind_speed.csv")
    direction_df = pd.read_csv("data/processed/wind_direction.csv")

    #aggregate
    df = pd.merge(speed_df, direction_df, how="outer", on=["station", "time"]).dropna()
    
    df['direction_rad'] = np.deg2rad(df['wind_direction'])

    # Calculate North and East components
    df['North'] = df['wind_speed'] * np.cos(df['direction_rad']).round(1)
    df['East'] = df['wind_speed'] * np.sin(df['direction_rad']).round(1)

    # Drop the temporary radian column
    df.drop(columns=['direction_rad', 'wind_speed', 'wind_direction'], inplace=True)

    # Save the transformed data
    df.to_csv("data/processed/wind_vectors.csv", index=False)   
    print("finished!")


if __name__ == "__main__":
    main()