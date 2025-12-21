#Plots of the merged_data
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    merged_data = pd.read_csv("merged_data.csv")
    merged_data.plot(x="Consumption_Factor_Normalizied", y="month_diff", kind="scatter")
    plt.show()
