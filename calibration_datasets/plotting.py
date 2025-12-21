#Plots of the merged_data
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    merged_data = pd.read_csv("merged_data.csv")
    merged_data.plot(x="Consumption_Factor_Normalizied", y="month_diff", kind="scatter")

    res = stats.linregress(merged_data["Consumption_Factor_Normalizied"], merged_data["month_diff"])

    plt.plot(merged_data["Consumption_Factor_Normalizied"],
             res.slope * merged_data["Consumption_Factor_Normalizied"],
             'r',
             label='fitted line')

    print(f"The slope is {res.slope}")

    merged_data["error"] = merged_data["month_diff"] - (res.slope * merged_data["Consumption_Factor_Normalizied"])
    merged_data["abs_error"] = merged_data["error"].abs()
    merged_data["abs_error_percent"] = merged_data["abs_error"] / merged_data["month_diff"] * 100

    print(f"Average Error is: {merged_data["abs_error_percent"].mean()}")
    merged_data.to_csv("merged_data_analyzed.csv")

    total_error = merged_data["abs_error"].sum()
    total_consumption = merged_data["month_diff"].abs().sum()

    print(f"Total Error is: {total_error / total_consumption}")

    plt.show()
