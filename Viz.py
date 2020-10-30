import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('benchmark.csv',  delimiter=',', quotechar='"', skipinitialspace=True)

data['Perc'] = data['Perc'] * 100

print(data.describe())

# plt.hist(data['Perc'], bins=10)

sb.distplot(data['Pred'], color='b', kde_kws={"color": "b", "lw": 3, "label": "Predicted"},)
sb.distplot(data['Actual'], color='g', kde_kws={"color": "g", "lw": 3, "label": "Actual"},)

plt.show()
