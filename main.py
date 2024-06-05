import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
header = [
    'longitude',
    'latitude',
    'housingMedianAge',
    'totalRooms',
    'totalBedrooms',
    'population',
    'households',
    'medianIncome',
    'medianHouseValue'
]
data = pd.read_csv('CaliforniaHousing/cal_housing.data', header=None, names=header)

# Correlation matrix
corr = data.corr()

# Plotting the heatmap
plt.figure()
sns.heatmap(corr, annot=True)
plt.savefig('result/corr_heatmap.png')
plt.show()
