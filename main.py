import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

corr = data.corr()

plt.figure()
sns.heatmap(corr, annot=True)

plt.savefig('result/corr_heatmap.png')
plt.show()
