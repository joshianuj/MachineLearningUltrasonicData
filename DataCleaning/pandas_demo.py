import pandas as pd
from sklearn.model_selection import train_test_split
file_1 = pd.read_csv("./DataCleaning/NewData/Stopper/1st/C1barrier_100cm_AS_00001.dat", header=None).T
file_2 = pd.read_csv("./DataCleaning/NewData/Stopper/1st/C1barrier_100cm_AS_00002.dat", header=None).T
file_3 = pd.read_csv("./DataCleaning/NewData/Stopper/1st/C1barrier_100cm_AS_00003.dat", header=None).T
file_4 = pd.read_csv("./DataCleaning/NewData/Stopper/1st/C1barrier_100cm_AS_00004.dat", header=None).T

print("data-frame shape: ", file_1.shape, file_2.shape )
overall_file = pd.concat([file_1, file_2, file_3, file_4])
print(overall_file)
train, test = train_test_split(overall_file, test_size=0.2)
print(train, test)
file_3.to_csv('./demo.csv', index=False)
