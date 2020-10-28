import pandas as pd
import numpy as np
import data_processing as dp

data = pd.read_csv('diabetes.csv')

print(len(data.values))

for person in data.values:
    print(person)
    #person['Outcome']
