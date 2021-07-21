import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns


# IMPORT DATASET
df = pd.read_csv('/Users/dhruv/code/export_yardi_jourentline.csv', lineterminator='\n')
df.columns = df.columns.map(lambda c: c.strip())

fullColumns = ['GLCODE', 'GLNAME', 'PROPERTY', 'PROPERTYNAME', 'UNIT', 'DATE', 'PERIOD', 'CONTROL']

colTests = 


def runModel(cols):

    df = df[cols]