# data science
import math
import random
import pandas as pd
import numpy as np
import statsmodels.api as sm
import geopandas as gpd
from shapely import wkt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL 
from statsmodels.tsa.api import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster

# time
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timedelta 
from astral import LocationInfo
from astral.sun import sun
# plotting
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
sns.set_theme(style="white")
# sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RFECV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# system
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
# misc
import json
import pickle
import re

# custom
from helper_functions import Evaluation





