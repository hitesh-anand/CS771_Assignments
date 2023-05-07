import numpy as np
import pickle

# Define your prediction method here
# df is a dfframe containing timestamps, weather df and potentials


def my_predict(df):
	month = []
	date = []
	minute = []
	hour = []
	for i in range(len(df)):
		month.append(int(df.loc[i,"Time"][5:7]))
		date.append(int(df.loc[i,"Time"][8:10]))
		hour.append(int(df.loc[i,"Time"][11:13]))
		minute.append(int(df.loc[i,"Time"][14:16]))
	df["Month"] = np.array(month)
	df["Hour"] = np.array(hour)
	df["Date"] = np.array(date)
	df["Minute"] = np.array(minute)
	df = df.drop(columns = "Time")
	X = np.array(df)
	with open( "ozone_model", "rb" ) as file:
		model = pickle.load( file )
	pred_o3 = model.predict(X)
	with open( "nitrogen_model", "rb" ) as file:
		model = pickle.load( file )
	pred_no2 = model.predict(X)
	return (pred_o3, pred_no2)



