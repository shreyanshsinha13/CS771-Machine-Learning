import numpy as np
import pickle as pkl
from datetime import datetime

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials

def feats( df ):
	date_strings = df['Time']
	extract_datetime = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
	hour = lambda x: extract_datetime(x).hour
	minute = lambda x: extract_datetime(x).minute
	X=df[['Time','temp','humidity','no2op1','no2op2','o3op1','o3op2']]
	# y = df[['OZONE','NO2']].to_numpy()

	
	
	X['minute']=date_strings.apply(minute)	
	X['hour']=date_strings.apply(hour) + X['minute']/60
	
	# apply cosine and sine transformation to the hour feature
	X['hour_sin'] = np.sin(X.hour*(2.*np.pi/24))
	X['hour_cos'] = np.cos(X.hour*(2.*np.pi/24))
	# # apply cosine and sine transformation to the minute feature
	# X['minute_sin'] = np.sin(X.minute*(2.*np.pi/60))
	# X['minute_cos'] = np.cos(X.minute*(2.*np.pi/60))
	X.drop(['Time'], axis=1, inplace=True)
	X.drop(['minute'], axis=1, inplace=True)
	X.drop(['hour'], axis=1, inplace=True)

	return X

def my_predict( df ):
	
	# Load your model file
	with open( "final_model2", "rb" ) as file:
		model = pkl.load( file )

	X = feats(df)
	# Make two sets of predictions, one for O3 and another for NO2
	preds = model.predict( X )
	pred_o3, pred_no2 = preds[:,0], preds[:,1]

	# Return both sets of predictions
	return ( pred_o3, pred_no2 )