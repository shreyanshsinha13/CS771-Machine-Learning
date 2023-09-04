import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
import sklearn

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def create_feats(Z):
    R = 64
    S = 4
    n = 2**S
    X = np.zeros((Z.shape[0], (R+1)*n + 1))
    for l in range(Z.shape[0]):
        p = Z[l,R:R+S]
        q = Z[l,R+S:R+2*S]
        i = int(''.join(str(int(x)) for x in p), 2)
        j = int(''.join(str(int(x)) for x in q), 2)       
        X[l, i*(R+1):(i+1)*(R+1)] = np.concatenate((np.ones(1), Z[l,0:R]) )
        X[l, j*(R+1):(j+1)*(R+1)] = np.concatenate((-np.ones(1), -Z[l,0:R]) )
        X[l, -1] = Z[l,-1]
    return X

    

def fit(Z_trn, model, loss):
    new_feats = create_feats(Z_trn)
    X = new_feats[:, :-1]
    y = new_feats[:, -1]
    
    if model == 'logistic':
        model = LogisticRegression(C = 10, penalty = 'l2', max_iter = 1000)
    else:
        model = LinearSVC(C = 10, loss= loss)
    model.fit(X, y)

    return model



################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################
    loss = 'squared_hinge'

    method = ['linear', 'logistic']
    model = fit(Z_train, method[0], loss)
	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	
    return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
    new_feats = create_feats(X_tst)
    X = new_feats[:, :-1]
    pred = model.predict(X)

	# Use this method to make predictions on test challenges	
    return pred
