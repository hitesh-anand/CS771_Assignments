import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import time as tm

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

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################
  X_train = Z_train[:,0:64]
  y_train = Z_train[:,72]
  mux2 = Z_train[:,68:72]
  mux1 = Z_train[:,64:68]
  upper_machine = mux1[:,0] + mux1[:,1]*2 + mux1[:,2]*4 + mux1[:,3]*8
  lower_machine = mux2[:,0] + mux2[:,1]*2 + mux2[:,2]*4 + mux2[:,3]*8

  X_new = np.zeros((len(X_train), 16*65))

  for i in range(len(X_new)):
    X_new[i, int(upper_machine[i]*64):int(upper_machine[i]*64+64)] = X_train[i,:]
    X_new[i, int(lower_machine[i]*64):int(lower_machine[i]*64+64)] = -X_train[i,:]
    X_new[i, int(-16+upper_machine[i])] = 1
    X_new[i, int(-16+lower_machine[i])] = -1

  model = LogisticRegression(C=1000)
  model.fit(X_new, y_train)
  print("Training accuracy is : ", sum(model.predict(X_new) == y_train)/Z_train.shape[0])
	
  return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
  X_test = X_tst[:,0:64]
  upper_machine = X_tst[:,64:68][:,0] +X_tst[:,64:68][:,1]*2 + X_tst[:,64:68][:,2]*4 + X_tst[:,64:68][:,3]*8
  lower_machine = X_tst[:,-4:][:,0] + X_tst[:,-4:][:,1]*2 + X_tst[:,-4:][:,2]*4 + X_tst[:,-4:][:,3]*8
  X_tst_new = np.zeros((len(X_tst), 16*65))  
  for i in range(X_tst.shape[0]):
    X_tst_new[i, int(upper_machine[i]*64):int(upper_machine[i]*64+64)] = X_test[i,:]
    X_tst_new[i, int(lower_machine[i]*64):int(lower_machine[i]*64+64)] = -X_test[i,:]
    X_tst_new[i, int(-16+upper_machine[i])] = 1
    X_tst_new[i, int(-16+lower_machine[i])] = -1

  pred = model.predict(X_tst_new)
  return pred
