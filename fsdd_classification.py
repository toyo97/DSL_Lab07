import os
from scipy.io import wavfile
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from utils.dump_util import dump_to_file
from utils.audio_preprocessing import transform

# Directories for input data
data_path = os.path.abspath('') + '/data'
dev_data_path = data_path + '/dev'

data = []
y = []

# Read the files with scipy's function and parse the labels at the same time
for file_name in os.listdir(dev_data_path):
    data.append(wavfile.read(os.path.join(dev_data_path, file_name))[1])
    y.append(file_name.split('.')[0].split('_')[1])

# Take roughly 1 second for each recording (enough for saying the number) and extract an array from the dataset
T = 2**13
fs = 8000
num_ceps = 13

# The transform function do the following steps:
#   - zero-pad every sample shorter than the longest one
#   - shrink the values in [-1.,1.] values (each sample separately from the other!)
#   - cut every sample to length T (~1 sec)
#   - extract GFCC features (reference: https://opensource.com/article/19/9/audio-processing-machine-learning-python)

X = transform(data, T=T, fs=fs, num_ceps=num_ceps)

# Grid search for SVD and Random Forest or K-Neighbors classifier

svd = TruncatedSVD()

# param_grid = {
#     'truncatedsvd__n_components': [100, 200, 500],
#     'randomforestclassifier__n_estimators': [10, 50, 100],
#     'randomforestclassifier__max_features': [20, 50, 100],
#     'randomforestclassifier__criterion': ['gini', 'entropy']
# }
# rf = RandomForestClassifier()
# pl = make_pipeline(svd, rf)

# K-Neighbors seems to perform better
param_grid = {
    'truncatedsvd__n_components': [100, 200, 500],
    'kneighborsclassifier__n_neighbors': [3, 5, 11],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski']
}
knn = neighbors.KNeighborsClassifier()
pl = make_pipeline(svd, knn)
# Concatenate SVD transformation and classifier fitting in a pipeline

# Create a grid search object to evaluate every parameters configuration
grid_search = GridSearchCV(pl, param_grid, scoring='accuracy', cv=5, verbose=1)
gs_res = grid_search.fit(X, y)

# After grid search, take the best estimator already configured with the best parameters
best_clf = gs_res.best_estimator_
print(f'Grid search result: {gs_res.best_score_}')
print(f'Parameters: {gs_res.best_params_}')

# Fit the best estimator with X, y which is the train labeled dataset
best_clf.fit(X, y)

# ********* PREDICTION ***********
#
# Extract the audio data and indices as before (no labels here)
eval_data_path = data_path + '/eval'
indices = []
test_data = []

for file_name in os.listdir(eval_data_path):
    indices.append(file_name.split('.')[0])
    test_data.append(wavfile.read(os.path.join(eval_data_path, file_name))[1])

# Transform the test set in the same way as the train set
X_test = transform(test_data)

# Predict the test labels on the previously trained estimator
y_hat = best_clf.predict(X_test)

# Save the predictions in a csv file as output (see function `dump_to_file()`)
output_dir = '/home/toyo/PycharmProjects/lab07/output/'
output_filename = os.path.join(output_dir, 'rf_svd_output.csv')

dump_to_file(indices, y_hat, output_filename)
