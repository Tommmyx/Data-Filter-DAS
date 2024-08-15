"""
===============================================================================
            SUBSTATION METER CHANGE DETECTION AND CLASSIFICATION
===============================================================================
Author: [Le Cam]
Date: [August 2024]

Description:
------------
This script is designed to analyze and classify changes in substation meters 
using various machine learning models. The workflow involves data preprocessing, 
feature engineering using sliding windows, and model selection through grid 
search. The script compares multiple classifiers to determine the best-performing 
model for predicting meter changes.

Dependencies:
-------------
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

Installation:
-------------
Make sure you have Python 3.x installed, along with the required libraries.
You can install the necessary packages using pip:

    pip install pandas numpy matplotlib scikit-learn scipy

Execution:
----------
1. Place the data file in the same directory as this script.
2. Run the script from the command line or an IDE :

    python meterChanged.py 2014.txt

3. The script will load the data, preprocess it, and run multiple classifiers.
4. The best-performing models will be selected using GridSearchCV, and the results 
   will be displayed and plotted.
   
Notes:
------
- Ensure the data format in the file matches the expected structure.
- You may need to modify the script if your data has different characteristics.
- The output includes model performance metrics like MAE, MAPE, and R², 
  along with corresponding bar plots.

===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import zscore
import argparse


# Set up argument parser
parser = argparse.ArgumentParser(description="Substation Meter Change Detection and Classification")
parser.add_argument('file_path', type=str, help='Path to the data file')
args = parser.parse_args()

# Load the data
file_path = args.file_path
data = pd.read_csv(file_path, delimiter='\t', header=None)
data.columns = ['meter_id', 'substation_id', 'timestamp', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7', 'value8', 'value9', 'changed_meter']
data['timestamp'] = pd.to_datetime(data['timestamp'])

columns_to_convert = ['value4', 'value5', 'value6', 'value7', 'value8', 'value9']
for col in columns_to_convert:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# ADDED --------------------------------------------- 
def correlation_analysis(data, threshold=0.8):
    corr_matrix = data.corr().abs()
    print("Correlation Matrix:\n", corr_matrix)
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return to_drop

features = ['value4', 'value5', 'value6', 'value7', 'value8', 'value9']
data_features = data[features]
to_drop = correlation_analysis(data_features)
data.drop(to_drop, axis='columns')
print(f"Features to drop (correlation > 0.9): {to_drop}")

for feature in to_drop:
    columns_to_convert.remove(feature)
print(columns_to_convert)
#ADDED ---------------------------------------------



def find_timestamps_change_meter(substation_id):
    substation = data[data['substation_id'] == substation_id]
    return substation[substation['changed_meter'] == 1]['timestamp'].to_list()

unique_substation_ids = data['substation_id'].unique()

def max_changed_month():
    list_changed = []
    list_unchanged = []
    
    for substation_id in unique_substation_ids:
        timestamp_changed_meter = find_timestamps_change_meter(substation_id)
        if timestamp_changed_meter:
            for timestamp in timestamp_changed_meter:
                list_changed.append({'meter_id': substation_id, 'month_of_change': timestamp.strftime('%B')})
        else:
            list_unchanged.append({'meter_id': substation_id})
    
    list_changed_df = pd.DataFrame(list_changed)
    list_unchanged_df = pd.DataFrame(list_unchanged)
    
    if not list_changed_df.empty:
        month_counts = list_changed_df['month_of_change'].value_counts()
        max_month = month_counts.idxmax()
        max_count = month_counts.max()
        print(f"Month with the most changes: {max_month} with {max_count} changes")

        # Histogram
        plt.figure(figsize=(5, 3))
        month_counts.plot(kind='bar')
        plt.xlabel('Month')
        plt.ylabel('Number of Meter Changes')
        plt.title('Number of Meter Changes per Month')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        list_changed_max_month = list_changed_df['meter_id'].tolist()
    else:
        max_month = None
        list_changed_max_month = []

    return max_month, list_changed_max_month, list_unchanged_df['meter_id'].tolist()

max_month, changed_substation_ids, unchanged_substation_ids = max_changed_month()

num_changed = len(changed_substation_ids)
num_unchanged = num_changed
balanced_unchanged_substation_ids = np.random.choice(unchanged_substation_ids, num_unchanged, replace=False)

balanced_substation_ids = np.concatenate([changed_substation_ids, balanced_unchanged_substation_ids])
class_labels = np.concatenate([np.ones(num_changed), np.zeros(num_unchanged)])


final_data = pd.DataFrame({
    'substation_id': balanced_substation_ids,
    'target': class_labels
})


def pad_window(window_data, window_size, padding_value=0):
    if len(window_data) < window_size:
        padding_needed = window_size - len(window_data)
        padding = pd.DataFrame(padding_value, index=range(padding_needed), columns=window_data.columns)
        window_data = pd.concat([window_data, padding], ignore_index=True)
    return window_data

def quantize_value(x, mean, std):
    if x >= mean + 3 * std:
        return 3
    elif x >= mean + 2 * std:
        return 2
    elif x >= mean + std:
        return 1
    elif x > mean - std:
        return 0
    elif x > mean - 2 * std:
        return -1
    elif x > mean - 3 * std:
        return -2
    else:
        return -3

def create_sliding_window_features(df, columns, window_size):
    sliding_features = []
    num_rows = len(df)
    
    for start in range(num_rows - window_size + 1):
        window = df.iloc[start:start + window_size]
        window = pad_window(window, window_size)
        means = window[columns].mean()
        stds = window[columns].std()
        quantized_window = []
        for col in columns:
            quantized_col = window[col].apply(lambda x: quantize_value(x, means[col], stds[col]))
            sliding_features.append(quantized_col.values)
    return np.array(sliding_features)


def summarize_window(features, method='mean'):
    if method == 'mean':
        return np.mean(features, axis=0)
    elif method == 'median':
        return np.median(features, axis=0)
    elif method == 'sum':
        return np.sum(features, axis=0)
    else:
        raise ValueError("Unknown method. Use 'mean', 'median', or 'sum'.")
        
def apply_sliding_for_all_dataset(data, columns, window_size, summary_method='mean', padding_value=0):
    all_features = []
    num_rows = len(data)
    
    for start in range(0, num_rows, window_size):
        end = start + window_size
        if end > num_rows:
            window_data = data.iloc[start:]
            window_data = pad_window(window_data, window_size, padding_value)
        else:
            window_data = data.iloc[start:end]
        features_sliding = create_sliding_window_features(window_data, columns, window_size)
        summarized_features = []
        for row in features_sliding:
            summarized_feature = summarize_window(row, method=summary_method)
            summarized_features.append(summarized_feature)
        all_features.append(summarized_features)
        all_features_concat = np.concatenate(all_features)
    return np.array(all_features_concat)

# Remove outliers/extreme datas with zscore
def replace_outliers_with_zscore(data, columns, threshold=3):
    data_copy = data.copy()
    for column in columns:
        if column in data_copy:
            z_scores = zscore(data_copy[column])
            outliers = np.abs(z_scores) > threshold
            data_copy.loc[outliers, column] = np.sign(z_scores[outliers]) * threshold
    return data_copy

def apply_rolling_window_to_substations(final_data, window_size, columns, summary_method='mean', padding_value=0):
    all_features_slide = []
    all_features_cumulative = []
    meter_ids = []
    
    for substation_id in final_data['substation_id']:
        substation_data = data[data['substation_id'] == substation_id]

        change_timestamps = find_timestamps_change_meter(substation_id)
   
        #TODO what features are important : the month which the meter was changed 

        substation_data = substation_data.sort_values('timestamp')
        
        # Filter datas 
        filtered_substation_data = replace_outliers_with_zscore(substation_data, columns)
        features_sliding = apply_sliding_for_all_dataset(filtered_substation_data, columns, window_size, summary_method, padding_value)
        
        all_features_slide.append(features_sliding)
        meter_ids.append(substation_data['meter_id'].iloc[0])
    
    return all_features_slide, all_features_cumulative, meter_ids

window_size = 24
summary_method = 'mean'  
padding_value = 0

##################################
# TMP JUST FOR TESTING ON VALUE 8  
columns_to_convert = ['value8']
#################################

slide_window_features, cumulative_window_features, meter_ids = apply_rolling_window_to_substations(final_data, window_size, columns_to_convert, summary_method, padding_value)

final_means_df = pd.DataFrame(slide_window_features)
final_means_df.insert(0, 'meter_id', final_data['substation_id'].values)
final_means_df['target'] = final_data['target'].values

print(final_means_df)

# Reduce nb of features PCA 

# Fill NaN values with 0 or another appropriate value
final_means_df = final_means_df.fillna(0)

# Diagnostics: Check for any inconsistencies
print(f"Shape of final_means_df: {final_means_df.shape}")
print(f"Number of NaN values in final_means_df: {final_means_df.isna().sum().sum()}")
print(final_means_df)

# Classifiers 
X = final_means_df.drop(columns=['meter_id', 'target'])
y = final_means_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Define parameter grids for each model
param_grids = {
    'gb': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'mlp': {'hidden_layer_sizes': [(50,), (50, 50)], 'max_iter': [300, 500]},
    'svm': {'svc__C': [0.1, 1, 10], 'svc__gamma': [0.01, 0.1, 1]},
    'rf': {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2']}
}

# Initialize models
models = {
    'gb': GradientBoostingClassifier(random_state=42),
    'mlp': MLPClassifier(random_state=42),
    'svm': make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42)),
    'rf': RandomForestClassifier(random_state=42)
}

# Perform Grid Search for each model
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

# Evaluate each optimized model
results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'MAPE': mape, 'R²': r2})

results_df = pd.DataFrame(results)
print(results_df)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(results_df['Model'], results_df['MAE'], color='b', alpha=0.6)
plt.ylabel('MAE')
plt.title('MAE Comparison')

plt.subplot(1, 2, 2)
plt.bar(results_df['Model'], results_df['MAPE'], color='g', alpha=0.6)
plt.ylabel('MAPE')
plt.title('MAPE Comparison')

plt.tight_layout()
plt.show()
