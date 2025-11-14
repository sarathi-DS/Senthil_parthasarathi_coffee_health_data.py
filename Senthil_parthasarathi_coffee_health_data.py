import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
# Data loading
df = pd.read_csv('senthil_parthasarathi_synthetic_coffee_health_10000.csv')
print(df)

#check the data types
df_datatpyes = df.dtypes
print(df_datatpyes)
print('<-----null data finding----------->')

# find null values
null_finding = df.isna().sum()
print(null_finding)

#Data information
df.info()

#data set statics
print(df.describe())

#unique finds counts
unique_columns = df.nunique()

#find health issue columns value
print('<-------------health issue columns unique values-------->')
unique_health = df['Health_Issues'].unique()
print(unique_health)

# Fill missing values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df[['Health_Issues']] = imputer.fit_transform(df[['Health_Issues']])

# Dictionary of unique values per column
unique_vals = {col: df[col].unique() for col in df.columns}

# Display the results
for col, values in unique_vals.items():
    print(f"{col}: {values}")
print(unique_columns)

# dataframe columns
print(df.columns)

# drop the columns
drop_columns = df.drop(['ID','Occupation'], axis = 1,inplace=True)
print(df.columns)
print(df['Coffee_Intake'])
categorical_data = ['Health_Issues','Stress_Level','Sleep_Quality','Gender']
numerical_data = ['Age','Coffee_Intake','Caffeine_mg','Sleep_Hours','BMI','Heart_Rate','Physical_Activity_Hours','Smoking','Alcohol_Consumption' ]

# preprocess and column transformer
preprocess = ColumnTransformer(
       transformers=[('num', StandardScaler(), numerical_data),
                     ('cat',OneHotEncoder(),categorical_data)

       ])
X = preprocess.fit_transform(df)
# find K value

#Train an test
X = df[numerical_data + categorical_data]
y = df['Alcohol_Consumption']  #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines
knn_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

svc_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('svc', SVC(kernel='rbf', C=1.0))
])

# Fit and evaluate
for name, model in [('KNN', knn_pipeline), ('SVC', svc_pipeline)]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('y_prediction',preds)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"{name} Report:\n{classification_report(y_test, preds)}")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(svc_pipeline, X, y, cv=5)
print("Cross-validated accuracy:", scores.mean())

# K-Means Clustering with Preprocessed Features and PCA Visualization
# for k in range(2, 10):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     score = silhouette_score(X, labels)
#     print(f"Silhouette Score for k={k}: {score:.3f}")

# Transform your data using the fitted preprocessor
X_transformed = preprocess.transform(df)

#  Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_transformed)

# Add cluster labels to your DataFrame
df['Cluster'] = clusters

# Optional — Reduce dimensions for visualization
X_pca = PCA(n_components=3).fit_transform(X_transformed)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['PC1'], df['PC2'], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Cluster Formation')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()
print('---------------------------')
print(df.groupby('Cluster')[numerical_data].mean())
print(df.groupby('Cluster')[categorical_data].agg(lambda x: x.value_counts().index[0]))
#visulaization
df.hist(figsize=(10, 10), bins=10)
plt.suptitle("Histograms for All Columns", fontsize=16)
plt.show()
#compare cofee inatake and sleep_qulity

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='Coffee_Intake', y='Sleep_Hours', hue='Cluster',   palette='viridis')  # Optional: add cluster labels
plt.title('Sleep vs Coffee Intake')
plt.xlabel('Coffee Intake (cups/day)')
plt.ylabel('Sleep Hours')
plt.grid(True)
plt.show()
