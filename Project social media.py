"""DATA PRE-PROCESSING"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

a = "dummy_data.csv"
data = pd.read_csv(a)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

data = data.dropna()

categorical_cols = ['gender', 'platform', 'interests', 'location', 'demographics', 'profession']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(['income'], axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns.drop('income'))
#print(scaled_data)

features = scaled_data

x = []
K = range(1, 11)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(features)
    x.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, x, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)
scaled_data['Cluster'] = clusters
data['Cluster'] = clusters
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)
sns.pairplot(data, hue='Cluster', palette='viridis')
plt.show()


