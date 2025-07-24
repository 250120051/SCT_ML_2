import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('data/customers.csv')

# Selecting features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('outputs/elbow_plot.png')
plt.close()

# Apply KMeans with K=5 (based on elbow)
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to data
data['Cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8, 6))
for cluster in range(5):
    plt.scatter(X_scaled[clusters == cluster, 0], 
                X_scaled[clusters == cluster, 1], 
                label=f'Cluster {cluster}')
    
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.savefig('outputs/cluster_plot.png')
plt.show()

# Save clustered data
data.to_csv('outputs/clustered_customers.csv', index=False)

print("Clustering complete. Results saved to outputs folder.")

