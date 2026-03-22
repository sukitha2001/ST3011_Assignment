import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. Load the data and the trained model
print("Loading data and models...")
WEATHER = pd.read_csv('data/Cleaned_SriLanka_Weather_Dataset.csv')
scaler = joblib.load('models/kmeans_scaler.pkl')
kmeans = joblib.load('models/kmeans_default.pkl')

# 2. Prepare the data
features = ['temperature_2m_max', 'windspeed_10m_max']
X_scaled = scaler.transform(WEATHER[features])

# 3. Predict clusters
WEATHER['cluster'] = kmeans.predict(X_scaled)

# 4. Create the Visualization
plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")

# Create the scatter plot
scatter = sns.scatterplot(
    data=WEATHER, 
    x='temperature_2m_max', 
    y='windspeed_10m_max', 
    hue='cluster', 
    palette='viridis', 
    alpha=0.6,
    s=60
)

# Plot the Cluster Centroids
# Note: We must inverse transform the centroids to show them on the original temperature/wind scale
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

plt.scatter(
    centroids[:, 0], centroids[:, 1], 
    c='red', s=200, marker='X', label='Centroids',
    edgecolor='black', linewidth=2
)

# Formatting
plt.title(f'K-Means Climatic Clustering (K={kmeans.n_clusters})', fontsize=16)
plt.xlabel('Maximum Temperature (°C)', fontsize=12)
plt.ylabel('Maximum Wind Speed (km/h)', fontsize=12)
plt.legend(title='Climatic Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('models/kmeans_visualization.png')
print("✅ Visualization saved to models/kmeans_visualization.png")
plt.show()