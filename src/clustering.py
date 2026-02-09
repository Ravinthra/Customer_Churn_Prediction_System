import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def customer_segmentation(path):
    df = pd.read_csv(path)

    features = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    features['TotalCharges'] = pd.to_numeric(features['TotalCharges'], errors='coerce')
    features.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)

    return df[['tenure', 'MonthlyCharges', 'Cluster']]
