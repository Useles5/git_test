import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# DBSCAN model to predict new charging station locations (without 1 km constraint)
def predict_locations(showrooms, existing_stations, eps_km=0.5, min_samples=2):
    # Step 1: Convert showroom data to DataFrame
    df = pd.DataFrame(showrooms)

    # Step 2: Validate data
    required_cols = ['lat', 'lon', 'sales_car', 'sales_bus', 'sales_bike']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns")
    if not (df['lat'].between(18.8, 19.3)).all() or not (df['lon'].between(72.7, 73.1)).all():
        raise ValueError("Coordinates outside Mumbai bounds")
    if not (df[['sales_car', 'sales_bus', 'sales_bike']] >= 0).all().all():
        raise ValueError("Sales must be non-negative")

    # Step 3: Calculate weighted sales
    weight_car, weight_bus, weight_bike = 1.0, 1.5, 0.5
    df['weighted_sales'] = (df['sales_car'] * weight_car +
                           df['sales_bus'] * weight_bus +
                           df['sales_bike'] * weight_bike)

    # Step 4: Normalize weighted sales to 1-10
    if df['weighted_sales'].max() == df['weighted_sales'].min():
        df['normalized_weighted_sales'] = 1
    else:
        df['normalized_weighted_sales'] = (
            (df['weighted_sales'] - df['weighted_sales'].min()) /
            (df['weighted_sales'].max() - df['weighted_sales'].min()) * 9 + 1
        ).round().astype(int)

    # Step 5: Duplicate coordinates based on normalized weighted sales
    coords = []
    original_indices = []
    for idx, row in df.iterrows():
        for _ in range(row['normalized_weighted_sales']):
            coords.append([radians(row['lat']), radians(row['lon'])])
            original_indices.append(idx)
    coords = np.array(coords)

    # Step 6: Run DBSCAN
    if len(coords) == 0:
        return []
    eps_radians = eps_km / 6371  # Convert km to radians (Earth's radius = 6371 km)
    db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine').fit(coords)
    labels = db.labels_  # -1 for noise, 0, 1, ... for clusters

    # Step 7: Calculate centroids for each cluster
    predicted_locations = []
    for cluster_label in set(labels) - {-1}:  # Exclude noise
        # Get indices of points in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        # Map back to original showrooms
        original_showroom_indices = [original_indices[i] for i in cluster_indices]
        unique_showroom_indices = list(set(original_showroom_indices))
        # Calculate centroid (mean lat, lon of original showrooms)
        cluster_showrooms = df.iloc[unique_showroom_indices]
        centroid_lat = cluster_showrooms['lat'].mean()
        centroid_lon = cluster_showrooms['lon'].mean()

        # Step 8: Ensure within Mumbai bounds (no 1 km adjustment)
        centroid_lat = max(18.8, min(19.3, centroid_lat))
        centroid_lon = max(72.7, min(73.1, centroid_lon))

        # Step 9: Add reasoning
        total_sales_car = int(cluster_showrooms['sales_car'].sum())
        total_sales_bus = int(cluster_showrooms['sales_bus'].sum())
        total_sales_bike = int(cluster_showrooms['sales_bike'].sum())
        total_weighted_sales = int(cluster_showrooms['weighted_sales'].sum())
        reason = (f"High demand: {total_weighted_sales} weighted sales "
                  f"(Cars: {total_sales_car}, Buses: {total_sales_bus}, Bikes: {total_sales_bike})")

        # Step 10: Add to predictions
        predicted_locations.append({
            "lat": centroid_lat,
            "lon": centroid_lon,
            "reason": reason
        })

    return predicted_locations

# Test the model with sample data
def test_dbscan_model():
    # Sample showroom data
    showrooms = [
        {"name": "Showroom A", "lat": 19.0760, "lon": 72.8777, "sales_car": 300, "sales_bus": 100, "sales_bike": 100},
        {"name": "Showroom B", "lat": 19.0660, "lon": 72.8677, "sales_car": 200, "sales_bus": 50, "sales_bike": 50},
        {"name": "Showroom C", "lat": 19.0800, "lon": 72.8800, "sales_car": 150, "sales_bus": 20, "sales_bike": 30},
        {"name": "Showroom D", "lat": 19.2000, "lon": 72.9500, "sales_car": 100, "sales_bus": 10, "sales_bike": 10},
    ]

    # Sample existing stations (not used for adjustment)
    existing_stations = pd.DataFrame([
        {"name": "Station 1", "lat": 19.0500, "lon": 72.8500},
        {"name": "Station 2", "lat": 19.0900, "lon": 72.8900},
    ])

    # Run the model
    try:
        predictions = predict_locations(showrooms, existing_stations, eps_km=0.5, min_samples=2)
        print("Predicted Locations:")
        for pred in predictions:
            print(pred)
    except Exception as e:
        print(f"Error: {e}")

# Run the test
if __name__ == "__main__":
    test_dbscan_model()