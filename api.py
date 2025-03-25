from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import logging
from math import radians, sin, cos, sqrt, atan2
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Showroom(BaseModel):
    name: str
    lat: float
    lon: float
    sales_car: float
    sales_bus: float
    sales_bike: float

class PredictRequest(BaseModel):
    showrooms: list[Showroom]

existing_stations = pd.DataFrame([
    {"name": "Station 1", "lat": 19.0500, "lon": 72.8500},
    {"name": "Station 2", "lat": 19.0900, "lon": 72.8900},
])

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Fetch OSM data for a specific coordinate using osmnx.features_from_point
def fetch_osm_data_for_location(lat, lon, dist=3000):  # Increased to 3 km radius
    try:
        # Fetch highways
        highway_tags = {'highway': True}
        logger.info(f"Fetching highways for ({lat}, {lon}) within {dist} meters")
        highways = ox.features_from_point((lat, lon), tags=highway_tags, dist=dist)
        if not highways.empty:
            highways = highways[highways.geom_type.isin(['LineString', 'Point'])]
            logger.info(f"Fetched {len(highways)} highways near ({lat}, {lon})")
            if len(highways) > 0:
                logger.info(f"Sample highway: {highways.iloc[0].to_dict()}")
        else:
            logger.warning(f"No highways found near ({lat}, {lon}) within {dist} meters")

        # Fetch land use
        landuse_tags = {'landuse': True}
        logger.info(f"Fetching landuse for ({lat}, {lon}) within {dist} meters")
        landuse = ox.features_from_point((lat, lon), tags=landuse_tags, dist=dist)
        if not landuse.empty:
            logger.info(f"Raw landuse GeoDataFrame: {landuse.head().to_dict()}")
            landuse = landuse[landuse.geom_type == 'Polygon']
            logger.info(f"Fetched {len(landuse)} landuse features near ({lat}, {lon}) after filtering")
            if len(landuse) > 0:
                logger.info(f"Sample landuse: {landuse.iloc[0].to_dict()}")
        else:
            logger.warning(f"No landuse features found near ({lat}, {lon}) within {dist} meters")

        # Fetch power infrastructure
        power_tags = {'power': True}
        logger.info(f"Fetching power infrastructure for ({lat}, {lon}) within {dist} meters")
        power = ox.features_from_point((lat, lon), tags=power_tags, dist=dist)
        if not power.empty:
            power = power[power.geom_type.isin(['Point', 'LineString'])]
            logger.info(f"Fetched {len(power)} power features near ({lat}, {lon})")
            if len(power) > 0:
                logger.info(f"Sample power: {power.iloc[0].to_dict()}")
        else:
            logger.warning(f"No power features found near ({lat}, {lon}) within {dist} meters")

        return highways, landuse, power

    except Exception as e:
        logger.error(f"Failed to fetch OSM data for ({lat}, {lon}): {str(e)}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()

# Find distance to nearest highway
def distance_to_nearest_highway(lat, lon, highways):
    if highways.empty:
        logger.warning(f"No highways found near ({lat}, {lon})")
        return float('inf')
    min_dist = float('inf')
    for _, row in highways.iterrows():
        geom = row.geometry
        if geom.type == 'LineString':
            coords = list(geom.coords)
            for coord in coords:
                h_lat, h_lon = coord[1], coord[0]
                dist = haversine(lat, lon, h_lat, h_lon)
                min_dist = min(min_dist, dist)
        elif geom.type == 'Point':
            h_lat, h_lon = geom.y, geom.x
            dist = haversine(lat, lon, h_lat, h_lon)
            min_dist = min(min_dist, dist)
    logger.info(f"Distance to nearest highway from ({lat}, {lon}): {min_dist:.2f} km")
    return min_dist

# Get land type at a location (use nearest landuse polygon with UTM projection)
def get_land_type(lat, lon, landuse):
    if landuse.empty:
        logger.warning(f"No landuse data found near ({lat}, {lon})")
        return 'unknown'
    point = gpd.points_from_xy([lon], [lat], crs="EPSG:4326")[0]
    # Project to UTM zone 43N (Mumbai) for accurate distance calculations
    landuse_utm = landuse.to_crs("EPSG:32643")
    point_utm = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:32643")[0]
    # Find the nearest landuse polygon
    landuse_utm['distance'] = landuse_utm.geometry.distance(point_utm)
    nearest_landuse = landuse_utm.loc[landuse_utm['distance'].idxmin()]
    land_type = nearest_landuse.get('landuse', 'unknown')
    logger.info(f"Nearest land type to ({lat}, {lon}): {land_type} (distance: {nearest_landuse['distance']:.2f} meters)")
    return land_type

# Find distance to nearest power infrastructure
def distance_to_nearest_power(lat, lon, power):
    if power.empty:
        logger.warning(f"No power infrastructure found near ({lat}, {lon})")
        return float('inf')
    min_dist = float('inf')
    for _, row in power.iterrows():
        geom = row.geometry
        if geom.type == 'Point':
            p_lat, p_lon = geom.y, geom.x
            dist = haversine(lat, lon, p_lat, p_lon)
            min_dist = min(min_dist, dist)
        elif geom.type == 'LineString':
            coords = list(geom.coords)
            for coord in coords:
                p_lat, p_lon = coord[1], coord[0]
                dist = haversine(lat, lon, p_lat, p_lon)
                min_dist = min(min_dist, dist)
    logger.info(f"Distance to nearest power infrastructure from ({lat}, {lon}): {min_dist:.2f} km")
    return min_dist

# Find distance to nearest existing station
def distance_to_nearest_station(lat, lon, existing_stations):
    if existing_stations.empty:
        logger.warning("No existing stations provided")
        return float('inf'), None
    min_dist = float('inf')
    nearest_station = None
    for _, station in existing_stations.iterrows():
        dist = haversine(lat, lon, station['lat'], station['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest_station = station['name']
    logger.info(f"Distance to nearest existing station ({nearest_station}) from ({lat}, {lon}): {min_dist:.2f} km")
    return min_dist, nearest_station

def predict_locations(showrooms, existing_stations, eps_km=0.5, min_samples=2, max_distance_to_highway=10.0, max_distance_to_power=20.0, min_distance_to_station=1.0):
    df = pd.DataFrame([s.dict() for s in showrooms])
    required_cols = ['lat', 'lon', 'sales_car', 'sales_bus', 'sales_bike']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns")
    if not (df['lat'].between(18.8, 19.3)).all() or not (df['lon'].between(72.7, 73.1)).all():
        raise ValueError("Coordinates outside Mumbai bounds")
    if not (df[['sales_car', 'sales_bus', 'sales_bike']] >= 0).all().all():
        raise ValueError("Sales must be non-negative")

    weight_car, weight_bus, weight_bike = 1.0, 1.5, 0.5
    df['weighted_sales'] = (df['sales_car'] * weight_car +
                           df['sales_bus'] * weight_bus +
                           df['sales_bike'] * weight_bike)

    if df['weighted_sales'].max() == df['weighted_sales'].min():
        df['normalized_weighted_sales'] = 1
    else:
        df['normalized_weighted_sales'] = (
            (df['weighted_sales'] - df['weighted_sales'].min()) /
            (df['weighted_sales'].max() - df['weighted_sales'].min()) * 9 + 1
        ).round().astype(int)

    coords = []
    original_indices = []
    for idx, row in df.iterrows():
        for _ in range(row['normalized_weighted_sales']):
            coords.append([radians(row['lat']), radians(row['lon'])])
            original_indices.append(idx)
    coords = np.array(coords)

    if len(coords) == 0:
        logger.warning("No coordinates to cluster")
        return []
    eps_radians = eps_km / 6371
    db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine').fit(coords)
    labels = db.labels_

    # Step 1: Calculate initial predicted locations
    predicted_locations = []
    for cluster_label in set(labels) - {-1}:
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        original_showroom_indices = [original_indices[i] for i in cluster_indices]
        unique_showroom_indices = list(set(original_showroom_indices))
        cluster_showrooms = df.iloc[unique_showroom_indices]
        centroid_lat = cluster_showrooms['lat'].mean()
        centroid_lon = cluster_showrooms['lon'].mean()

        centroid_lat = max(18.8, min(19.3, centroid_lat))
        centroid_lon = max(72.7, min(73.1, centroid_lon))

        total_sales_car = int(cluster_showrooms['sales_car'].sum())
        total_sales_bus = int(cluster_showrooms['sales_bus'].sum())
        total_sales_bike = int(cluster_showrooms['sales_bike'].sum())
        total_weighted_sales = int(cluster_showrooms['weighted_sales'].sum())

        predicted_locations.append({
            "lat": centroid_lat,
            "lon": centroid_lon,
            "total_weighted_sales": total_weighted_sales,
            "total_sales_car": total_sales_car,
            "total_sales_bus": total_sales_bus,
            "total_sales_bike": total_sales_bike
        })
    logger.info(f"Predicted {len(predicted_locations)} initial locations")

    # Step 2: Post-prediction filtering and prioritization
    final_locations = []
    suitable_land_types = {"commercial", "urban", "unknown"}

    for loc in predicted_locations:
        lat, lon = loc["lat"], loc["lon"]

        # Check distance to nearest existing station
        dist_to_station, nearest_station = distance_to_nearest_station(lat, lon, existing_stations)
        if dist_to_station < min_distance_to_station:
            logger.info(f"Excluding location at ({lat}, {lon}): Too close to existing station {nearest_station} ({dist_to_station:.2f} km)")
            continue

        # Fetch OSM data for this specific location using osmnx.features_from_point
        local_highways, local_landuse, local_power = fetch_osm_data_for_location(lat, lon, dist=3000)

        # Add a delay to avoid Overpass API rate limits
        time.sleep(2)

        # If no data is fetched, fall back to default values
        if local_highways.empty and local_landuse.empty and local_power.empty:
            logger.info(f"No OSM data fetched for ({lat}, {lon}); using fallback values")
            dist_to_highway = 0.0
            land_type = "unknown"
            dist_to_power = 0.0
        else:
            # Calculate distance to nearest highway
            dist_to_highway = distance_to_nearest_highway(lat, lon, local_highways)
            
            # Check land type
            land_type = get_land_type(lat, lon, local_landuse)
            
            # Filter out unsuitable locations
            if land_type not in suitable_land_types:
                logger.info(f"Excluding location at ({lat}, {lon}): Unsuitable land type ({land_type})")
                continue
            
            # Filter out locations too far from highways
            if dist_to_highway > max_distance_to_highway:
                logger.info(f"Excluding location at ({lat}, {lon}): Too far from highway ({dist_to_highway} km)")
                continue

            # Check distance to power infrastructure
            dist_to_power = distance_to_nearest_power(lat, lon, local_power)
            if dist_to_power > max_distance_to_power:
                logger.info(f"Excluding location at ({lat}, {lon}): Too far from power infrastructure ({dist_to_power} km)")
                continue

        # Assign priority score
        priority_score = (1 / (dist_to_highway + 0.1)) + (1 / (dist_to_power + 0.1))
        
        # Add to final locations with priority
        final_locations.append({
            "lat": lat,
            "lon": lon,
            "priority_score": priority_score,
            "dist_to_highway": dist_to_highway,
            "land_type": land_type,
            "dist_to_power": dist_to_power,
            "dist_to_station": dist_to_station,
            "nearest_station": nearest_station,
            "total_weighted_sales": loc["total_weighted_sales"],
            "total_sales_car": loc["total_sales_car"],
            "total_sales_bus": loc["total_sales_bus"],
            "total_sales_bike": loc["total_sales_bike"]
        })

    # Step 3: Sort by priority score and format output
    final_locations.sort(key=lambda x: x["priority_score"], reverse=True)

    output_locations = []
    for loc in final_locations:
        reason = (f"High demand: {loc['total_weighted_sales']} weighted sales "
                  f"(Cars: {loc['total_sales_car']}, Buses: {loc['total_sales_bus']}, Bikes: {loc['total_sales_bike']}), "
                  f"{loc['dist_to_highway']:.2f} km from highway, "
                  f"Land type: {loc['land_type']}, "
                  f"{loc['dist_to_power']:.2f} km from power infrastructure, "
                  f"{loc['dist_to_station']:.2f} km from nearest existing station ({loc['nearest_station']})")
        output_locations.append({
            "lat": loc["lat"],
            "lon": loc["lon"],
            "reason": reason
        })

    return output_locations

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        logger.info(f"Received request with {len(request.showrooms)} showrooms")
        predictions = predict_locations(request.showrooms, existing_stations)
        logger.info(f"Returning {len(predictions)} predicted locations")
        return {"predicted_locations": predictions}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the EV Charging Station Location Finder API"}