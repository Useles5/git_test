import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# API URL
API_URL = "http://localhost:8000/predict"

# Load existing stations (simplified for this example)
existing_stations = pd.DataFrame([
    {"name": "Station 1", "lat": 19.0500, "lon": 72.8500},
    {"name": "Station 2", "lat": 19.0900, "lon": 72.8900},
])

# Initialize session state
if "predicted_locations" not in st.session_state:
    st.session_state.predicted_locations = []

# Bento layout
st.title("EV Charging Station Location Finder")
col1, col2 = st.columns([0.3, 0.7])

# Left column: File upload and summaries
with col1:
    st.subheader("Upload Showroom Data")
    uploaded_file = st.file_uploader("Upload CSV (name, lat, lon, sales_car, sales_bus, sales_bike)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ['name', 'lat', 'lon', 'sales_car', 'sales_bus', 'sales_bike']
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain: name, lat, lon, sales_car, sales_bus, sales_bike")
        else:
            # Send data to API
            showrooms = df.to_dict("records")
            try:
                response = requests.post(API_URL, json={"showrooms": showrooms})
                response.raise_for_status()
                st.session_state.predicted_locations = response.json()["predicted_locations"]
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

    # Summaries
    st.subheader("Summaries")
    st.metric("Total Existing Stations", len(existing_stations))
    st.metric("Total Showrooms Uploaded", len(df) if uploaded_file else 0)
    st.metric("Total Predicted Locations", len(st.session_state.predicted_locations))
    if st.session_state.predicted_locations:
        st.write("Predicted Locations:")
        for loc in st.session_state.predicted_locations:
            st.write(f"Lat: {loc['lat']}, Lon: {loc['lon']}, Reason: {loc['reason']}")

# Right column: Map
with col2:
    st.subheader("Map")
    # Prepare data for map
    map_data = existing_stations.copy()
    map_data['type'] = 'Existing Station'
    map_data['color'] = 'red'

    if st.session_state.predicted_locations:
        predicted_df = pd.DataFrame(st.session_state.predicted_locations)
        predicted_df['type'] = 'Predicted Location'
        predicted_df['color'] = 'blue'
        map_data = pd.concat([map_data, predicted_df[['lat', 'lon', 'type', 'color', 'reason']]], ignore_index=True)

    # Plot map
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        color="type",
        color_discrete_map={"Existing Station": "red", "Predicted Location": "blue"},
        hover_data=["reason"] if 'reason' in map_data.columns else None,
        zoom=10,
        height=500
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# Run with: streamlit run app_streamlit.py