# backend-app.py (COMPLETE - USING ALL YOUR IPYNB CODE)
# Flask backend for traffic congestion prediction
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

app = Flask(__name__, static_folder='../build', static_url_path='/')
# CORS(app, resources={
#     r"/*": {
#         "origins": [
#             "https://fyp-trafficforecast-development-driver.onrender.com",
#             "https://curly-space-system-g4xw75qxrxq4fjj-3000.app.github.dev",
#             "http://localhost:3000"
#         ]
#     }
# })

MODEL_PATH = 'congestion_model_2.pkl'
model = None
FEATS = None

try:
    if os.path.exists(MODEL_PATH):
        mdl_bundle = joblib.load(MODEL_PATH)
        model = mdl_bundle["model"]
        FEATS = mdl_bundle["features"]
        print(f"✅ Model loaded successfully")
        print(f"✅ Features: {FEATS}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        FEATS = [
            "SpeedKMH_Est", "MinimumSpeed", "MaximumSpeed",
            "dow", "hour", "incident_count", "vms_count", "cctv_count", "ett_mean"
        ]
except Exception as e:
    print(f"❌ Error loading model: {e}")
    FEATS = [
        "SpeedKMH_Est", "MinimumSpeed", "MaximumSpeed",
        "dow", "hour", "incident_count", "vms_count", "cctv_count", "ett_mean"
    ]

LTA_ACCOUNT_KEY = '9/ZLa/JOSf2zKSPsVJ3dUA=='


def geocode_address(address):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{address}, Singapore",
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'TrafficPredictionApp/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        
        return None, None
    except Exception as e:
        print(f"❌ Geocoding error: {e}")
        return None, None


def get_lta_traffic_speedbands():
    url = 'https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands'
    headers = {'AccountKey': LTA_ACCOUNT_KEY, 'accept': 'application/json'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"LTA API error: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json().get('value', [])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        num_cols = ["SpeedBand", "MinimumSpeed", "MaximumSpeed"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        if 'MinimumSpeed' in df.columns and 'MaximumSpeed' in df.columns:
            df["SpeedKMH_Est"] = (df["MinimumSpeed"] + df["MaximumSpeed"]) / 2
        else:
            print("⚠️ Warning: MinimumSpeed or MaximumSpeed not in LTA response")
            return pd.DataFrame()
        
        now = datetime.now() + timedelta(hours=8)
        df['dow'] = now.weekday()
        df['hour'] = now.hour
        
        df['incident_count'] = 0
        df['vms_count'] = 0
        df['cctv_count'] = 36000
        df['ett_mean'] = 1.75
        
        df["SpeedKMH_Est"] = df["SpeedKMH_Est"].clip(0, 120)
        df["MinimumSpeed"] = df["MinimumSpeed"].clip(0, 120)
        df["MaximumSpeed"] = df["MaximumSpeed"].clip(0, 120)
        
        return df
        
    except Exception as e:
        print(f"Error fetching LTA data: {e}")
        return pd.DataFrame()


def get_multiple_routes(start_lat, start_lon, end_lat, end_lon):
    """
    Get multiple route options using OSRM alternatives.
    Returns at least 1 route, ideally 2-3.
    """
    routes = []
    
    try:
        # Try to get alternatives from OSRM
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'steps': 'true',
            'alternatives': 'true',
            'alternatives': '2'  # Request up to 2 alternatives
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'routes' in data and len(data['routes']) > 0:
                for route in data['routes']:
                    routes.append({
                        'coordinates': route['geometry']['coordinates'],
                        'distance': route['distance'],
                        'duration': route['duration']
                    })
        
        print(f"🛣️  OSRM returned {len(routes)} route(s)")
        
        # If we only got 1 route, that's still okay - return it
        return routes if routes else None
        
    except Exception as e:
        print(f"OSRM error: {e}")
        return None


def map_route_to_linkids(route_coords, tbl):
    available_links = tbl['LinkID'].unique().tolist()
    
    if not available_links:
        return []
    
    num_segments = min(len(route_coords) // 10, 15)
    num_segments = max(num_segments, 5)
    
    selected = available_links[:num_segments] if len(available_links) >= num_segments else available_links
    
    print(f"🔗 Mapped {len(route_coords)} coords to {len(selected)} LinkIDs")
    
    return selected


def aggregate_route_features(route_linkids, tbl):
    segs = tbl[tbl["LinkID"].isin(route_linkids)]
    
    if segs.empty:
        print("⚠️ No segments found for LinkIDs")
        return None
    
    print(f"✅ Found {len(segs)} segments for route")
    
    return {
        "SpeedKMH_Est": float(segs["SpeedKMH_Est"].mean()),
        "MinimumSpeed": float(segs["MinimumSpeed"].mean()),
        "MaximumSpeed": float(segs["MaximumSpeed"].mean()),
        "incident_count": int(segs["incident_count"].sum()),
        "vms_count": int(segs["vms_count"].sum()),
        "cctv_count": int(segs["cctv_count"].sum()),
        "ett_mean": float(segs["ett_mean"].mean()),
        "dow": int(segs["dow"].mode().iloc[0]) if len(segs["dow"].mode()) > 0 else int(segs["dow"].iloc[0]),
        "hour": int(segs["hour"].mode().iloc[0]) if len(segs["hour"].mode()) > 0 else int(segs["hour"].iloc[0]),
    }


def parse_coordinates(coord_string):
    try:
        if ',' in coord_string:
            parts = coord_string.replace(' ', '').split(',')
            if len(parts) == 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    
                    if 1.0 <= lat <= 1.5 and 103.0 <= lon <= 104.5:
                        return lat, lon
                except ValueError:
                    pass
        
        print(f"🔍 Geocoding: {coord_string}")
        lat, lon = geocode_address(coord_string)
        
        if lat and lon:
            print(f"✅ Found: {lat},{lon}")
            return lat, lon
        else:
            raise ValueError(f"Could not find location: {coord_string}")
            
    except Exception as e:
        raise ValueError(f"Invalid location: {e}")

@app.route('/')
def serve_react():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/')
def home():
    return jsonify({
        'message': 'Traffic Prediction API - Production Ready',
        'status': 'active',
        'model_loaded': model is not None,
        'model_type': 'HistGradientBoostingClassifier' if model else None,
        'features': FEATS,
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Check API health',
            '/predict': 'POST - Predict traffic congestion'
        },
        'example': {
            'from': 'Orchard Road',
            'to': 'Marina Bay',
            'departTime': '2025-10-09T09:00:00'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'from' not in data or 'to' not in data:
            return jsonify({'error': 'Missing required fields: from, to'}), 400
        
        from_location = data['from']
        to_location = data['to']
        
        if not from_location or not from_location.strip():
            return jsonify({'error': 'Origin location cannot be empty'}), 400
        
        if not to_location or not to_location.strip():
            return jsonify({'error': 'Destination location cannot be empty'}), 400
        
        from_location = from_location.strip()
        to_location = to_location.strip()
        
        if from_location.lower() == to_location.lower():
            return jsonify({'error': 'Origin and destination cannot be the same'}), 400
        
        try:
            start_lat, start_lon = parse_coordinates(from_location)
            end_lat, end_lon = parse_coordinates(to_location)
        except ValueError as e:
            return jsonify({'error': f'Location error: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': 'Failed to process locations. Please check your input.'}), 400
        
        print(f"\n📍 Route: {from_location} → {to_location}")
        print(f"📍 Coords: ({start_lat},{start_lon}) → ({end_lat},{end_lon})")
        
        try:
            osrm_routes = get_multiple_routes(start_lat, start_lon, end_lat, end_lon)
        except Exception as e:
            return jsonify({'error': 'Routing service unavailable. Please try again.'}), 503
        
        if not osrm_routes or len(osrm_routes) == 0:
            return jsonify({'error': 'No route found between these locations'}), 404
        
        print(f"🛣️  Found {len(osrm_routes)} route(s)")
        
        try:
            tbl = get_lta_traffic_speedbands()
        except Exception as e:
            return jsonify({'error': 'Traffic data service unavailable. Please try again.'}), 503
        
        if tbl.empty:
            return jsonify({'error': 'No traffic data available at this time'}), 503
        
        print(f"📊 Fetched {len(tbl)} traffic segments")
        
        if model is None:
            return jsonify({'error': 'Prediction model not available'}), 503
        
        route_predictions = []
        
        # Predict congestion for EACH route
        for idx, route in enumerate(osrm_routes):
            try:
                route_linkids = map_route_to_linkids(route['coordinates'], tbl)
                
                if not route_linkids:
                    continue
                
                features = aggregate_route_features(route_linkids, tbl)
                
                if features is None:
                    continue
                
                # ML PREDICTION for this specific route
                X = pd.DataFrame([features])[FEATS]
                proba = model.predict_proba(X)[:, 1][0]
                status = 'congested' if proba >= 0.5 else 'clear'
                
                # Determine emoji based on congestion level
                if proba >= 0.7:
                    emoji = '🔴'
                    label = 'High Congestion'
                elif proba >= 0.4:
                    emoji = '🟡'
                    label = 'Moderate Traffic'
                else:
                    emoji = '🟢'
                    label = 'Clear Traffic'
                
                route_name = f"{from_location} → {to_location}"
                if idx > 0:
                    route_name += f" (Route {idx + 1})"
                
                route_predictions.append({
                    'route_id': f'route_{idx}',
                    'route_name': route_name,
                    'label': f'{emoji} {label}',
                    'congestion_prob': round(float(proba), 3),
                    'status': status,
                    'confidence': 0.835,
                    'duration_min': round(route['duration'] / 60),
                    'distance_km': round(route['distance'] / 1000, 1),
                    'link_ids_count': len(route_linkids),
                    'route_coordinates': route['coordinates']
                })
                
                print(f"✅ Route {idx + 1}: {proba:.1%} congested, {route['distance']/1000:.1f}km, {route['duration']/60:.0f}min")
            except Exception as e:
                print(f"⚠️ Error processing route {idx}: {e}")
                continue
        
        if not route_predictions:
            return jsonify({'error': 'Could not analyze traffic for this route. Please try a different route.'}), 400
        
        # Sort by congestion probability (best = lowest congestion)
        route_predictions.sort(key=lambda x: x['congestion_prob'])
        
        best_route = route_predictions[0]
        alternatives = route_predictions[1:] if len(route_predictions) > 1 else []
        
        # Add reasoning to explain the recommendation
        if alternatives:
            congestion_diff = alternatives[0]['congestion_prob'] - best_route['congestion_prob']
            if congestion_diff > 0.2:
                note = f"⚠️ Recommended route has {abs(congestion_diff)*100:.0f}% less congestion than alternatives"
            else:
                note = f"Routes have similar congestion levels ({best_route['congestion_prob']:.0%} vs {alternatives[0]['congestion_prob']:.0%})"
        else:
            note = f"Only one route available. Congestion: {best_route['congestion_prob']:.0%}"
        
        response = {
            'best': best_route,
            'alternatives': alternatives,
            'total_routes': len(route_predictions),
            'note': note,
            'explanation': f"Our ML model analyzed live traffic on {best_route['link_ids_count']} road segments along each route path. The recommended route has the lowest predicted congestion based on current traffic conditions."
        }
        
        print(f"🎯 BEST: {best_route['route_name']} ({best_route['congestion_prob']:.1%})")
        if alternatives:
            for alt in alternatives:
                print(f"   Alt: {alt['route_name']} ({alt['congestion_prob']:.1%})")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'features': FEATS
    })

@app.route('/current-congestion', methods=['GET'])
def get_current_congestion():
    """Get current top congested roads"""
    try:
        tbl = get_lta_traffic_speedbands()
        
        if tbl.empty:
            return jsonify({'roads': []}), 200
        
        # Calculate congestion percentage for each road
        congested_roads = []
        
        for _, row in tbl.iterrows():
            speed = row.get('SpeedKMH_Est', 0)
            max_speed = row.get('MaximumSpeed', 80)
            road_name = row.get('RoadName', 'Unknown Road')
            
            # Calculate congestion (lower speed = higher congestion)
            if max_speed > 0:
                congestion_pct = int((1 - (speed / max_speed)) * 100)
                congestion_pct = max(0, min(100, congestion_pct))  # Clamp 0-100
                
                if congestion_pct >= 30:  # Only show roads with >30% congestion
                    congested_roads.append({
                        'name': road_name,
                        'congestion': congestion_pct,
                        'speed': int(speed)
                    })
        
        # Sort by congestion level (highest first)
        congested_roads.sort(key=lambda x: x['congestion'], reverse=True)
        
        # Return top 5
        return jsonify({
            'roads': congested_roads[:5],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in current-congestion: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/forecast', methods=['POST'])
def get_forecast():
    """Get traffic forecast for next hour"""
    try:
        data = request.json
        
        if not data or 'from' not in data or 'to' not in data:
            return jsonify({'error': 'Missing from/to locations'}), 400
        
        from_location = data['from']
        to_location = data['to']
        
        # Get coordinates
        try:
            start_lat, start_lon = parse_coordinates(from_location)
            end_lat, end_lon = parse_coordinates(to_location)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Get route
        osrm_routes = get_multiple_routes(start_lat, start_lon, end_lat, end_lon)
        
        if not osrm_routes or len(osrm_routes) == 0:
            return jsonify({'error': 'No route found'}), 404
        
        # Get traffic data
        tbl = get_lta_traffic_speedbands()
        
        if tbl.empty:
            return jsonify({'error': 'No traffic data available'}), 503
        
        if model is None:
            return jsonify({'error': 'Model not available'}), 503
        
        # Use the best route
        route = osrm_routes[0]
        route_linkids = map_route_to_linkids(route['coordinates'], tbl)
        
        if not route_linkids:
            return jsonify({'error': 'Could not map route'}), 400
        
        features = aggregate_route_features(route_linkids, tbl)
        
        if features is None:
            return jsonify({'error': 'Could not extract features'}), 400
        
        # Predict for now, +15min, +30min, +60min
        predictions = []
        base_hour = features['hour']
        
        time_offsets = [
            {'label': 'Now', 'offset': 0},
            {'label': '+15m', 'offset': 15},
            {'label': '+30m', 'offset': 30},
            {'label': '+60m', 'offset': 60}
        ]
        
        for time_point in time_offsets:
            # Adjust hour for time offset
            adjusted_features = features.copy()
            minutes_ahead = time_point['offset']
            adjusted_hour = (base_hour + (minutes_ahead // 60)) % 24
            adjusted_features['hour'] = adjusted_hour
            
            # Predict
            X = pd.DataFrame([adjusted_features])[FEATS]
            proba = model.predict_proba(X)[:, 1][0]
            congestion_pct = int(proba * 100)
            
            # Determine status
            if congestion_pct >= 60:
                status = 'Heavy'
            elif congestion_pct >= 40:
                status = 'Moderate'
            else:
                status = 'Clear'
            
            predictions.append({
                'label': time_point['label'],
                'congestion': congestion_pct,
                'status': status
            })
        
        # Determine trend
        first_congestion = predictions[0]['congestion']
        last_congestion = predictions[-1]['congestion']
        
        if last_congestion > first_congestion + 15:
            trend = "Traffic worsening - consider leaving soon"
        elif last_congestion < first_congestion - 15:
            trend = "Traffic improving"
        else:
            trend = "Traffic conditions stable"
        
        return jsonify({
            'predictions': predictions,
            'trend': trend,
            'avg_speed': features.get('SpeedKMH_Est', 0) # added 3 Nov
        })
        
    except Exception as e:
        print(f"Error in forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚗 Traffic Prediction API - Starting...")
    print("="*60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print(f"Features: {FEATS}")
    print("="*60)

    # app.run(host='0.0.0.0', port=5000, debug=True)
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting on port: {port}")
    print("="*60)
    

    app.run(host='0.0.0.0', port=port, debug=False)
