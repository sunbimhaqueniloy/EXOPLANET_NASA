from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained models
print("🚀 Loading AI models...")
try:
    detective = joblib.load('exoplanet_detector.pkl')
    imputer = joblib.load('feature_imputer.pkl')
    # If you have feature_info.pkl, load it too
    # feature_info = joblib.load('feature_info.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    detective = None
    imputer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_planet():
    try:
        # Get data from frontend
        data = request.json
        
        # Prepare features
        features = {
            'koi_period': float(data.get('period', 10)),
            'koi_depth': float(data.get('depth', 1000)),
            'koi_duration': float(data.get('duration', 3)),
            'koi_impact': float(data.get('impact', 0.3)),
            'koi_teq': float(data.get('teq', 500)),
            'koi_model_snr': float(data.get('snr', 20)),
            'koi_steff': float(data.get('steff', 6000)),
            'koi_slogg': float(data.get('slogg', 4.4)),
            'koi_fpflag_nt': int(data.get('fp_nt', 0)),
            'koi_fpflag_ss': int(data.get('fp_ss', 0)),
            'koi_fpflag_co': int(data.get('fp_co', 0)),
            'koi_fpflag_ec': int(data.get('fp_ec', 0))
        }
        
        # Convert to DataFrame
        feature_names = [
            'koi_period', 'koi_depth', 'koi_duration', 'koi_impact',
            'koi_teq', 'koi_model_snr', 'koi_steff', 'koi_slogg',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        input_df = pd.DataFrame([features])[feature_names]
        
        # Make prediction
        if detective and imputer:
            X_imputed = imputer.transform(input_df)
            prediction = detective.predict(X_imputed)[0]
            confidence = detective.predict_proba(X_imputed)[0, 1]
            
            feature_importance = dict(zip(feature_names, detective.feature_importances_))
        else:
            # Fallback simulation
            prediction, confidence, feature_importance = simulate_prediction(features)
        
        result = {
            'success': True,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'feature_importance': feature_importance,
            'key_factors': get_key_factors(features, confidence)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def simulate_prediction(features):
    """Fallback if models don't load"""
    confidence = 0.5
    if 5 < features['koi_period'] < 400: confidence += 0.1
    if 50 < features['koi_depth'] < 50000: confidence += 0.1
    if 1 < features['koi_duration'] < 10: confidence += 0.1
    if features['koi_impact'] < 0.5: confidence += 0.1
    confidence = max(0, min(1, confidence))
    prediction = 1 if confidence > 0.5 else 0
    
    feature_importance = {name: 0.08 for name in [
        'koi_period', 'koi_depth', 'koi_duration', 'koi_impact',
        'koi_teq', 'koi_model_snr', 'koi_steff', 'koi_slogg',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
    ]}
    
    return prediction, confidence, feature_importance

def get_key_factors(features, confidence):
    factors = []
    
    if 5 < features['koi_period'] < 400:
        factors.append("Orbital period is within typical range")
    else:
        factors.append("Orbital period is unusual")
    
    if features['koi_fpflag_nt'] == 0:
        factors.append("Signal appears transit-like")
    else:
        factors.append("Signal doesn't look like a transit")
    
    return factors

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)