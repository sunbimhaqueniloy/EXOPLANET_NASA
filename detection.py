import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load the data
df = pd.read_csv('kepler_koi.csv', comment='#')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nBasic info:")
print(df.info())

# Data exploration
print("\n=== KEPLER KOI DATA ANALYSIS ===")

# 1. Disposition analysis
print("\n1. Planet Disposition Analysis:")
print(df['koi_disposition'].value_counts())

# 2. Confirmed planets analysis
confirmed_planets = df[df['koi_disposition'] == 'CONFIRMED']
print(f"\nNumber of confirmed planets: {len(confirmed_planets)}")

# 3. Planetary radius distribution
plt.figure(figsize=(15, 10))

# Plot 1: Planet dispositions
plt.subplot(2, 3, 1)
disposition_counts = df['koi_disposition'].value_counts()
plt.pie(disposition_counts.values, labels=disposition_counts.index, autopct='%1.1f%%')
plt.title('Planet Dispositions')

# Plot 2: Planetary radius distribution
plt.subplot(2, 3, 2)
confirmed_with_radius = confirmed_planets[confirmed_planets['koi_prad'].notna()]
plt.hist(confirmed_with_radius['koi_prad'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Planetary Radius (Earth Radii)')
plt.ylabel('Count')
plt.title('Distribution of Planetary Radii\n(Confirmed Planets)')
plt.axvline(confirmed_with_radius['koi_prad'].median(), color='red', linestyle='--', 
           label=f'Median: {confirmed_with_radius["koi_prad"].median():.1f} R⊕')
plt.legend()

# Plot 3: Orbital period distribution
plt.subplot(2, 3, 3)
confirmed_with_period = confirmed_planets[confirmed_planets['koi_period'].notna()]
plt.hist(confirmed_with_period['koi_period'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Orbital Period (days)')
plt.ylabel('Count')
plt.title('Distribution of Orbital Periods\n(Confirmed Planets)')
plt.axvline(confirmed_with_period['koi_period'].median(), color='red', linestyle='--',
           label=f'Median: {confirmed_with_period["koi_period"].median():.1f} days')
plt.legend()

# Plot 4: Equilibrium temperature
plt.subplot(2, 3, 4)
confirmed_with_teq = confirmed_planets[confirmed_planets['koi_teq'].notna()]
plt.hist(confirmed_with_teq['koi_teq'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Equilibrium Temperature (K)')
plt.ylabel('Count')
plt.title('Distribution of Equilibrium Temperatures\n(Confirmed Planets)')
plt.axvline(confirmed_with_teq['koi_teq'].median(), color='red', linestyle='--',
           label=f'Median: {confirmed_with_teq["koi_teq"].median():.0f} K')
plt.legend()

# Plot 5: Planetary radius vs orbital period
plt.subplot(2, 3, 5)
valid_data = confirmed_planets.dropna(subset=['koi_prad', 'koi_period'])
plt.scatter(valid_data['koi_period'], valid_data['koi_prad'], alpha=0.6)
plt.xlabel('Orbital Period (days)')
plt.ylabel('Planetary Radius (Earth Radii)')
plt.title('Planetary Radius vs Orbital Period\n(Confirmed Planets)')
plt.xscale('log')

# Plot 6: Stellar temperature distribution
plt.subplot(2, 3, 6)
confirmed_with_stellar = confirmed_planets[confirmed_planets['koi_steff'].notna()]
plt.hist(confirmed_with_stellar['koi_steff'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Stellar Effective Temperature (K)')
plt.ylabel('Count')
plt.title('Distribution of Stellar Temperatures\n(Host Stars of Confirmed Planets)')
plt.axvline(confirmed_with_stellar['koi_steff'].median(), color='red', linestyle='--',
           label=f'Median: {confirmed_with_stellar["koi_steff"].median():.0f} K')
plt.legend()

plt.tight_layout()
plt.show()

# Statistical summary
print("\n2. Statistical Summary of Confirmed Planets:")
numeric_columns = ['koi_period', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_depth', 'koi_duration']
print(confirmed_planets[numeric_columns].describe())

# 3. Multi-planet systems analysis
print("\n3. Multi-Planet Systems Analysis:")
# Group by Kepler ID to find systems with multiple planets
multi_planet_systems = df.groupby('kepid').filter(lambda x: len(x) > 1)
multi_planet_counts = multi_planet_systems['kepid'].value_counts()

print(f"Number of systems with multiple planets: {len(multi_planet_counts)}")
print(f"Maximum number of planets in one system: {multi_planet_counts.max()}")

# 4. Interesting findings
print("\n4. Interesting Findings:")

# Largest confirmed planet
largest_planet = confirmed_planets.loc[confirmed_planets['koi_prad'].idxmax()]
print(f"Largest confirmed planet: {largest_planet.get('kepler_name', 'Unknown')} "
      f"({largest_planet['koi_prad']:.1f} Earth radii)")

# Smallest confirmed planet
smallest_planet = confirmed_planets.loc[confirmed_planets['koi_prad'].idxmin()]
print(f"Smallest confirmed planet: {smallest_planet.get('kepler_name', 'Unknown')} "
      f"({smallest_planet['koi_prad']:.2f} Earth radii)")

# Shortest orbital period
shortest_period = confirmed_planets.loc[confirmed_planets['koi_period'].idxmin()]
print(f"Shortest orbital period: {shortest_period.get('kepler_name', 'Unknown')} "
      f"({shortest_period['koi_period']:.3f} days)")

# Longest orbital period
longest_period = confirmed_planets.loc[confirmed_planets['koi_period'].idxmax()]
print(f"Longest orbital period: {longest_period.get('kepler_name', 'Unknown')} "
      f"({longest_period['koi_period']:.1f} days)")

# 5. False positive analysis
false_positives = df[df['koi_disposition'] == 'FALSE POSITIVE']
print(f"\n5. False Positives Analysis:")
print(f"Number of false positives: {len(false_positives)}")

# Reasons for false positives (checking the flags)
fp_flags = false_positives[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']].sum()
print("\nFalse positive flags:")
for flag, count in fp_flags.items():
    flag_name = {
        'koi_fpflag_nt': 'Not Transit-like',
        'koi_fpflag_ss': 'Stellar Eclipse', 
        'koi_fpflag_co': 'Centroid Offset',
        'koi_fpflag_ec': 'Ephemeris Match'
    }.get(flag, flag)
    print(f"  {flag_name}: {int(count)}")

# 6. Signal-to-noise analysis
print("\n6. Signal-to-Noise Analysis:")
if 'koi_model_snr' in df.columns:
    confirmed_snr = confirmed_planets['koi_model_snr'].dropna()
    print(f"Median SNR for confirmed planets: {confirmed_snr.median():.1f}")
    print(f"Max SNR for confirmed planets: {confirmed_snr.max():.1f}")

print("\n" + "="*50)
print("🚀 STARTING MACHINE LEARNING MODEL TRAINING")
print("="*50)

# 🚀 STEP 1: Create Target Variable
print("🎯 STEP 1: Defining our mission...")
df['is_real_planet'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
print(f"Real planets: {df['is_real_planet'].sum()}")
print(f"Fake planets: {len(df) - df['is_real_planet'].sum()}")

# 📊 STEP 2: Select Features
print("\n📊 STEP 2: Choosing detective clues...")
features = [
    'koi_period',        # Orbit time
    'koi_depth',         # How much star dims
    'koi_duration',      # Transit length  
    'koi_impact',        # Transit path
    'koi_teq',          # Planet temperature
    'koi_model_snr',     # Signal strength
    'koi_steff',        # Star temperature
    'koi_slogg',        # Star surface gravity
    'koi_fpflag_nt',     # Not transit-like flag
    'koi_fpflag_ss',     # Stellar eclipse flag
    'koi_fpflag_co',     # Centroid offset flag
    'koi_fpflag_ec'      # Ephemeris match flag
]

# Create ML dataset
X = df[features]
y = df['is_real_planet']

print(f"Features: {len(features)}")
print(f"Samples: {len(X)}")

# Check for missing values
print(f"\nMissing values in features:")
print(X.isnull().sum())

# 🔧 STEP 3: Handle Missing Values
print("\n🔧 STEP 3: Cleaning up missing clues...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
print("✅ Missing values filled!")

# 🎪 STEP 4: Split Data
print("\n🎪 STEP 4: Training and testing split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} planets")
print(f"Testing set: {X_test.shape[0]} planets")
print(f"Real planets in training: {y_train.sum()}")
print(f"Real planets in testing: {y_test.sum()}")

# 🤖 STEP 5: Train Your First Detective!
print("\n🤖 STEP 5: Training the AI detective...")
detective = RandomForestClassifier(n_estimators=100, random_state=42)
detective.fit(X_train, y_train)
print("✅ Detective trained successfully!")

# 🎯 STEP 6: Test the Detective
print("\n🎯 STEP 6: Testing detective skills...")
y_pred = detective.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_proba = detective.predict_proba(X_test)[:, 1]  # Probability scores
roc_auc = roc_auc_score(y_test, y_proba)

print(f"📊 Detective Accuracy: {accuracy:.2%}")
print(f"🎯 ROC-AUC Score: {roc_auc:.2%}")
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# 🔍 STEP 7: What Clues Matter Most?
print("\n🔍 STEP 7: Detective's most important clues:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': detective.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# 📈 STEP 8: Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features for Planet Detection')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 🎭 STEP 9: Confusion Matrix
print("\n🎭 STEP 9: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Fake', 'Predicted Real'],
            yticklabels=['Actual Fake', 'Actual Real'])
plt.title('Confusion Matrix - Planet Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\n" + "="*50)
print("🎉 MACHINE LEARNING MODEL COMPLETE!")
print("="*50)
print(f"✅ Your AI detective can spot real planets with {accuracy:.1%} accuracy!")
print(f"✅ The most important clues are: {feature_importance.iloc[0]['feature']} and {feature_importance.iloc[1]['feature']}")
# 🎮 INTERACTIVE PLANET DETECTOR
print("\n" + "="*60)
print("🎮 INTERACTIVE EXOPLANET DETECTOR")
print("="*60)

def interactive_planet_detector():
    """Interactive tool to detect planets in real-time!"""
    
    print("\n🌌 WELCOME TO THE EXOPLANET DETECTOR!")
    print("I'll help you determine if a celestial object is a REAL planet!")
    print("Please enter the following features of your candidate:\n")
    
    # Sample values for reference
    print("💡 SAMPLE VALUES FOR REFERENCE:")
    print("   - Good planet: Period=10d, Depth=1000ppm, Duration=3h, SNR=20")
    print("   - Likely fake: Period=1d, Depth=50000ppm, Duration=1h, SNR=5")
    print()
    
    try:
        # Get user input with helpful descriptions
        print("📅 ORBITAL CHARACTERISTICS:")
        period = float(input("Orbital Period (days) - Time for one orbit: "))
        depth = float(input("Transit Depth (ppm) - How much star dims during transit: "))
        duration = float(input("Transit Duration (hours) - How long transit lasts: "))
        impact = float(input("Impact Parameter (0-1) - Transit path across star (0=center, 1=edge): "))
        
        print("\n🌡️ PHYSICAL PROPERTIES:")
        teq = float(input("Equilibrium Temperature (K) - Planet surface temp: "))
        snr = float(input("Signal-to-Noise Ratio - How clear the signal is (higher=better): "))
        
        print("\n⭐ HOST STAR PROPERTIES:")
        steff = float(input("Star Temperature (K) - How hot the host star is: "))
        slogg = float(input("Star Surface Gravity - Star's surface gravity (usually 4.0-4.9): "))
        
        print("\n🚫 FALSE POSITIVE FLAGS (Enter 0 for none, 1 if present):")
        fp_nt = int(input("Not Transit-like flag (0/1): "))
        fp_ss = int(input("Stellar Eclipse flag (0/1): "))
        fp_co = int(input("Centroid Offset flag (0/1): "))
        fp_ec = int(input("Ephemeris Match flag (0/1): "))
        
        # Create feature dictionary
        candidate = {
            'koi_period': max(period, 0.1),      # Avoid zero/negative
            'koi_depth': max(depth, 1),          # Avoid zero
            'koi_duration': max(duration, 0.1),  # Avoid zero
            'koi_impact': max(min(impact, 0.99), 0),  # Keep between 0-1
            'koi_teq': max(teq, 100),           # Reasonable minimum
            'koi_model_snr': max(snr, 0.1),     # Avoid zero
            'koi_steff': max(steff, 3000),      # Reasonable star temp
            'koi_slogg': max(min(slogg, 5.0), 3.5),  # Reasonable range
            'koi_fpflag_nt': fp_nt,
            'koi_fpflag_ss': fp_ss,
            'koi_fpflag_co': fp_co,
            'koi_fpflag_ec': fp_ec
        }
        
        # Convert to dataframe
        candidate_df = pd.DataFrame([candidate])
        
        # Handle missing values (same way as training)
        candidate_imputed = imputer.transform(candidate_df)
        
        # Get prediction
        prediction = detective.predict(candidate_imputed)[0]
        confidence = detective.predict_proba(candidate_imputed)[0, 1]
        
        print("\n" + "="*50)
        print("🔭 ANALYSIS RESULTS:")
        print("="*50)
        
        if prediction == 1:
            if confidence > 0.8:
                print(f"🎉 EXCELLENT NEWS! This is very likely a REAL PLANET!")
                print(f"🪐 Confidence Level: {confidence:.1%}")
                print("🌟 This candidate shows strong planetary characteristics!")
            elif confidence > 0.6:
                print(f"✅ PROMISING! This appears to be a REAL PLANET!")
                print(f"🪐 Confidence Level: {confidence:.1%}")
                print("💫 This shows good planetary signatures!")
            else:
                print(f"🤔 INTERESTING! This might be a planet!")
                print(f"🪐 Confidence Level: {confidence:.1%}")
                print("🔍 More observation might be needed!")
        else:
            if confidence < 0.2:
                print(f"🚫 UNLIKELY to be a planet")
                print(f"📊 Confidence: {confidence:.1%}")
                print("💡 This shows characteristics of a false positive")
            elif confidence < 0.4:
                print(f"❓ UNCLEAR - Probably not a planet")
                print(f"📊 Confidence: {confidence:.1%}")
                print("🔧 The signals don't strongly indicate a planet")
            else:
                print(f"⚠️  AMBIGUOUS - Could go either way")
                print(f"📊 Confidence: {confidence:.1%}")
                print("🎯 Some planetary features, but also some red flags")
        
        print("\n📈 KEY FACTORS:")
        # Analyze which features helped the decision
        feature_contributions = detective.feature_importances_ * candidate_imputed[0]
        top_contributors = pd.DataFrame({
            'feature': features,
            'contribution': feature_contributions
        }).sort_values('contribution', ascending=False)
        
        print(f"➕ Most supportive features:")
        for i, (_, row) in enumerate(top_contributors.head(3).iterrows()):
            feature_name = row['feature'].replace('koi_', '').replace('_', ' ').title()
            print(f"   - {feature_name}")
        
        print(f"➖ Least supportive features:")
        for i, (_, row) in enumerate(top_contributors.tail(2).iterrows()):
            feature_name = row['feature'].replace('koi_', '').replace('_', ' ').title()
            print(f"   - {feature_name}")
            
        return candidate, prediction, confidence
        
    except ValueError:
        print("❌ Please enter valid numbers!")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def quick_test_cases():
    """Test some known planet examples"""
    print("\n🧪 QUICK TEST CASES:")
    
    test_cases = [
        {
            'name': 'Earth-like Planet',
            'features': {
                'koi_period': 365, 'koi_depth': 84, 'koi_duration': 13,
                'koi_impact': 0.1, 'koi_teq': 255, 'koi_model_snr': 15,
                'koi_steff': 5780, 'koi_slogg': 4.4, 'koi_fpflag_nt': 0,
                'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0
            }
        },
        {
            'name': 'Hot Jupiter', 
            'features': {
                'koi_period': 3.5, 'koi_depth': 15000, 'koi_duration': 2.5,
                'koi_impact': 0.3, 'koi_teq': 1200, 'koi_model_snr': 30,
                'koi_steff': 6000, 'koi_slogg': 4.3, 'koi_fpflag_nt': 0,
                'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0
            }
        },
        {
            'name': 'Likely False Positive',
            'features': {
                'koi_period': 1.2, 'koi_depth': 50000, 'koi_duration': 1.0,
                'koi_impact': 0.9, 'koi_teq': 2000, 'koi_model_snr': 5,
                'koi_steff': 6500, 'koi_slogg': 4.2, 'koi_fpflag_nt': 1,
                'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0
            }
        }
    ]
    
    for test in test_cases:
        candidate_df = pd.DataFrame([test['features']])
        candidate_imputed = imputer.transform(candidate_df)
        prediction = detective.predict(candidate_imputed)[0]
        confidence = detective.predict_proba(candidate_imputed)[0, 1]
        
        result = "🪐 PLANET" if prediction == 1 else "🚫 NOT PLANET"
        print(f"{test['name']}: {result} (Confidence: {confidence:.1%})")

# Run the interactive detector
while True:
    print("\nOptions:")
    print("1. 🔍 Detect a new planet candidate")
    print("2. 🧪 Run test cases") 
    print("3. 🚪 Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        candidate, prediction, confidence = interactive_planet_detector()
        
        # Ask if they want to save this candidate
        if candidate is not None:
            save = input("\n💾 Save this candidate for later? (y/n): ").lower()
            if save == 'y':
                # You could save to a file here
                print("✅ Candidate saved! (Feature: add file saving code)")
                
    elif choice == '2':
        quick_test_cases()
        
    elif choice == '3':
        print("👋 Thanks for using the Exoplanet Detector! Happy planet hunting! 🚀")
        break
        
    else:
        print("❌ Please enter 1, 2, or 3")

print("\n" + "="*60)
print("🎉 YOUR INTERACTIVE PLANET DETECTOR IS READY!")
print("="*60)
print("You can now:")
print("🔍 Analyze real planet candidates")
print("🧪 Test known examples") 
print("📊 Get confidence scores and explanations")
print("🚀 Discover new worlds!")
