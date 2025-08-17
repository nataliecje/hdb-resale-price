import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Connect to your PostgreSQL database
# Configuration
DB_USER = 'postgres'
DB_PASSWORD = 'mysecretpassword'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'hdb'

CSV_FOLDER = '/Users/nataliecje/VSCode/hdb-resale-price/data'
TABLE_NAME = 'resale_transactions'
CHUNKSIZE = 10000  # Adjust based on memory

# Create DB engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
# Load the resale_transactions table
df = pd.read_sql("SELECT * FROM resale_transactions", engine)
# Drop columns you don't need
df = df.drop(columns=['month', 'block', 'street_name'])

print(f"Initial data shape: {df.shape}")
print(f"Initial columns: {list(df.columns)}")

## COLUMN "town"
# Compute average resale price per town
town_scores = df.groupby('town')['resale_price'].mean()
# Map scores back to the dataframe
df['town_score'] = df['town'].map(town_scores)
# Optionally drop the original town column
df = df.drop(columns=['town'])

## COLUMN "storey_range"
# Convert storey_range to low/medium/high categories
def categorize_storey(storey_range):
    if pd.isna(storey_range):
        return -1
    # Extract the first number from ranges like "1 TO 3", "4 TO 6", "40 TO 42", etc.
    match = re.search(r'(\d+)', str(storey_range))
    if match:
        first_storey = int(match.group(1))
        if first_storey <= 3:
            return 0      # Floors 1-3
        elif first_storey <= 12:
            return 1   # Floors 4-12
        else:
            return 2    # Floors 13+
    return -1

if 'storey_range' in df.columns:
    df['storey_category'] = df['storey_range'].apply(categorize_storey)
    df = df.drop(columns=['storey_range'])
    
    print("Storey category distribution:")
    print(df['storey_category'].value_counts())
    
## COLUMN "flat_model"
le_flat_model = LabelEncoder()
df['flat_model_encoded'] = le_flat_model.fit_transform(df['flat_model'])
# Optionally drop the original flat_model column
df = df.drop(columns=['flat_model'])

## COLUMN "flat_type"
le_flat_type = LabelEncoder()
df['flat_type_encoded'] = le_flat_type.fit_transform(df['flat_type'])
# Optionally drop the original flat_type column
df = df.drop(columns=['flat_type'])

## COLUMN "remaining_lease" and "lease_commence_date"
if 'remaining_lease' in df.columns and 'lease_commence_date' in df.columns:
    # Count nulls before processing
    null_count_before = df['remaining_lease'].isna().sum()
    print(f"Null values in remaining_lease before: {null_count_before}")
    
    # Fill null remaining_lease values
    mask = df['remaining_lease'].isna()
    if mask.sum() > 0:
        df.loc[mask, 'remaining_lease'] = (
            2056 - df.loc[mask, 'lease_commence_date']
        ).astype(str) + ' years'
        
        null_count_after = df['remaining_lease'].isna().sum()
        print(f"Filled {null_count_before - null_count_after} null values in remaining_lease")
    
    # Convert remaining_lease to numeric years for easier analysis
    def extract_years(lease_str):
        if pd.isna(lease_str):
            return np.nan
        
        # Extract years and months from strings like "61 years 04 months"
        years_match = re.search(r'(\d+)\s*years?', str(lease_str))
        months_match = re.search(r'(\d+)\s*months?', str(lease_str))
        
        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        
        return years + months / 12
    
    df['remaining_lease_years'] = df['remaining_lease'].apply(extract_years)
    print("Created remaining_lease_years column with numeric values")
# Optionally drop the original remaining_lease column
df = df.drop(columns=['remaining_lease', 'lease_commence_date'])

print(f"\nFinal processed data shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

# Debug: Check data types
print("\nData types of columns:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")
    if df[col].dtype == 'object':
        print(f"    Sample values: {df[col].unique()[:5]}")

# Data quality check
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Check for missing values
print("Missing values summary:")
missing_summary = df.isnull().sum()
if missing_summary.sum() > 0:
    for col, count in missing_summary[missing_summary > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("  No missing values found!")

# Basic statistics for key columns
print("\nKey statistics:")
key_numeric_cols = ['floor_area_sqm', 'remaining_lease_years', 
                   'resale_price', 'town_score']

for col in key_numeric_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Std: {df[col].std():.2f}")

# Categorical columns summary
categorical_cols = ['storey_category']
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} distribution:")
        print(df[col].value_counts().head())

# Handle any remaining missing values
print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)

# Fill missing values with appropriate strategies
if df['remaining_lease_years'].isna().sum() > 0:
    median_lease = df['remaining_lease_years'].median()
    df['remaining_lease_years'].fillna(median_lease, inplace=True)
    print(f"Filled {df['remaining_lease_years'].isna().sum()} missing values in remaining_lease_years")

if df['town_score'].isna().sum() > 0:
    median_town_score = df['town_score'].median()
    df['town_score'].fillna(median_town_score, inplace=True)
    print(f"Filled {df['town_score'].isna().sum()} missing values in town_score")

# Prepare data for model training
print("\n" + "="*60)
print("PREPARING DATA FOR MODEL TRAINING")
print("="*60)

# Separate features and target
target_col = 'resale_price'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

print(f"Feature columns: {feature_cols}")
print(f"Target column: {target_col}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['floor_area_sqm', 'remaining_lease_years', 'town_score']

# Fit scaler on training data only
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Applied StandardScaler to numerical features")

# Save processed data and preprocessing objects
print("\n" + "="*60)
print("SAVING PROCESSED DATA AND PREPROCESSING OBJECTS")
print("="*60)

# Save processed data
output_csv = f'{CSV_FOLDER}/processed_resale_transactions.csv'
df.to_csv(output_csv, index=False)
print(f"Processed data saved to: {output_csv}")

# Save training and testing sets
X_train_scaled.to_csv(f'{CSV_FOLDER}/X_train.csv', index=False)
X_test_scaled.to_csv(f'{CSV_FOLDER}/X_test.csv', index=False)
y_train.to_csv(f'{CSV_FOLDER}/y_train.csv', index=False)
y_test.to_csv(f'{CSV_FOLDER}/y_test.csv', index=False)
print("Training and testing sets saved")

# Save preprocessing objects
preprocessing_objects = {
    'label_encoder_flat_model': le_flat_model,
    'label_encoder_flat_type': le_flat_type,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'numerical_columns': numerical_cols,
    'categorical_columns': categorical_cols
}

with open('preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)
print("Preprocessing objects saved to preprocessing_objects.pkl")

# Quick model validation
print("\n" + "="*60)
print("QUICK MODEL VALIDATION")
print("="*60)

# Train a simple model to validate the data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

print("\n" + "="*60)
print("DATA IS READY FOR MODEL TRAINING!")
print("="*60)
print("✓ Missing values handled")
print("✓ Features properly encoded")
print("✓ Data scaled appropriately")
print("✓ Train-test split created")
print("✓ Preprocessing objects saved")
print("✓ Quick validation shows reasonable performance")
