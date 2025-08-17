import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Convert y_train and y_test to series
y_train = y_train.iloc[:, 0]  # Take first column if it's a DataFrame
y_test = y_test.iloc[:, 0]

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Feature columns: {list(X_train.columns)}")

# Load preprocessing objects for future use
try:
    with open('preprocessing_objects.pkl', 'rb') as f:
        preprocessing_objects = pickle.load(f)
    print("Loaded preprocessing objects")
except FileNotFoundError:
    print("Warning: preprocessing_objects.pkl not found")

# Define models to train
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=6
    ),
    'Linear Regression': LinearRegression()
}

# Train and evaluate models
results = {}
best_model = None
best_score = float('-inf')

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Create progress bar for all models
model_names = list(models.keys())
for i, name in enumerate(tqdm(model_names, desc="Training Models", unit="model")):
    print(f"\nTraining {name}...")
    
    # Train the model with progress bar
    if name == 'Random Forest':
        # For Random Forest, show progress for each tree
        rf_model = models[name]
        rf_model.warm_start = True
        rf_model.n_estimators = 1
        
        for n_trees in tqdm(range(1, 101), desc=f"Training {name}", unit="tree", leave=False):
            rf_model.n_estimators = n_trees
            rf_model.fit(X_train, y_train)
    elif name == 'Gradient Boosting':
        # For Gradient Boosting, show progress for each boosting stage
        gb_model = models[name]
        gb_model.warm_start = True
        gb_model.n_estimators = 1
        
        for n_estimators in tqdm(range(1, 101), desc=f"Training {name}", unit="stage", leave=False):
            gb_model.n_estimators = n_estimators
            gb_model.fit(X_train, y_train)
    else:
        # For other models, just fit normally
        models[name].fit(X_train, y_train)
    
    # Make predictions
    y_pred = models[name].predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    print(f"  Running cross-validation...")
    cv_scores = cross_val_score(models[name], X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'model': models[name],
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred
    }
    
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = name

print(f"\nBest model: {best_model} (R²: {best_score:.4f})")

# Hyperparameter tuning for the best model
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

if best_model == 'Random Forest':
    print("Tuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Calculate total combinations for progress bar
    total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
    print(f"Testing {total_combinations} parameter combinations...")
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    
    # Custom progress bar for grid search
    with tqdm(total=total_combinations, desc="Hyperparameter Tuning", unit="combination") as pbar:
        # Monkey patch the GridSearchCV to show progress
        original_fit = grid_search.fit
        
        def fit_with_progress(X, y):
            result = original_fit(X, y)
            pbar.update(total_combinations)
            return result
        
        grid_search.fit = fit_with_progress
        grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Update best model with tuned parameters
    results['Random Forest (Tuned)'] = {
        'model': grid_search.best_estimator_,
        'mae': mean_absolute_error(y_test, grid_search.predict(X_test)),
        'mse': mean_squared_error(y_test, grid_search.predict(X_test)),
        'rmse': np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test))),
        'r2': r2_score(y_test, grid_search.predict(X_test)),
        'cv_mean': grid_search.best_score_,
        'cv_std': 0,  # GridSearchCV doesn't provide std
        'predictions': grid_search.predict(X_test)
    }
    
    best_model = 'Random Forest (Tuned)'

# Feature importance analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")

# Model comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[name]['mae'] for name in results.keys()],
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'R²': [results[name]['r2'] for name in results.keys()],
    'CV R²': [results[name]['cv_mean'] for name in results.keys()]
})

print(comparison_df.sort_values('R²', ascending=False))

# Save the best model
print("\n" + "="*60)
print("SAVING BEST MODEL")
print("="*60)

best_model_obj = results[best_model]['model']
joblib.dump(best_model_obj, 'resale_price_model.pkl')
print(f"Best model ({best_model}) saved as 'resale_price_model.pkl'")

# Save all results
with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("All model results saved as 'model_results.pkl'")

# Save comparison table
comparison_df.to_csv('model_comparison.csv', index=False)
print("Model comparison saved as 'model_comparison.csv'")

# Final evaluation of best model
print(f"\nFinal evaluation of {best_model}:")
print(f"  MAE: ${results[best_model]['mae']:,.2f}")
print(f"  RMSE: ${results[best_model]['rmse']:,.2f}")
print(f"  R²: {results[best_model]['r2']:.4f}")

# Prediction vs Actual plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, results[best_model]['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Prices ({best_model})')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("Prediction vs Actual plot saved as 'prediction_vs_actual.png'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print(f"✓ Best model: {best_model}")
print(f"✓ Model saved: resale_price_model.pkl")
print(f"✓ Results saved: model_results.pkl")
print(f"✓ Comparison saved: model_comparison.csv")
print(f"✓ Plots saved: feature_importance.png, prediction_vs_actual.png")