import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = r"D:\Vs Code\Python_Project\Mini Project (Price Prediction)\Book1(Ritika).csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical features
categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoders[col] = encoder

# Define features and target variable
features = ['State', 'Market', 'Commodity', 'Variety', 'Grade', 'Min Price', 'Max Price']
target = 'Modal Price'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict prices
y_pred = rf_model.predict(X_test_scaled)

# Add Commodity column back for visualization
X_test['Commodity'] = df.loc[X_test.index, 'Commodity']
y_test_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Commodity': X_test['Commodity']})

# Reverse Map Commodity Labels
if 'Commodity' in label_encoders:
    y_test_df['Commodity'] = label_encoders['Commodity'].inverse_transform(y_test_df['Commodity'])

# Compute mean prices per commodity
avg_prices = y_test_df.groupby('Commodity').mean().reset_index()

# Define manual colors for actual and predicted prices
actual_colors = ['seagreen', 'yellowgreen', 'darkviolet', 'orange', 'darkgreen']
predicted_colors = ['lightgreen', 'lightyellow', 'violet', 'darkorange', 'green']

# Define manual colors for each commodity
commodity_custom_colors = {
    "Bhindi(Ladies Finger)": "seagreen",
    "Brinjal": "darkviolet",
    "Spinach": "lightgreen",
    "Bitter gourd": "yellowgreen",
    "Mango": "darkorange"
}

# Assign colors dynamically if a commodity is not in the predefined dictionary
commodity_color_map = {
    commodity: commodity_custom_colors.get(commodity, color) 
    for commodity, color in zip(avg_prices["Commodity"], sns.color_palette("tab10", len(avg_prices["Commodity"])))
}

# Define different colors for actual and predicted prices
actual_price_colors = {commodity: commodity_color_map[commodity] for commodity in avg_prices["Commodity"]}
predicted_price_colors = {commodity: sns.desaturate(actual_price_colors[commodity], 0.5) for commodity in avg_prices["Commodity"]}


# Visualization Function
def visualize_data():
    """Function to generate graphs for Actual vs Predicted Prices."""
    
# --- 1️⃣ Line Graph ---
    plt.figure(figsize=(10, 6))
    plt.plot(avg_prices["Commodity"], avg_prices["Actual"], marker='o', linestyle='-', color='blue', label="Actual Price")
    plt.plot(avg_prices["Commodity"], avg_prices["Predicted"], marker='s', linestyle='--', color='red', label="Predicted Price")
    plt.xlabel("Commodity")
    plt.ylabel("Price (₹ per Kg)")
    plt.title("Actual vs Predicted Prices per Commodity (Line Graph)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# --- 2️⃣ Bar Graph ---
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    x_indexes = np.arange(len(avg_prices))

    bars_actual = plt.bar(x_indexes - bar_width / 2, avg_prices["Actual"], width=bar_width,
                           color=[actual_price_colors[commodity] for commodity in avg_prices["Commodity"]],
                           alpha=0.8, label="Actual Price")
    bars_predicted = plt.bar(x_indexes + bar_width / 2, avg_prices["Predicted"], width=bar_width,
                              color=[predicted_price_colors[commodity] for commodity in avg_prices["Commodity"]],
                              alpha=0.6, label="Predicted Price")

    for bar in bars_actual:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'₹{bar.get_height():.2f}', ha="center", va="bottom", fontsize=8, color='black')
    for bar in bars_predicted:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'₹{bar.get_height():.2f}', ha="center", va="bottom", fontsize=8, color='black')

    plt.xticks(x_indexes, avg_prices["Commodity"], rotation=90)
    plt.xlabel("Commodity")
    plt.ylabel("Price (Per Kg)")
    plt.title("Actual vs Predicted Prices per Commodity (Bar Graph)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    
# --- 3️⃣ Pie Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract actual and predicted values
    actual_sizes = avg_prices['Actual'].values
    predicted_sizes = avg_prices['Predicted'].values

    # Labels for Actual Prices (Outer Circle)
    labels_actual = [f"$\\bf{{{commodity}}}$\nA:\n ₹{actual:.2f}" 
                     for commodity, actual in zip(avg_prices['Commodity'], avg_prices['Actual'])]

    # Labels for Predicted Prices (Inner Circle)
    labels_predicted = [f"P: ₹{predicted:.2f}" 
                        for predicted in avg_prices['Predicted']]

    # Outer Pie Chart (Actual Prices)
    wedges1, texts1, autotexts1 = ax.pie(
        actual_sizes, labels=labels_actual, autopct=lambda p: ' ({:.1f}%)'.format(p) if p > 3 else "",
        startangle=140, colors=actual_colors, wedgeprops={'edgecolor': 'black'}, 
        radius=1.2, pctdistance=0.8, labeldistance=1.05, textprops={'fontsize': 10}
    )

    # Inner Pie Chart (Predicted Prices)
    wedges2, texts2, autotexts2 = ax.pie(
        predicted_sizes, labels=labels_predicted, autopct=lambda p: '\n\n({:.1f}%)'.format(p) if p > 4 else "",
        startangle=140, colors=predicted_colors, wedgeprops={'edgecolor': 'black'}, 
        radius=0.85, pctdistance=0.7, labeldistance=0.5, textprops={'fontsize': 10}
    )

    # Adjust text alignment
    for text in texts1 + texts2:
        text.set_horizontalalignment('center')
    
    # Set title with better spacing
    plt.title("Actual vs Predicted Prices per Kg by Commodity\n(Outer: Actual, Inner: Predicted)", fontsize=14, pad=20)

    # Show the pie chart
    plt.show()

# Execute visualization (in order: Line → Bar → Pie)
visualize_data()
