# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# I'm curious to see the structure of the data, so I'll start by loading it and checking the first few rows.
file_path = '/content/road_accident_dataset.csv'
data = pd.read_csv(file_path)

# Let's take a quick look at the dataset to understand its structure and see if there are any obvious issues.
print("Dataset Info:")
print(data.info())  # This will show column names, data types, and non-null counts.
print("\nPreview of Data:")
print(data.head())  # Display the first few rows for a quick overview.

# Step 2: Analyze the distribution of accident severity
# Accident severity seems like an important feature to explore. I'll start by checking its distribution.
print("\nAnalyzing Accident Severity Distribution:")
severity_distribution = data["Accident Severity"].value_counts(normalize=True)
print(severity_distribution)

# Visualizing the distribution for better clarity.
plt.figure(figsize=(8, 5))
severity_distribution.plot(kind="bar", color=["skyblue", "orange", "green"])
plt.title("Accident Severity Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Proportion")
plt.show()

# Step 3: Investigate the average medical cost for different severities
# Medical costs are another important factor. I'll calculate the average cost for each severity level.
print("\nAverage Medical Costs by Accident Severity:")
avg_medical_cost = data.groupby("Accident Severity")["Medical Cost"].mean()
print(avg_medical_cost)

# Let's visualize this to make the differences (if any) more apparent.
plt.figure(figsize=(8, 5))
avg_medical_cost.plot(kind="bar", color="purple")
plt.title("Average Medical Cost by Accident Severity")
plt.ylabel("Average Medical Cost")
plt.show()

# Step 4: Weather conditions during fatal accidents
# I want to see which weather conditions are associated with the most fatal accidents.
fatal_accidents_weather = data[data["Number of Fatalities"] > 0]["Weather Conditions"].value_counts()
print("\nTop Weather Conditions Leading to Fatal Accidents:")
print(fatal_accidents_weather.head(5))

# Visualizing the top 5 weather conditions for fatal accidents.
plt.figure(figsize=(8, 5))
fatal_accidents_weather.head(5).plot(kind="bar", color="red")
plt.title("Top Weather Conditions in Fatal Accidents")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Fatal Accidents")
plt.show()

# Step 5: Road conditions and injuries
# Road conditions likely play a significant role in the number of injuries. Let's analyze this.
avg_injuries_by_road_condition = data.groupby("Road Condition")["Number of Injuries"].mean()
print("\nAverage Number of Injuries by Road Condition:")
print(avg_injuries_by_road_condition.sort_values(ascending=False))

# Visualizing the findings.
plt.figure(figsize=(8, 5))
avg_injuries_by_road_condition.sort_values(ascending=False).plot(kind="bar", color="teal")
plt.title("Average Number of Injuries by Road Condition")
plt.ylabel("Average Number of Injuries")
plt.show()

# Step 6: Speed limit and vehicles involved
# Finally, I'm curious if there's a relationship between speed limits and the number of vehicles involved in accidents.
avg_vehicles_by_speed_limit = data.groupby("Speed Limit")["Number of Vehicles Involved"].mean()
print("\nSpeed Limit vs. Average Number of Vehicles Involved:")
print(avg_vehicles_by_speed_limit.head(5))

# Plotting the relationship.
plt.figure(figsize=(8, 5))
avg_vehicles_by_speed_limit.plot(kind="line", marker="o", color="darkgreen")
plt.title("Speed Limit vs Average Number of Vehicles Involved")
plt.xlabel("Speed Limit")
plt.ylabel("Average Number of Vehicles Involved")
plt.show()

# The EDA results have provided some meaningful insights so far! Each step reveals a different layer of the data.
