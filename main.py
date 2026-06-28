import pandas as pd
import matplotlib.pyplot as plt

print("=====================================")
print(" PYTHON DATA ANALYSIS DASHBOARD ")
print("=====================================")

# Load the Titanic dataset
df = pd.read_csv("data/titanic.csv")

# Display first 5 rows
print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

print("\nTotal Passengers:")
print(len(df))

print("\nTotal Survivors:")
print(df["Survived"].sum())

print("\nTotal Non-Survivors:")
print((df["Survived"] == 0).sum())

print("\nAverage Age:")
print(df["Age"].mean())

print("\nOldest Passenger:")
print(df["Age"].max())

print("\nYoungest Passenger:")
print(df["Age"].min())

print("\nMale and Female Passengers:")
print(df["Sex"].value_counts())

print("\nPassenger Classes:")
print(df["Pclass"].value_counts())

print("\nFemale Passengers:")
print(df[df["Sex"] == "female"])

print("\nPassengers Older Than 30:")
print(df[df["Age"] > 30])

print("\nPassengers Who Survived:")
print(df[df["Survived"] == 1])

print("\nPassengers Sorted by Age (Ascending):")
print(df.sort_values("Age"))

print("\nPassengers Sorted by Age (Descending):")
print(df.sort_values("Age", ascending=False))

gender_counts = df["Sex"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(gender_counts.index, gender_counts.values)

plt.title("Male vs Female Passengers")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.savefig("visualizations/gender_distribution.png")

plt.show()

gender_counts = df["Sex"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    gender_counts.values,
    labels=gender_counts.index,
    autopct="%1.1f%%"
)

plt.title("Gender Distribution")

plt.savefig("visualizations/gender_pie_chart.png")

plt.show()

plt.figure(figsize=(6,4))

plt.hist(df["Age"].dropna(), bins=5)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number of Passengers")

plt.savefig("visualizations/age_histogram.png")

plt.show()

print("\nAverage Age by Gender:")
print(df.groupby("Sex")["Age"].mean())

print("\nGender Statistics:")
print(df.groupby("Sex")["Age"].agg(["count", "mean", "min", "max"]))

print("\nDataset Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset with Missing Age Filled:")

df_copy = df.copy()

df_copy["Age"] = df_copy["Age"].fillna(df_copy["Age"].mean())

print(df_copy)

# Save cleaned dataset
df_copy.to_csv("data/titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")