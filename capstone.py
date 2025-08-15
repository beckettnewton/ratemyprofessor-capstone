#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: beckettnewton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import random
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress, pearsonr
from scipy.stats import f_oneway

N_number = 15179480
#DATA CLEANING/PREPROCESSING
N_number = 15179480
random.seed(N_number)
np.random.seed(N_number)
file_path = '/Users/beckettnewton/Desktop/ratemyprofessor-capstone/rmpCapstoneNum.csv'
data = pd.read_csv(file_path, header=None)
data.columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Received Pepper", "Proportion Retake", "Number Online Ratings", 
    "Male", "Female"
]
filtered_data = data.dropna(thresh=len(data.columns) - 1)

def print_regression_stats(model_name, X, y, y_pred, slope, intercept, p_value, std_err, r_squared):
    print(f"Regression: {model_name}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Standard Error: {std_err:.4f}")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)
    
##QUESTION 1:
male_ratings_filtered = filtered_data.loc[filtered_data["Male"] == 1, "Average Rating"].dropna()
female_ratings_filtered = filtered_data.loc[filtered_data["Female"] == 1, "Average Rating"].dropna()
mean_male = male_ratings_filtered.mean()
std_male = male_ratings_filtered.std()
n_male = len(male_ratings_filtered)
mean_female = female_ratings_filtered.mean()
std_female = female_ratings_filtered.std()
n_female = len(female_ratings_filtered)
pooled_std = np.sqrt((std_male**2 / n_male) + (std_female**2 / n_female))
z_stat = (mean_male - mean_female) / pooled_std
p_value = 2 * (1 - norm.cdf(abs(z_stat)))
plt.figure(figsize=(10, 6))
plt.hist(male_ratings_filtered, bins=30, alpha=0.6, label='Male Ratings', density=True)
plt.hist(female_ratings_filtered, bins=30, alpha=0.6, label='Female Ratings', density=True)
plt.axvline(mean_male, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean Male: {mean_male:.2f}')
plt.axvline(mean_female, color='orange', linestyle='dashed', linewidth=1.5, label=f'Mean Female: {mean_female:.2f}')
plt.title('Distribution of Ratings by Gender')
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print('Z-stat = ', z_stat)
print('P-value = ', p_value)
print('Male Mean = ', mean_male)
print('Female Mean = ', mean_female)

##QUESTION 2:
experience_data = filtered_data.dropna(subset=["Average Rating", "Number of Ratings"])
quartiles = np.percentile(experience_data["Number of Ratings"], [25, 50, 75])
experience_data["Experience Quartile"] = pd.cut(
    experience_data["Number of Ratings"],
    bins=[-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf],
    labels=["Q1", "Q2", "Q3", "Q4"]
)
quartile_means = experience_data.groupby("Experience Quartile")["Average Rating"].mean()
anova_result = f_oneway(
    experience_data.loc[experience_data["Experience Quartile"] == "Q1", "Average Rating"],
    experience_data.loc[experience_data["Experience Quartile"] == "Q2", "Average Rating"],
    experience_data.loc[experience_data["Experience Quartile"] == "Q3", "Average Rating"],
    experience_data.loc[experience_data["Experience Quartile"] == "Q4", "Average Rating"],
)
plt.figure(figsize=(10, 6))
quartile_means.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Mean Average Ratings by Experience Quartile", fontsize=16)
plt.xlabel("Experience Quartile", fontsize=14)
plt.ylabel("Mean Average Rating", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
for i, value in enumerate(quartile_means):
    plt.text(i, value + 0.01, f"{value:.2f}", ha="center", fontsize=12)
plt.show()
quartile_means, anova_result

##QUESTION 3
rating_difficulty_data = filtered_data.dropna(subset=["Average Rating", "Average Difficulty"])
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
    rating_difficulty_data["Average Difficulty"], rating_difficulty_data["Average Rating"]
)
print(f"Exact p-value: {p_value_reg:.2e}")
x = np.linspace(rating_difficulty_data["Average Difficulty"].min(), 
                rating_difficulty_data["Average Difficulty"].max(), 100)
y = slope * x + intercept
plt.figure(figsize=(10, 6))
plt.scatter(rating_difficulty_data["Average Difficulty"], rating_difficulty_data["Average Rating"], 
            alpha=0.5, label="Data Points", color="skyblue")
plt.plot(x, y, color="red", label=f"Regression Line (R²={r_value**2:.2f})")
plt.title("Relationship Between Average Rating and Average Difficulty", fontsize=16)
plt.xlabel("Average Difficulty", fontsize=14)
plt.ylabel("Average Rating", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
print(f"Correlation: {r_value:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
slope, intercept, r_value, p_value_reg, std_err = linregress(
    rating_difficulty_data["Average Difficulty"], rating_difficulty_data["Average Rating"]
)
y_pred = slope * rating_difficulty_data["Average Difficulty"] + intercept
r_squared = r_value**2
print_regression_stats(
    "Question 3: Difficulty vs Rating", 
    rating_difficulty_data["Average Difficulty"], 
    rating_difficulty_data["Average Rating"], 
    y_pred, slope, intercept, p_value_reg, std_err, r_squared
)

##QUESTION 4
group_0_online = filtered_data[filtered_data["Number Online Ratings"] == 0]
group_5_plus_online = filtered_data[filtered_data["Number Online Ratings"] >= 5]
assert group_0_online.shape[0] == 59077, "The sample size for 0 online ratings should be 59,077."
bootstrap_samples = 10000
bootstrap_means_group_5_plus = []
for _ in range(bootstrap_samples):
    bootstrap_sample = group_5_plus_online.sample(n=group_5_plus_online.shape[0], replace=True)
    bootstrap_means_group_5_plus.append(bootstrap_sample["Average Rating"].mean())
mean_rating_group_0_online = group_0_online["Average Rating"].mean()
mean_rating_group_5_plus_online = np.mean(bootstrap_means_group_5_plus)
lower_bound = np.percentile(bootstrap_means_group_5_plus, 2.5)
upper_bound = np.percentile(bootstrap_means_group_5_plus, 97.5)
plt.figure(figsize=(10, 6))
plt.bar(
    ["0 Online Ratings", "5+ Online Ratings (Bootstrapped)"], 
    [mean_rating_group_0_online, mean_rating_group_5_plus_online],
    yerr=[0, (upper_bound - lower_bound) / 2],
    capsize=10,
    alpha=0.7,
    color=["skyblue", "salmon"]
)
plt.ylabel("Average Rating", fontsize=14)
plt.title("Comparison of Mean Ratings", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
t_stat, p_value = stats.ttest_ind(
    group_0_online["Average Rating"], 
    group_5_plus_online["Average Rating"], 
    equal_var=False
)
{
    "Mean Rating Group 0 Online": mean_rating_group_0_online,
    "Bootstrapped Mean Rating Group 5+ Online": mean_rating_group_5_plus_online,
    "Confidence Interval Group 5+": (lower_bound, upper_bound),
    "T-statistic": t_stat,
    "P-value": p_value
}

##QUESTION 5
filtered_data["Proportion Retake"] = filtered_data["Proportion Retake"] / 100
retake_data = filtered_data.dropna(subset=["Average Rating", "Proportion Retake"])
train_data, test_data = train_test_split(retake_data, test_size=0.2, random_state=N_number)
slope, intercept, r_value, p_value_reg, std_err = linregress(
    train_data["Average Rating"], train_data["Proportion Retake"]
)
test_predictions = slope * test_data["Average Rating"] + intercept
test_r_squared = np.corrcoef(test_data["Proportion Retake"], test_predictions)[0, 1] ** 2
x = np.linspace(retake_data["Average Rating"].min(), retake_data["Average Rating"].max(), 100)
y = slope * x + intercept
plt.figure(figsize=(10, 6))
plt.scatter(test_data["Average Rating"], test_data["Proportion Retake"], 
            alpha=0.5, label="Test Data Points", color="skyblue")
plt.plot(x, y, color="red", label=f"Regression Line (Train R²={r_value**2:.2f})")
plt.title("Relationship Between Average Rating and Proportion of Retakes (Train-Test Split)", fontsize=16)
plt.xlabel("Average Rating", fontsize=14)
plt.ylabel("Proportion of Retakes", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
{
    "Train Pearson Correlation": pearsonr(train_data["Average Rating"], train_data["Proportion Retake"]),
    "Test R-squared": test_r_squared,
    "Train Slope": slope,
    "Train Intercept": intercept
}
y_pred_train = slope * train_data["Average Rating"] + intercept
print_regression_stats(
    "Question 5: Rating vs Retake (Train)", 
    train_data["Average Rating"], 
    train_data["Proportion Retake"], 
    y_pred_train, slope, intercept, p_value_reg, std_err, r_squared
)

##QUESTION 6
hotness_data = filtered_data.dropna(subset=["Average Rating", "Received Pepper"])
hot_ratings = hotness_data.loc[hotness_data["Received Pepper"] == 1, "Average Rating"]
not_hot_ratings = hotness_data.loc[hotness_data["Received Pepper"] == 0, "Average Rating"]
group_means_hotness = hotness_data.groupby("Received Pepper")["Average Rating"].mean()
t_stat_hotness, p_value_hotness = stats.ttest_ind(hot_ratings, not_hot_ratings, equal_var=False)
plt.figure(figsize=(10, 6))
boxprops = dict(patch_artist=True, showfliers=False)
hotness_data.boxplot(column="Average Rating", by="Received Pepper", grid=False, **boxprops)
plt.scatter(
    x=[1, 2], 
    y=[group_means_hotness[0], group_means_hotness[1]], 
    color="red", 
    zorder=3, 
    label="Mean Ratings"
)
plt.title("Comparison of Ratings by Hotness", fontsize=16)
plt.suptitle("") 
plt.xlabel("Received Pepper (0 = Not Hot, 1 = Hot)", fontsize=14)
plt.ylabel("Average Rating", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
group_means_hotness, t_stat_hotness, p_value_hotness

##QUESTION 7
regression_data = filtered_data.dropna(subset=["Average Rating", "Average Difficulty"])
X = regression_data["Average Difficulty"].values.reshape(-1, 1) 
y = regression_data["Average Rating"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N_number)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
slope = model.coef_[0]
intercept = model.intercept_
print(f"Train R^2: {r2_train:.2f}, Train RMSE: {rmse_train:.2f}")
print(f"Test R^2: {r2_test:.2f}, Test RMSE: {rmse_test:.2f}")
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Test Data Points", color="skyblue")
plt.plot(X_test, y_test_pred, color="red", label=f"Regression Line (R²={r2_test:.2f})")
plt.title("Regression Model: Predicting Average Rating from Difficulty (Test Set)", fontsize=16)
plt.xlabel("Average Difficulty", fontsize=14)
plt.ylabel("Average Rating", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
print_regression_stats(
    "Question 7: Difficulty vs Rating (Test)", 
    X_test.flatten(), 
    y_test, 
    y_test_pred, slope, intercept, p_value_reg, std_err, r2_test
)

##QUESTION 8
regression_data_all = filtered_data.dropna(subset=[
    "Average Rating", "Average Difficulty", "Number of Ratings",
    "Received Pepper", "Proportion Retake", "Number Online Ratings",
    "Male", "Female"
])
X_all = regression_data_all[[
    "Average Difficulty", "Number of Ratings", "Received Pepper",
    "Proportion Retake", "Number Online Ratings", "Male", "Female"
]]
y_all = regression_data_all["Average Rating"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=N_number
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_with_const = add_constant(X_train_scaled)
X_test_scaled_with_const = add_constant(X_test_scaled)
ols_model = OLS(y_train, X_train_scaled_with_const).fit()
y_train_pred = ols_model.predict(X_train_scaled_with_const)
y_test_pred = ols_model.predict(X_test_scaled_with_const)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Train R^2: {r2_train:.2f}, Train RMSE: {rmse_train:.2f}")
print(f"Test R^2: {r2_test:.2f}, Test RMSE: {rmse_test:.2f}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color="skyblue", label="Test Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label=f"Regression Line (R²={r2_test:.2f})")
plt.title("Predicted vs. Actual Ratings (Test Set)", fontsize=16)
plt.xlabel("Actual Average Rating", fontsize=14)
plt.ylabel("Predicted Average Rating", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
betas = ols_model.params
p_values = ols_model.pvalues
column_names = ["Intercept"] + [
    "Average Difficulty", "Number of Ratings", "Received Pepper",
    "Proportion Retake", "Number Online Ratings", "Male", "Female"
]
print("OLS Regression Results:")
for i, beta in enumerate(betas):
    column_name = column_names[i] if i < len(column_names) else f"Unknown_{i}"
    print(f"Coefficient for {column_name}: {beta:.4f}, P-value: {p_values[i]:.2e}")
def print_regression_stats(model_name, X, y, y_pred, slope, intercept, p_value, std_err, r_squared):
    print(f"Regression: {model_name}")
    print(f"Slope: {slope if slope is not None else 'N/A'}")
    print(f"Intercept: {intercept if intercept is not None else 'N/A'}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value: {p_value if p_value is not None else 'N/A'}")
    print(f"Standard Error: {std_err if std_err is not None else 'N/A'}")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)
print_regression_stats(
    "Question 8: Multiple Linear Regression (OLS)",
    X_test_scaled, 
    y_test, 
    y_test_pred, 
    None, None, None, None, r2_test
)

##QUESTION 9
classification_data = filtered_data.dropna(subset=["Average Rating", "Received Pepper"])
X_class = classification_data["Average Rating"].values.reshape(-1, 1) 
y_class = classification_data["Received Pepper"].values 
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=15179480, stratify=y_class
)
train_data = pd.DataFrame({"Average Rating": X_train.flatten(), "Received Pepper": y_train})
not_hot = train_data[train_data["Received Pepper"] == 0]
hot = train_data[train_data["Received Pepper"] == 1]
hot_oversampled = resample(hot, replace=True, n_samples=len(not_hot), random_state=15179480)
balanced_train_data = pd.concat([not_hot, hot_oversampled])
X_train_balanced = balanced_train_data["Average Rating"].values.reshape(-1, 1)
y_train_balanced = balanced_train_data["Received Pepper"].values
log_reg = LogisticRegression(random_state=15179480)
log_reg.fit(X_train_balanced, y_train_balanced)
y_test_proba = log_reg.predict_proba(X_test)[:, 1]
y_test_pred = log_reg.predict(X_test)
auc_score = roc_auc_score(y_test, y_test_proba)
accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred, target_names=["Not Hot", "Hot"])
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
accuracy, auc_score, classification_rep

##QUESTION 10
data = pd.read_csv(file_path, header=None)
data.columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Received Pepper", "Proportion Retake", "Number Online Ratings", 
    "Male", "Female"
]
filtered_data = data.dropna(thresh=len(data.columns) - 1)
classification_data = filtered_data.dropna(subset=["Average Rating", "Received Pepper"])
X_full = classification_data[["Average Rating", "Average Difficulty", "Number of Ratings", 
                               "Proportion Retake", "Number Online Ratings", "Male", "Female"]].values
y_full = classification_data["Received Pepper"].values
imputer = SimpleImputer(strategy="median")
X_full_imputed = imputer.fit_transform(X_full)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full_imputed, y_full, test_size=0.2, random_state=15179480, stratify=y_full
)
train_data_full = pd.DataFrame(X_train_full, columns=[
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Proportion Retake", "Number Online Ratings", "Male", "Female"
])
train_data_full["Received Pepper"] = y_train_full
not_hot_full = train_data_full[train_data_full["Received Pepper"] == 0]
hot_full = train_data_full[train_data_full["Received Pepper"] == 1]
hot_oversampled_full = resample(hot_full, replace=True, n_samples=len(not_hot_full), random_state=15179480)
balanced_train_data_full = pd.concat([not_hot_full, hot_oversampled_full])
X_train_balanced_full = balanced_train_data_full.drop("Received Pepper", axis=1).values
y_train_balanced_full = balanced_train_data_full["Received Pepper"].values
log_reg_full = LogisticRegression(random_state=15179480, max_iter=1000)
log_reg_full.fit(X_train_balanced_full, y_train_balanced_full)
y_test_proba_full = log_reg_full.predict_proba(X_test_full)[:, 1]
y_test_pred_full = log_reg_full.predict(X_test_full)
auc_score_full = roc_auc_score(y_test_full, y_test_proba_full)
accuracy_full = accuracy_score(y_test_full, y_test_pred_full)
classification_rep_full = classification_report(y_test_full, y_test_pred_full, target_names=["Not Hot", "Hot"])
fpr_full, tpr_full, _ = roc_curve(y_test_full, y_test_proba_full)
plt.figure(figsize=(10, 6))
plt.plot(fpr_full, tpr_full, label=f"ROC Curve (AUC = {auc_score_full:.2f})", color="blue")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")
plt.title("Receiver Operating Characteristic (ROC) Curve (Full Model)", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
accuracy_full, auc_score_full, classification_rep_full

##EXTRA CREDIT
num_file_path = '/Users/beckettnewton/Desktop/ratemyprofessor-capstone/rmpCapstoneNum.csv'
qual_file_path = '/Users/beckettnewton/Desktop/ratemyprofessor-capstone/rmpCapstoneQual.csv'
data_num = pd.read_csv(num_file_path, header=None)
data_qual = pd.read_csv(qual_file_path, header=None)
data_num.columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Received Pepper", "Proportion Retake", "Number Online Ratings", 
    "Male", "Female"
]
data_qual.columns = ["Major", "University", "US State"]
data_combined = pd.concat([data_num, data_qual], axis=1)
data_combined = data_combined.dropna(subset=["Major", "University", "US State"])  
data_combined = data_combined.loc[:, ~data_combined.columns.duplicated()] 
top_majors = data_combined["Major"].value_counts().head(5).index
pepper_by_major = data_combined[data_combined["Major"].isin(top_majors)].groupby("Major").agg({
    "Received Pepper": "mean",
    "Number of Ratings": "count"
}).rename(columns={"Number of Ratings": "Count of Professors"}).reset_index()
plt.figure(figsize=(10, 6))
plt.bar(pepper_by_major["Major"], pepper_by_major["Received Pepper"], color="green", alpha=0.7)
plt.title("Likelihood of Receiving a 'Pepper' by Major (Top 5 Majors)")
plt.xlabel("Major")
plt.ylabel("Proportion Receiving 'Pepper'")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
top_major_groups = data_combined[data_combined["Major"].isin(top_majors)]
pepper_by_major_test = top_major_groups.groupby("Major")["Received Pepper"].apply(list)
f_stat, p_value = stats.f_oneway(*pepper_by_major_test)
f_stat, p_value
