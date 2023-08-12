# Armut_arl_Project_Final
Armut Association Rule Learning and Product Recommendation

# Armut Association Rule Learning and Product Recommendation

This repository contains code for performing association rule learning on Armut's customer service data to generate product recommendations using the Apriori algorithm.

## Introduction

Armut, Turkey's largest online service platform, connects service providers with users seeking various services such as cleaning, renovation, and transportation. This project aims to create a product recommendation system using Association Rule Learning based on service data obtained from users and their selected categories.

## Data Preparation

The raw data consists of customer service selections, including service IDs, category IDs, and purchase dates. The data preprocessing steps include anonymizing service and category IDs, converting purchase dates, and creating a basket ID for association rule mining.

## Association Rule Generation

The code performs association rule mining using the Apriori algorithm to discover patterns and relationships between services selected by users. It generates frequent itemsets and association rules based on user service preferences.

## Product Recommendation

The `arl_recommender` function utilizes the generated association rules to provide product recommendations for a given service. It sorts the rules based on confidence and suggests related products as recommendations.

## Usage

1. Clone this repository to your local machine.
2. Install the required Python packages using:
   ```bash
   pip install pandas mlxtend
   
Prepare your dataset in CSV format with columns: UserId, ServiceId, CategoryId, and CreateDate.
Update the data file path in the provided code.
Run the code sections for data preparation, association rule generation, and product recommendation.
