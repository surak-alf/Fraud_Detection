import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging (do this once)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def univariate_analysis(data):
    """Performs and displays univariate analysis on the given DataFrame."""

    def plot_categorical(data, column, title, figsize=(8, 5)):
        """Plots a categorical variable."""
        plt.figure(figsize=figsize)
        sns.countplot(x=column, data=data, order=data[column].value_counts().index)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_numerical(data, column, title, figsize=(10, 6), bins=30):  # Added bins parameter
        """Plots a numerical variable (histogram with KDE and box plot)."""
        plt.figure(figsize=figsize)
        sns.histplot(data[column], kde=True, bins=bins)  # Use bins parameter
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.boxplot(y=data[column])
        plt.title(title + " (Box Plot)")
        plt.ylabel(column)
        plt.show()

    # Categorical Variables
    plot_categorical(data, 'sex', 'Sex Distribution')
    plot_categorical(data, 'source', 'Source Distribution', figsize=(10, 6))
    plot_categorical(data, 'browser', 'Browser Distribution', figsize=(12, 7))
    plot_categorical(data, 'class', 'Class Distribution (Fraud vs. Not Fraud)')  # Class distribution

    # Numerical Variables
    plot_numerical(data, 'purchase_value', 'Distribution of Purchase Values')
    plot_numerical(data, 'age', 'Age Distribution')  # Age distribution (histogram + boxplot)



