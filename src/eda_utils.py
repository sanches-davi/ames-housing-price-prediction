import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# DEF-01 Define a function to analyze a categorical feature against a target variable
def analyze_categorical_feature(data, feature, target='SalePrice', figsize=(10, 5)):
    
    summary = data.groupby(feature)[target].agg(                                    # Group the data by the selected feature and calculate target statistics
        ['count', 'mean', 'median', 'min', 'max']
    )

    summary = summary.sort_values(by='median', ascending=False)                     # Sort the summary table by median target value in descending order
    
    styled_summary = summary.style.format({                                         # Format the summary table for better readability
        'count': '{:,.0f}',
        'mean': '${:,.2f}',
        'median': '${:,.2f}',
        'min': '${:,.2f}',
        'max': '${:,.2f}'
    })
    
    display(styled_summary)                                                         # Display the formatted summary table
    
    category_order = data.groupby(feature)[target].median().sort_values().index     # Create an order for the categories based on median target value
    
    plt.figure(figsize=figsize)                                                     # Set the figure size for the chart
    sns.boxplot(                                                                    # Create a boxplot to compare target values across categories
        data=data,
        x=feature,
        y=target,
        order=category_order
    )
    plt.title(f'{target} by {feature}')                                             # Add title to the chart
    plt.xlabel(feature)                                                             # Add x-axis label
    plt.ylabel(target)                                                              # Add y-axis label
    plt.xticks(rotation=45, ha='right')                                             # Rotate x-axis labels for better readability
    plt.tight_layout()                                                              # Adjust layout to avoid cutting labels
    plt.show()                                                                      # Display the chart

    return summary