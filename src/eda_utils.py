import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def analyze_categorical_feature(data, feature, target='SalePrice', figsize=(10, 5)):
    
    summary = data.groupby(feature)[target].agg(                                   
        ['count', 'mean', 'median', 'min', 'max']
    )

    summary = summary.sort_values(by='median', ascending=False)                     
    
    styled_summary = summary.style.format({                                         
        'count': '{:,.0f}',
        'mean': '${:,.2f}',
        'median': '${:,.2f}',
        'min': '${:,.2f}',
        'max': '${:,.2f}'
    })                                                         
    
    category_order = data.groupby(feature)[target].median().sort_values().index     
    
    plt.figure(figsize=figsize)                                                     
    sns.boxplot(                                                                    
        data=data,
        x=feature,
        y=target,
        order=category_order
    )
    plt.title(f'{target} by {feature}')                                             
    plt.xlabel(feature)                                                             
    plt.ylabel(target)                                                              
    plt.xticks(rotation=45, ha='right')                                             
    plt.tight_layout()                                                              
    plt.show()                                                                      

    return summary

def calculate_iqr_outliers(data, feature):
    
    q1 = data[feature].quantile(0.25)                                               
    q3 = data[feature].quantile(0.75)                                               
    iqr = q3 - q1                                                                   
    lower_limit = q1 - 1.5 * iqr                                                    
    upper_limit = q3 + 1.5 * iqr                                                    
    lower_outliers = data[data[feature] < lower_limit].shape[0]                     
    upper_outliers = data[data[feature] > upper_limit].shape[0]                     
    total_outliers = lower_outliers + upper_outliers                                
    
    return q1, q3, iqr, lower_limit, upper_limit, lower_outliers, upper_outliers, total_outliers