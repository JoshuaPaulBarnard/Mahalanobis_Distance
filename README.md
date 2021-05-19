# Mahalanobis_Distance
Multivariate Outlier Analysis and Removal using Mahalanobis Distances and Z-scores

Receives a list, numpy array, or pandas dataframe.
Calculates mahalanobis distance values and p_values.
Returns a pandas dataframe of the z-scores of each data point, with the mahalanobis values and p-values attached.
When removal is True then the function returns the original dataset with outliers removed based on p-values.
When with_p is True then the function returns the original dataset with the p-values added as the last column.
