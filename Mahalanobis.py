import pandas as pd
import numpy as np
from scipy.stats import chi2

def mahalanobis( data=None, removal=False, with_p=False ):
    """
    Receives a list, numpy array, or pandas dataframe.
    Calculates mahalanobis distance values and p_values.
    Returns a pandas dataframe of the z-scores of each data point, with the mahalanobis values and p-values attached.
    When removal is True then the function returns the original dataset with outliers removed based on p-values.
    When with_p is True then the function returns the original dataset with the p-values added as the last column.
    :param data:
    :param removal:
    :param with_p:
    :return view_results:
    """

    # Prepare our Data
    data = pd.DataFrame( data )

    #   Compute our neded statistics
    data_residual = data - np.mean(data)
    covariance = np.cov(data.values.T)
    inv_covariance = np.linalg.inv(covariance)
    left = np.dot(data_residual, inv_covariance)

    #   Calculate our Mahalanobis Distance Values and their probabilities
    mahalanobis_values = np.dot(left, data_residual.T).diagonal()
    p_values = 1 - chi2.cdf(mahalanobis_values, 3)

    #   Return the dataset with outliers removed
    if removal is True:
        outliers_removed = data
        outliers_removed['p'] = p_values
        outliers_removed = outliers_removed.drop(outliers_removed[outliers_removed.p < .001].index)
        del outliers_removed['p']
        return outliers_removed.iloc[:, :]

    #   Return the original dataset with p-values added
    elif with_p is True:
        pvalues_added = data
        pvalues_added['p'] = p_values
        return pvalues_added

    #   Returns a new dataset for analsis, with the z-score of each datapoint and the p-value added as the last column
    else:
        view_results = (data - np.mean(data)) / np.std(data)
        view_results['Mahalanobis'] = mahalanobis_values
        view_results['p'] = p_values
        print(view_results)
        return view_results
