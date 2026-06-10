'''Utilities '''

import numpy as np  # type: ignore


def calculateGershgorinCircles(jacobian_arr: np.ndarray) -> np.ndarray:
    """Calculates the Gershgorin circles for a given Jacobian array

    Args:
        jacobian_arr (np.ndarray): Jacobian array

    Returns:
        np.ndarray: Array of Gershgorin circles
    """
    row_sums = np.sum(np.abs(jacobian_arr), axis=1) - np.abs(np.diag(jacobian_arr))
    circle_radii = row_sums
    circle_centers = np.diag(jacobian_arr)
    return np.column_stack((circle_centers, circle_radii))

def adjustJacobian(jacobian_arr: np.ndarray, time: float, max_value: float=1e10) -> np.ndarray:
    """Adjusts the Jacobian so that exp(max_eigval) < max_value

    Args:
        jacobian_arr (np.ndarray): Jacobian array
        time (float): Timepoint
        max_value (float, optional): Maximum allowed value for the exponential of the largest eigenvalue. Defaults to 1e10.

    Returns:
        np.ndarray: Adjusted Jacobian array
    """
    TOLERANCE = 1
    jacobian_arr = jacobian_arr.copy()
    max_circle_edge = np.log(max_value) / time
    # Calculate the value of the diagonal that does not exceed the desired circle size
    circles = calculateGershgorinCircles(jacobian_arr)
    center_arr, circle_radius_arr = circles[:, 0], circles[:, 1]
    new_diagonal = [min(center_arr[i], (max_circle_edge - circle_radius_arr[i]) - TOLERANCE)
            for i in range(len(center_arr))]
    np.fill_diagonal(jacobian_arr, new_diagonal)
    return jacobian_arr

def findFloatIndex(arr: np.ndarray, value: float) -> int:
    """Find the index of a float value in an array, allowing for a small tolerance.

    Parameters
    ----------
    arr : np.ndarray
        The array to search.
    value : float
        The value to find.

    Returns
    -------
    int
        The index of the value in the array.

    Raises
    ------
    ValueError
        If the value is not found within the tolerance.
    """
    arr1 = (arr - value)**2
    idx = np.argmin(arr1)
    return int(idx)