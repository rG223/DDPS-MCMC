import numpy as np
def component_polygon_area(poly):
    """Compute the area of a component of a polygon.
    Args:
        x (ndarray): x coordinates of the component
        y (ndarray): y coordinates of the component

    Return:
        float: the are of the component
    """
    x = poly[:,0]
    y = poly[:,1]
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # np.roll 意即“滚动”，类似移位操作
        # 注意这里的np.dot表示一维向量相乘
