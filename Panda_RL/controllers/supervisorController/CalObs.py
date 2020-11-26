import math

def dist(x, y):
    return math.sqrt(x * x + y * y)

def get_delta_r_and_theta(origin_coord, endpoint_coord, target_coord):
    """
    Note:
        - the last element of 3 input argument should be the coordinate that we ignored
        - if want to change the delta_theta from degrees to radians,
            just comment out 'delta_theta = math.degrees(delta_theta)'

    Args:
        origin_coord: [x, y, z]
            an array of size 3 which containsorigin coordinate
        endpoint_coord: [x, y, z]
            an array of size 3 which containsendpoint coordinate
        target_coord: [x, y, z]
            an array of size 3 which containstarget coordinate

    Returns:
        delta_r: the absolute value of the diff of
            'the distance of origin, endpoint' and
            'the distance of origin, target'

        delta_tehta: the degree of the angle between
            'origin to endpoint' and
            'origin to target'
    """

    dist_oe = dist(origin_coord[0] - endpoint_coord[0],
                   origin_coord[1] - endpoint_coord[1])
    dist_et = dist(endpoint_coord[0] - target_coord[0],
                   endpoint_coord[1] - target_coord[1])
    dist_to = dist(target_coord[0] - origin_coord[0],
                   target_coord[1] - origin_coord[1])
    
    delta_r = abs(dist_to - dist_oe)

    cos_theta = (dist_oe * dist_oe + dist_to * dist_to - dist_et * dist_et) / (2 * dist_oe * dist_to)
    delta_theta = math.acos(cos_theta)
    delta_theta = math.degrees(delta_theta)
    
    return delta_r, delta_theta
