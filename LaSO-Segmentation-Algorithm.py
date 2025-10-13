from typing import List, Tuple
import numpy as np
import random

# function that returns the squared distance between two points
def distance_between_points_squared(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
    return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2

# function that segments predicted points of all LaSOs into different LaSOs using correlation
# points -> all predicted points of LaSOs
# correlation_threshold -> a threshold if broken will imply that newly added point is not part of the current LaSO
# window_size -> how many points to look back and find the correlation coefficient to see if the newly added point is part of the LaSO
# i.e., if the newly added point drastically changes the correlation coefficient
def segment_LaSOs(points: List[Tuple[float, float]], correlation_drop_threshold: float, window_size: int = 3) -> List[List[Tuple[float, float]]]:

    LaSOs = [] # will contained the segmented LaSOS from the original set of points
    remaining_points = points # remaining points that we must process to segment the LaSOs

    while remaining_points:
        collected_points = []
        collected_points_x = []
        collected_points_y = []
        previous_correlation = None

        # we want to start from the most bottom left point
        anchor = min(remaining_points, key=lambda point: (point[0], point[1]))

        # since the LaSOs resemble a line, all points closest to the anchor have the highest probability to lie together in a LaSO (heuristic)
        remaining_points = sorted(remaining_points, key=lambda point: distance_between_points_squared(anchor, point))

        for point in remaining_points:

            # we consider the ith point to add it newly
            collected_points.append(point)
            collected_points_x.append(point[0])
            collected_points_y.append(point[1])

            # if we have enough points to calculate correlation coefficient
            if len(collected_points) >= window_size:
                # get the recently collected points (how recent depends on window size)
                window_x = collected_points_x[-window_size:]
                window_y = collected_points_y[-window_size:]
                current_correlation = np.corrcoef(window_x, window_y)[0, 1] # calculate correlation after addition of new point

                if previous_correlation is not None:
                    delta_correlation = abs(current_correlation - previous_correlation) # calculate change in correlation after addition of new point

                    # if the change is too drastic, do not consider the point
                    if delta_correlation > correlation_drop_threshold:
                        collected_points.pop(-1)
                        collected_points_x.pop(-1)
                        collected_points_y.pop(-1)
                        break # break out of the loop as no more points from this point forward are likely be part of the current LaSO

                previous_correlation = current_correlation # update previous correlation for the next iteration

        if collected_points: # append the collected points as a single LaSO
            LaSOs.append(collected_points.copy())

        for point in collected_points: # remove the collected points from the remaining points as they are now part of a LaSO
            remaining_points.remove(point)

    return LaSOs

if __name__ == "__main__":
    line1 = [(i, 2 * i + 1) for i in range(5)]
    line2 = [(i, -i + 20) for i in range(10, 15)]
    line3 = [(i, 0.5 * i - 5) for i in range(20, 25)]

    all_points = line1 + line2 + line3
    random.shuffle(all_points)

    result = segment_LaSOs(all_points, correlation_drop_threshold=0.1, window_size=3)

    for i, laso in enumerate(result):
        print(f"\nLaSO {i + 1}:")
        for point in laso:
            print(f"  {point}")
