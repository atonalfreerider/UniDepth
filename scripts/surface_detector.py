import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import PchipInterpolator

def find_wall_floor_intersections_for_frame(depth_map):
    """Find wall-floor and wall-wall intersections using vertical stripe analysis"""
    if depth_map is None:
        return []

    # Parameters for analysis
    min_points = 10
    initial_points = 10
    stripe_width = 5

    # Parameters for wall slope analysis
    edge_width = 50  # Width of edge region to analyze
    slope_difference_threshold = 0.3  # Minimum difference in slopes to indicate a corner

    def analyze_wall_depth(depths):
        """Analyze wall depth using first N points from top"""
        if len(depths) < initial_points:
            return None

        # Take first N points and their average depth
        wall_depths = depths[:initial_points]
        mean_depth = np.mean(wall_depths)
        std_depth = np.std(wall_depths)

        # Continue adding points while they're within threshold
        for i in range(initial_points, len(depths)):
            if abs(depths[i] - mean_depth) > (3 * std_depth):  # Using 3-sigma rule
                break
            wall_depths = np.append(wall_depths, depths[i])

        return np.mean(wall_depths) if len(wall_depths) >= min_points else None

    def analyze_corner_point(depths, max_depth_idx, window_size=5):
        """Verify corner point by checking depth consistency of adjacent points"""
        if max_depth_idx < window_size or max_depth_idx >= len(depths) - window_size:
            return False

        corner_depth = depths[max_depth_idx]

        # Check left side consistency
        left_depths = depths[max_depth_idx - window_size:max_depth_idx]
        left_diff = np.abs(left_depths - corner_depth) / corner_depth
        if not np.all(left_diff < 0.15):  # 15% threshold
            return False

        # Check right side consistency
        right_depths = depths[max_depth_idx + 1:max_depth_idx + window_size + 1]
        right_diff = np.abs(right_depths - corner_depth) / corner_depth
        if not np.all(right_diff < 0.15):  # 15% threshold
            return False

        return True

    def detect_intersection_open_floor_to_wall(depths, y_coords):
        """Detect the most extreme discontinuity as the wall intersection for open-floor scenarios."""
        # Define y range for detecting discontinuity
        valid_indices = [i for i, y in enumerate(y_coords) if 380 >= y >= 100]
        filtered_depths = np.array([depths[i] for i in valid_indices])
        filtered_y_coords = np.array([y_coords[i] for i in valid_indices])

        # Calculate absolute differences in depth to detect the largest discontinuity
        depth_diffs = np.abs(np.diff(filtered_depths))
        max_discontinuity_idx = np.argmax(depth_diffs)

        # Get the y-coordinate and depth at the discontinuity point
        discontinuity_y = int(filtered_y_coords[max_discontinuity_idx])
        return discontinuity_y if 380 >= discontinuity_y >= 100 else None

    def detect_intersection_with_obstacle(depths, y_coords, wall_depth, slope_jump_threshold=20):
        """Analyze floor using a monotonic spline fit and extrapolate to find wall intersection point."""
        if len(depths) < 30:  # Ensure sufficient data points
            return None, None

        # Reverse arrays to work from bottom up
        depths = depths[::-1]
        y_coords = y_coords[::-1]

        # Calculate average slopes for groups of 10 points
        group_size = 10
        slopes = []

        for i in range(0, len(depths) - group_size, group_size):
            group1_depths = depths[i:i + group_size]
            group1_y = y_coords[i:i + group_size]

            # Calculate average slope for this group using linear regression
            try:
                slope, _ = np.polyfit(group1_depths, group1_y, 1)
                slopes.append((slope, i))
            except:
                continue

        if len(slopes) < 3:  # Need at least 3 groups
            return None, None

        # Find all sequences of decreasing slopes with additional checks
        sequences = []
        current_sequence = []

        for i in range(len(slopes)):
            slope, idx = slopes[i]

            if current_sequence:
                last_slope, last_idx = current_sequence[-1]
                # Stop the sequence if:
                # 1. The current slope is more negative than the last slope (indicating a reverse in direction).
                # 2. The current slope is positive.
                # 3. The jump between slopes exceeds the threshold.
                if slope < last_slope or slope > 0 or abs(slope - last_slope) > slope_jump_threshold:
                    if len(current_sequence) >= 3:  # Only keep sequences with at least 3 groups
                        sequences.append(current_sequence.copy())
                    current_sequence = []  # Start a new sequence

            current_sequence.append((slope, idx))

        # Don't forget to check the last sequence
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)

        if not sequences:  # No valid sequences found
            return None, None

        # Find the longest sequence
        longest_sequence = max(sequences, key=len)

        if len(longest_sequence) < 3:  # Double check we have enough groups
            return None, None

        # Get the range of points covered by the longest sequence
        start_idx = longest_sequence[0][1]
        end_idx = longest_sequence[-1][1] + group_size

        # Select the points within this range for fitting
        sequence_depths = np.array(depths[start_idx:end_idx])
        sequence_y = np.array(y_coords[start_idx:end_idx])

        """Extrapolate spline behind obstacles to find the wall intersection."""
        # Apply monotonic spline fit
        try:
            spline = PchipInterpolator(sequence_depths, sequence_y, extrapolate=False)
        except ValueError:
            return None, None

        # Define the fitted function with extrapolation support for scalar and array inputs
        def fitted_func(x):
            x = np.asarray(x)  # Convert x to a numpy array to handle both scalars and arrays
            within_range = x <= sequence_depths[-1]

            # Use the spline within the range, and linear extrapolation outside the range
            result = np.where(
                within_range,
                spline(x),  # Spline interpolation within the range
                sequence_y[-1] + (x - sequence_depths[-1]) * (
                        (sequence_y[-1] - sequence_y[-2]) / (sequence_depths[-1] - sequence_depths[-2]))
                # Linear extrapolation
            )
            return result

        try:
            intersection_y = int(fitted_func(wall_depth))
            if np.isnan(intersection_y) or 0 > intersection_y or intersection_y > 480:
                intersection_y = None
        except Exception:
            intersection_y = None

        return fitted_func, intersection_y

    def analyze_edge_slope(depths, x_coords, from_left=True):
        """Analyze wall slope at frame edges"""
        if len(depths) < initial_points:
            return None

        # Take points from edge region
        if from_left:
            edge_depths = depths[:edge_width]
            edge_x = x_coords[:edge_width]
        else:
            edge_depths = depths[-edge_width:]
            edge_x = x_coords[-edge_width:]

        # Fit line to edge points
        try:
            slope, _ = np.polyfit(edge_x, edge_depths, 1)
            return slope
        except:
            return None

    # Storage for intersection points
    floor_wall_points = []

    # First, analyze wall slopes at edges of frame
    has_corner = False
    for y in range(0, depth_map.shape[0] // 2, stripe_width):
        depths = depth_map[y, :]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]
        valid_x = np.arange(depth_map.shape[1])[valid_mask]

        if len(valid_depths) < depth_map.shape[1] * 0.3:
            continue

        left_slope = analyze_edge_slope(valid_depths, valid_x, from_left=True)
        right_slope = analyze_edge_slope(valid_depths, valid_x, from_left=False)

        if left_slope is not None and right_slope is not None:
            # Check if slopes indicate a corner
            if left_slope * right_slope < 0:  # Slopes in opposite directions
                has_corner = True
                break
            elif abs(left_slope - right_slope) > slope_difference_threshold:
                has_corner = True
                break

    # Only look for corner if slopes indicate one exists
    corner_x = None
    if has_corner:
        corner_candidates = []
        for y in range(0, depth_map.shape[0] // 2, stripe_width):
            depths = depth_map[y, :]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_x = np.arange(depth_map.shape[1])[valid_mask]

            if len(valid_depths) < depth_map.shape[1] * 0.3:
                continue

            # Find the deepest point (corner candidate)
            max_depth_idx = np.argmax(valid_depths)
            if initial_points < max_depth_idx < len(valid_depths) - initial_points:
                # Verify it's a local maximum
                if (valid_depths[max_depth_idx] > valid_depths[
                                                  max_depth_idx - initial_points:max_depth_idx]).all() and \
                        (valid_depths[max_depth_idx] > valid_depths[
                                                       max_depth_idx + 1:max_depth_idx + initial_points + 1]).all():
                    corner_x = valid_x[max_depth_idx]
                    corner_candidates.append(corner_x)

        # Determine if we have a consistent corner
        if len(corner_candidates) >= min_points:
            median_x = np.median(corner_candidates)
            deviations = np.abs(np.array(corner_candidates) - median_x)
            if np.mean(deviations) < 50:
                corner_x = median_x

    # Process vertical stripes for floor-wall intersections
    for x in range(0, depth_map.shape[1], stripe_width):
        depths = depth_map[:, x]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]
        valid_y = np.arange(depth_map.shape[0])[valid_mask]

        if len(valid_depths) < depth_map.shape[0] * 0.3:
            continue

        # Find wall depth starting from top of frame
        wall_depth = analyze_wall_depth(valid_depths)

        if wall_depth is not None:
            # Find floor curve starting from bottom of frame
            _, intersect_y = detect_intersection_with_obstacle(valid_depths, valid_y, wall_depth)

            if intersect_y is not None:
                frame_x = x
                frame_y = intersect_y
                floor_wall_points.append((frame_x, frame_y))

    # Convert points to lines
    intersection_lines = []

    # Add wall-wall intersection line if corner was found
    if corner_x is not None:
        intersection_lines.append(("wall", ((corner_x, 0), (corner_x, 480))))

    # Process floor-wall points
    if len(floor_wall_points) >= min_points:
        points = np.array(floor_wall_points)

        # Adjust clustering based on whether we found a corner
        eps = 50 if corner_x is None else 30
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        unique_labels = np.unique(clustering.labels_[clustering.labels_ >= 0])

        if corner_x is not None:
            # Split points into left and right of corner
            left_points = points[points[:, 0] < corner_x]
            right_points = points[points[:, 0] >= corner_x]

            # Create lines for each side if enough points
            if len(left_points) >= min_points:
                sorted_points = left_points[left_points[:, 0].argsort()]
                p1 = tuple(sorted_points[0])
                p2 = tuple(sorted_points[-1])
                intersection_lines.append(("floor", (p1, p2)))

            if len(right_points) >= min_points:
                sorted_points = right_points[right_points[:, 0].argsort()]
                p1 = tuple(sorted_points[0])
                p2 = tuple(sorted_points[-1])
                intersection_lines.append(("floor", (p1, p2)))
        else:
            # Single wall case - one floor line
            for label in unique_labels:
                cluster_points = points[clustering.labels_ == label]
                if len(cluster_points) >= min_points:
                    sorted_points = cluster_points[cluster_points[:, 0].argsort()]
                    p1 = tuple(sorted_points[0])
                    p2 = tuple(sorted_points[-1])
                    intersection_lines.append(("floor", (p1, p2)))

    # Process vertical stripes
    for x in range(0, depth_map.shape[1], stripe_width):
        depths = depth_map[:, x]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]

        if len(valid_depths) < depth_map.shape[0] * 0.3:
            continue

    # Process horizontal stripes for corner detection
    corner_candidates = []
    for y in range(0, depth_map.shape[0] // 2, stripe_width):
        depths = depth_map[y, :]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]
        valid_x = np.arange(depth_map.shape[1])[valid_mask]

        if len(valid_depths) < depth_map.shape[1] * 0.3:
            continue

        # Find the deepest point
        max_depth_idx = np.argmax(valid_depths)

        # Verify corner point consistency
        if analyze_corner_point(valid_depths, max_depth_idx):
            corner_candidates.append((valid_x[max_depth_idx], y))

    return intersection_lines
