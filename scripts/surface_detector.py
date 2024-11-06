import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def find_wall_floor_intersections_for_frame(depth_map, debug_frame):
    """Find wall-floor and wall-wall intersections using vertical stripe analysis"""
    if depth_map is None:
        return []

    # Get the deepest point in the entire depth map for reference
    max_depth_in_frame = np.max(depth_map[depth_map > 0])
    depth_threshold = max_depth_in_frame * 0.5  # 50% of max depth

    # Parameters for analysis
    min_points = 10
    initial_points = 10
    stripe_width = 5

    # Parameters for wall slope analysis
    edge_width = 50  # Width of edge region to analyze

    # Set up matplotlib for interactive plotting
    plt.ion()  # Turn on interactive mode

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

        final_wall_depth = np.mean(wall_depths) if len(wall_depths) >= min_points else None
        
        # Filter out shallow walls (likely foreground objects/people)
        if final_wall_depth is not None and final_wall_depth < depth_threshold:
            return None
            
        return final_wall_depth

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

    def detect_intersection_with_obstacle(depths, y_coords, wall_depth):
        """Analyze floor using a monotonic spline fit and extrapolate to find wall intersection point."""
        if len(depths) < 30:
            return None, None, 0.0  # Added confidence score

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
            return None, None, 0.0

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
                if slope < last_slope or slope > 0 or abs(slope - last_slope) > 20:
                    if len(current_sequence) >= 3:  # Only keep sequences with at least 3 groups
                        sequences.append(current_sequence.copy())
                    current_sequence = []  # Start a new sequence

            current_sequence.append((slope, idx))

        # Don't forget to check the last sequence
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)

        if not sequences:  # No valid sequences found
            return None, None, 0.0

        # Find the longest sequence
        longest_sequence = max(sequences, key=len)

        if len(longest_sequence) < 3:  # Double check we have enough groups
            return None, None, 0.0

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
            return None, None, 0.0

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

        # Calculate confidence score based on extrapolation distance
        max_depth_in_sequence = np.max(sequence_depths)
        extrapolation_distance = abs(wall_depth - max_depth_in_sequence)
        
        # Normalize confidence score (0 to 1)
        # Higher score when extrapolation distance is smaller
        confidence = 1.0 - min(1.0, extrapolation_distance / wall_depth)
        
        # Apply additional weight if the deepest point is close to wall depth
        if max_depth_in_sequence >= 0.8 * wall_depth:  # Within 20% of wall depth
            confidence *= 2.0  # Double the confidence

        try:
            intersection_y = int(fitted_func(wall_depth))
            if np.isnan(intersection_y) or 0 > intersection_y or intersection_y > 480:
                intersection_y = None
        except Exception:
            intersection_y = None

        return fitted_func, intersection_y, confidence

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
            # Convert slope to cm/pixel (assuming depths are in cm)
            return slope
        except:
            return None

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
            # Check if left slope is positive (getting deeper as x increases)
            # and right slope is negative (getting deeper as x decreases)
            # and at least one slope has magnitude >= 1 cm/pixel
            if (left_slope > 0 and right_slope < 0 and 
                (abs(left_slope) >= 0.01 or abs(right_slope) >= 0.01)):
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
    floor_wall_points = []
    point_confidences = []  # Store confidence scores

    for x in range(0, depth_map.shape[1], stripe_width):
        depths = depth_map[:, x]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]
        valid_y = np.arange(depth_map.shape[0])[valid_mask]

        if len(valid_depths) < depth_map.shape[0] * 0.3:
            continue

        wall_depth = analyze_wall_depth(valid_depths)

        if wall_depth is not None:
            _, intersect_y, confidence = detect_intersection_with_obstacle(valid_depths, valid_y, wall_depth)

            if intersect_y is not None:
                frame_x = x
                frame_y = intersect_y
                floor_wall_points.append((frame_x, frame_y))
                point_confidences.append(confidence)

    def on_mouse_click(event, x, y, flags, param, current_depth_map):
        if event == cv2.EVENT_LBUTTONDOWN:
            stripe_x = x
            depths = current_depth_map[:, stripe_x]  # Use stored depth map
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_y = np.arange(480)[valid_mask]

            if len(valid_depths) >= 480 * 0.3:
                wall_depth = analyze_wall_depth(valid_depths)
                floor_func, intersect_y, confidence = detect_intersection_with_obstacle(valid_depths, valid_y, wall_depth)
                plot_stripe_analysis(stripe_x, valid_depths, valid_y, wall_depth, floor_func, intersect_y, confidence)

    # Storage for debug visualization
    debug_points = {
        'wall_depths': [],  # (x, depth) pairs
        'floor_points': [],  # (x, y) pairs
        'corner_candidates': [],  # x coordinates
        'floor_wall_points': []  # (x, y) pairs
    }

    # Convert points to lines
    # Process floor-wall points with confidence weighting
    intersection_lines = []
    if corner_x is not None:
        intersection_lines.append(("wall", ((corner_x, 0), (corner_x, 480))))

    if len(floor_wall_points) >= min_points:
        points = np.array(floor_wall_points)
        confidences = np.array(point_confidences)

        # Apply confidence-based filtering
        confidence_threshold = np.mean(confidences) * 0.5
        high_confidence_mask = confidences >= confidence_threshold
        points = points[high_confidence_mask]

        if len(points) >= min_points:
            if corner_x is not None:
                # Split points into left and right of corner
                left_points = points[points[:, 0] < corner_x]
                right_points = points[points[:, 0] >= corner_x]

                # Create single line for each wall if enough points
                if len(left_points) >= min_points:
                    # Fit line to left points
                    left_x = left_points[:, 0]
                    left_y = left_points[:, 1]
                    left_coeffs = np.polyfit(left_x, left_y, 1)
                    
                    # Get line endpoints
                    x_start = np.min(left_x)
                    x_end = np.max(left_x)
                    y_start = np.polyval(left_coeffs, x_start)
                    y_end = np.polyval(left_coeffs, x_end)
                    
                    intersection_lines.append(("floor", ((x_start, y_start), (x_end, y_end))))

                if len(right_points) >= min_points:
                    # Fit line to right points
                    right_x = right_points[:, 0]
                    right_y = right_points[:, 1]
                    right_coeffs = np.polyfit(right_x, right_y, 1)
                    
                    # Get line endpoints
                    x_start = np.min(right_x)
                    x_end = np.max(right_x)
                    y_start = np.polyval(right_coeffs, x_start)
                    y_end = np.polyval(right_coeffs, x_end)
                    
                    intersection_lines.append(("floor", ((x_start, y_start), (x_end, y_end))))
            else:
                # Single back wall case - cluster points and fit a single line
                eps = 50
                clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
                labels = clustering.labels_
                
                # Combine all valid clusters
                valid_points = points[labels >= 0]
                
                if len(valid_points) >= min_points:
                    # Fit single line to all valid points
                    x = valid_points[:, 0]
                    y = valid_points[:, 1]
                    coeffs = np.polyfit(x, y, 1)
                    
                    # Get line endpoints
                    x_start = np.min(x)
                    x_end = np.max(x)
                    y_start = np.polyval(coeffs, x_start)
                    y_end = np.polyval(coeffs, x_end)
                    
                    intersection_lines.append(("floor", ((x_start, y_start), (x_end, y_end))))

    # Process vertical stripes
    for x in range(0, depth_map.shape[1], stripe_width):
        depths = depth_map[:, x]
        valid_mask = depths > 0
        valid_depths = depths[valid_mask]
        valid_y = np.arange(depth_map.shape[0])[valid_mask]

        if len(valid_depths) < depth_map.shape[0] * 0.3:
            continue

        # Find wall depth
        wall_depth = analyze_wall_depth(valid_depths)
        if wall_depth is not None:
            debug_points['wall_depths'].append((x, wall_depth))

        # Find floor curve
        floor_func, intersect_y, confidence = detect_intersection_with_obstacle(valid_depths, valid_y, wall_depth)
        if floor_func is not None:
            # Store floor points for visualization
            test_depths = np.linspace(valid_depths.min(), valid_depths.max(), 20)
            floor_y_values = floor_func(test_depths)
            debug_points['floor_points'].extend(zip([x] * len(test_depths), floor_y_values))

        if intersect_y is not None:
            debug_points['floor_wall_points'].append((x, intersect_y))

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

    # Draw debug visualization
    # Draw wall depths
    for x, depth in debug_points['wall_depths']:
        try:
            cv2.circle(debug_frame, (int(x), int(depth)), 2, (0, 255, 0), -1)  # Green
        except:
            continue

    # Draw floor points
    for x, y in debug_points['floor_points']:
        try:
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(debug_frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue
        except:
            continue

    # Draw corner candidates in red
    for x, y in corner_candidates:
        try:
            cv2.circle(debug_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
        except:
            continue

    # Draw floor-wall intersection points in cyan
    for x, y in debug_points['floor_wall_points']:
        try:
            cv2.circle(debug_frame, (int(x), int(y)), 4, (255, 255, 0), -1)  # Cyan (255, 255, 0)
        except:
            continue

    # Draw final intersection lines
    for line_type, (p1, p2) in intersection_lines:
        try:
            p1_depth = (int(p1[0]), int(p1[1]))
            p2_depth = (int(p2[0]), int(p2[1]))
            color = (0, 255, 255) if line_type == "wall" else (255, 255, 0)  # Cyan for wall, Yellow for floor
            cv2.line(debug_frame, p1_depth, p2_depth, color, 2)
        except:
            continue

    # Add text labels for clarity
    cv2.putText(debug_frame, "Wall depths (green)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_frame, "Floor points (blue)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(debug_frame, "Corner candidates (red)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(debug_frame, "Floor-wall intersections (cyan)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 1)

    # Show debug visualization
    cv2.imshow('Depth Analysis Debug', debug_frame)
    cv2.waitKey(1)

    # Create window and set mouse callback
    cv2.namedWindow('Depth Analysis Debug')
    cv2.setMouseCallback('Depth Analysis Debug',
                         lambda event, x, y, flags, param: on_mouse_click(event, x, y, flags, param, depth_map))

    return intersection_lines

def plot_stripe_analysis(x, depths, y_coords, wall_depth, floor_func, intersect_y, confidence):
    """Plot depth analysis for a single vertical stripe"""

    stripe_fig = plt.figure(figsize=(8, 6))
    stripe_ax = stripe_fig.add_subplot(111)

    # Plot actual depth values
    stripe_ax.scatter(depths, y_coords, c='blue', s=10, label='Depth measurements')

    if wall_depth is not None:
        stripe_ax.axvline(x=wall_depth, color='red', linestyle='--', label=f'Wall depth: {wall_depth:.2f}')

    if floor_func is not None:
        # Plot quadratic fit
        test_depths = np.linspace(np.min(depths), np.max(depths), 100)
        fit_y = floor_func(test_depths)
        stripe_ax.plot(test_depths, fit_y, 'g-', label='Floor fit')

    # Plot intersection point
    if wall_depth is not None and intersect_y is not None:
        stripe_ax.plot(wall_depth, intersect_y, 'cyan', marker='o', markersize=10,
                       label=f'Intersection: ({wall_depth:.2f}, {intersect_y:.2f})')


    stripe_ax.set_title(f'Depth Analysis for Vertical Stripe at x={x}')
    stripe_ax.set_xlabel('Depth')
    stripe_ax.set_ylabel('Y coordinate')
    stripe_ax.invert_yaxis()
    stripe_ax.grid(True)
    stripe_ax.legend()

    stripe_fig.tight_layout()
    stripe_fig.canvas.draw()
    stripe_fig.canvas.flush_events()

