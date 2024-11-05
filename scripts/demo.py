import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
import os

import surface_detector
from unidepth.models import UniDepthV2
from unidepth.utils import colorize

def process_video(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        rgb_torch = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Predict depth
        with torch.no_grad():
            predictions = model.infer(rgb_torch.unsqueeze(0).to(device))

        # Get depth prediction
        depth_pred = predictions["depth"].squeeze().cpu().numpy()

        # Save depth map as .npz
        np.savez_compressed(os.path.join(output_dir, f"{frame_num:06d}.npz"), depth=depth_pred)

    # Release resources
    cap.release()

    print(f"Depth maps saved in {output_dir}")

def debug_video(video_path, output_dir):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare for debug video
    debug_video_path = os.path.join(output_dir, "debug_depth_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(debug_video_path, fourcc, fps, (640, 480))

    for frame_num in tqdm(range(total_frames), desc="debug video"):
        depth_pred = load_depth_map(output_dir, frame_num)

        if depth_pred is not None:
            # Determine the deepest point (max depth) in the depth map for this frame
            vmax = depth_pred.max()

            # Colorize the depth map with dynamic vmax
            depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=vmax, cmap="magma_r")
            debug_frame = cv2.cvtColor(depth_pred_col, cv2.COLOR_RGB2BGR)

            intersection_lines = surface_detector.find_wall_floor_intersections_for_frame(depth_pred)

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

            out.write(debug_frame)
        else:
            print(f"Warning: Depth map for frame {frame_num} is missing or empty.")

    out.release()
    print(f"Debug video saved as {debug_video_path}")

def load_depth_map(output_dir, frame_num):
    depth_file = os.path.join(output_dir, f'{frame_num:06d}.npz')
    if os.path.exists(depth_file):
        with np.load(depth_file) as data:
            # Try to get the first key in the archive
            keys = list(data.keys())
            if keys:
                return data[keys[0]]
            else:
                print(f"Warning: No data found in {depth_file}")
                return None
    else:
        print(f"Warning: Depth file not found: {depth_file}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for depth mapping")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    args = parser.parse_args()

    print("Torch version:", torch.__version__)

    #process_video(args.video_path, args.output_dir)
    debug_video(args.video_path, args.output_dir)
