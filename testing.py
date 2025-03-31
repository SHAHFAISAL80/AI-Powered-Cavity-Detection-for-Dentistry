import cv2    
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO(r'cavity detection.pt')

# Path to the video file
video_path = r'9709780-uhd_3840_2160_25fps.mp4'

# Start video capture
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Desired processing size
target_width = 1280  # Set for HD resolution
target_height = 720

# Output video properties
fps = cap.get(cv2.CAP_PROP_FPS)
output_filename = 'output_video.mp4'

# Define the video writer for .mp4 format with H.264 codec
output_video = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Change to 'H264' if needed
    fps,
    (target_width, target_height)
)

# Define class labels
class_labels = ["Cavity", "Normal"]

# Define colors for each class
class_colors = [(255, 255, 255), (0, 255, 0)]  # White for cavity, Bright Neon Green for normal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Resize the frame to HD (1280x720) resolution
    frame = cv2.resize(frame, (target_width, target_height))

    # Perform detection
    try:
        results = model.predict(frame, verbose=False)
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())  # Ensure correct indexing
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors[cls], 4)
                cv2.putText(
                    annotated_frame,
                    class_labels[cls],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    class_colors[cls],
                    2
                )

                # Apply segmentation for cavity (highlighting the region with white overlay)
                if cls == 0:  # Cavity class
                    cavity_region = frame[y1:y2, x1:x2]
                    mask = np.full_like(cavity_region, (255, 255, 255), dtype=np.uint8)  # White overlay
                    annotated_frame[y1:y2, x1:x2] = cv2.addWeighted(cavity_region, 0.5, mask, 0.5, 0)
    
    except Exception as e:
        print(f"Error during model prediction: {e}")
        break

    # Display the frame
    cv2.imshow('Cavity Detection', annotated_frame)

    # Write the frame to the output video
    output_video.write(annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Video processing completed. Output saved as {output_filename}")