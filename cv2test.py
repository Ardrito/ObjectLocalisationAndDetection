import cv2
import live_model as decoderModel
from PIL import Image
# import torch

# device = "cuda"

# model = decoderModel.decoder().to(device)
# checkpoint = torch.load("checkpoints_4/best_model.pt", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

# Try opening the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Draw a test message
    cv2.putText(frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("OpenCV Test Window", frame)

    print ("done")

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("✅ Camera released and window closed.")