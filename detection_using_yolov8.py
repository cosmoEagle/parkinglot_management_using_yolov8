from ultralytics import YOLO
import cv2
import pickle

# Load YOLOv8 model (use 'yolov8n.pt' for smallest, or 'yolov8s.pt'/'yolov8m.pt' based on speed/accuracy)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("carPark.mp4")

with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

width, height = 107, 48

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    results = model(img, stream=True)

    cars = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 2 or cls == 5:  # 2: car, 5: bus (add more if needed)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cars.append(((x1 + x2) // 2, (y1 + y2) // 2))  # center point

    spaceCounter = 0
    for pos in posList:
        x, y = pos
        rect_center = (x + width // 2, y + height // 2)

        occupied = False
        for cx, cy in cars:
            if x < cx < x + width and y < cy < y + height:
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            spaceCounter += 1

        cv2.rectangle(img, pos, (x + width, y + height), color, 3)

    cv2.putText(img, f"Free: {spaceCounter}/{len(posList)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 3)

    cv2.imshow("YOLOv8 Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
