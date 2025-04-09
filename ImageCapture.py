import cv2

def capture_image():
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try CAP_DSHOW or CAP_VFW

    cv2.namedWindow("FacePay Registration")
    
    captured_frame = None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("FacePay Registration", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            captured_frame = frame.copy()
            print("Frame captured...")
            break

    cam.release()
    cv2.destroyAllWindows()

    return captured_frame