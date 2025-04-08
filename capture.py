import cv2

def capture_image(path):

    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try CAP_DSHOW or CAP_VFW

    cv2.namedWindow("Capture Window")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Capture Window", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(path, frame)
            print("{} written!".format(path))
            break

    cam.release()

    cv2.destroyAllWindows()