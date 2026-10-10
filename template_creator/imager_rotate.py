import cv2

image_path = "./ref_pngs/cfd_rect.bmp"
image = cv2.imread(image_path)
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()