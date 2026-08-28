import numpy as np
import cv2
from ultralytics import YOLO

def get_masks(image, model):
    results = model.predict(image)
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask = mask.cpu().numpy() #convert to numpy array
                mask = (mask > 0.5).astype(np.uint8) * 255 #convert to binary mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) #resize mask to original image size
    return mask

#optional: augment the leaf image by rotating and scaling -- Needed if the leaf is at a angle that is not desired
def augment_leaf(leaf_image, angle, scale):
    # Get the dimensions of the image
    (h, w) = leaf_image.shape[:2]
    center = (w // 2, h // 2)
    # Create a rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Perform the rotation and scaling
    augmented_leaf = cv2.warpAffine(leaf_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return augmented_leaf

model = YOLO("runs/segment/train/weights/best.pt")#YOLO model path

leaf_A = cv2.imread("Leaves/leaf_0_sub_2.jpg")#image path
leaf_B = cv2.imread("Leaves/leaf_0_sub_3.jpg")#image path

scale = 0.2
leaf_A = cv2.resize(leaf_A, None, fx=scale, fy=scale)
leaf_B = cv2.resize(leaf_B, None, fx=scale, fy=scale)
# Extract leaf

leaf_A_mask = get_masks(leaf_A, model)
leaf_B_mask = get_masks(leaf_B, model)

isolated_leaf_A = cv2.bitwise_and(leaf_A,leaf_A,mask=leaf_A_mask)
leaf_A_rgba = cv2.cvtColor(isolated_leaf_A, cv2.COLOR_BGR2BGRA)
leaf_A_rgba[:, :, 3] = leaf_A_mask

isolated_leaf_B = cv2.bitwise_and(leaf_B,leaf_B,mask=leaf_B_mask)
leaf_B_rgba = cv2.cvtColor(isolated_leaf_B, cv2.COLOR_BGR2BGRA)
leaf_B_rgba[:, :, 3] = leaf_B_mask

#Test the output
cv2.imshow("Leaf A", leaf_A_rgba)
cv2.imshow("Leaf B", leaf_B_rgba)
cv2.waitKey(0)

#set random angle and scale for augmentation
#angle = np.random.uniform(0, 360)
#scale = np.random.uniform(0.7, 1.3)
#Augmented_leaf_B = augment_leaf(leaf_B_rgba, 90, 1)
