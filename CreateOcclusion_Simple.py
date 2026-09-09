import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import random
#----------------------------------------------------------------Extract Leaf-----------------------------------------------------------------
def get_masks(image):
    model = YOLO("runs/segment/train/weights/best.pt")#YOLO model path
    results = model.predict(image)
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask = mask.cpu().numpy() #convert to numpy array
                mask = (mask > 0.5).astype(np.uint8) * 255 #convert to binary mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) #resize mask to original image size
                return mask
    return None

#----------------------------------------------------------------Augment Leaf-----------------------------------------------------------------
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

#overlap the two masks to create occlusion
def create_overlap(image_a, image_b, mask_a, mask_b, desired_overlap, step=5):

    xa, ya, wa, ha = cv2.boundingRect(mask_a)
    xb, yb, wb, hb = cv2.boundingRect(mask_b)

    leaf_b = image_b[yb:yb+hb, xb:xb+wb]
    leaf_b_mask = mask_b[yb:yb+hb, xb:xb+wb]

    area_b = np.count_nonzero(leaf_b_mask)

    best_difference = float("inf")
    best_dx = None
    best_dy = None
    best_overlap = 0

    for dy in range(-ha, ha, step):
        for dx in range(-wa, wa, step):

            new_x = xa + dx
            new_y = ya + dy

            if new_x < 0 or new_y < 0:
                continue
            if new_x + wb > image_a.shape[1]:
                continue
            if new_y + hb > image_a.shape[0]:
                continue

            region_a = mask_a[new_y:new_y+hb, new_x:new_x+wb]
            intersection = np.count_nonzero(region_a & leaf_b_mask)
            overlap = intersection / area_b
            difference = abs(overlap - desired_overlap)

            if difference < best_difference:
                best_difference = difference
                best_dx = dx
                best_dy = dy
                best_overlap = overlap

    if best_dx is None:
        raise ValueError("No valid placement found for leaf_b within leaf_a's search range")

    output = image_a.copy()

    new_x = xa + best_dx
    new_y = ya + best_dy

    # Clip to output's actual bounds as a safety net
    h_out, w_out = output.shape[:2]
    y0, x0 = max(new_y, 0), max(new_x, 0)
    y1, x1 = min(new_y + hb, h_out), min(new_x + wb, w_out)

    by0, bx0 = y0 - new_y, x0 - new_x
    by1, bx1 = by0 + (y1 - y0), bx0 + (x1 - x0)

    region = output[y0:y1, x0:x1]
    mask_crop = leaf_b_mask[by0:by1, bx0:bx1]
    leaf_b_crop = leaf_b[by0:by1, bx0:bx1]

    region[mask_crop > 0] = leaf_b_crop[mask_crop > 0]

    return output, best_overlap
#---------------------------------------------------------------Load Model and Images-----------------------------------------------------------------
model = YOLO("runs/segment/train/weights/best.pt")#YOLO model path

leaf_A = cv2.imread("Leaves/leaf_0_sub_2.jpg")#image path
leaf_B = cv2.imread("Leaves/leaf_0_sub_3.jpg")#image path

#resize the images to a smaller size for viewing purposes and processing time.
scale = 0.2
leaf_A = cv2.resize(leaf_A, None, fx=scale, fy=scale)
leaf_B = cv2.resize(leaf_B, None, fx=scale, fy=scale)

# Extract leaf masks
leaf_A_mask = get_masks(leaf_A)
leaf_B_mask = get_masks(leaf_B)

#isolate the leaves from the background using the masks and convert them to RGBA format
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

#-----------------------------------------------------------------Create Occlusion-----------------------------------------------------------------
result, overlap = create_overlap(
    leaf_A,
    leaf_B,
    leaf_A_mask,
    leaf_B_mask,
    desired_overlap=0.25,
    step=5 #time saving step
)

print(f"Requested overlap: 25%")
print(f"Actual overlap: {overlap * 100:.2f}%")

cv2.imshow("Result", result)
cv2.imwrite("occlusion_result25.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(10):
    Leaf_A = random.choice(os.listdir("Single_Leaves"))
    print(Leaf_A)
    for file in os.listdir("Single_Leaves"):
        Leaf_B = file
        print(Leaf_B)