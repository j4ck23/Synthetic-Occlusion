import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
#----------------------------------------------------------------Extract Leaf-----------------------------------------------------------------
def get_masks(image, model):
    results = model.predict(image)
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask = mask.cpu().numpy() #convert to numpy array
                mask = (mask > 0.5).astype(np.uint8) * 255 #convert to binary mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) #resize mask to original image size
    return mask

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
def create_overlap(mask_a, mask_b, desired_overlap, coarse_step=10, fine_radius=10):

    # Convert masks to binary
    mask_a = (mask_a > 0).astype(np.uint8)
    mask_b = (mask_b > 0).astype(np.uint8)

    if not 0 <= desired_overlap <= 1:
        raise ValueError("desired_overlap must be between 0 and 1")

    # Get bounding boxes
    x_a, y_a, w_a, h_a = cv2.boundingRect(mask_a)
    x_b, y_b, w_b, h_b = cv2.boundingRect(mask_b)

    # Crop leaves
    leaf_a = mask_a[y_a:y_a+h_a, x_a:x_a+w_a]
    leaf_b = mask_b[y_b:y_b+h_b, x_b:x_b+w_b]

    # Area of leaf B
    area_b = np.count_nonzero(leaf_b)

    # --------------------------------------------------
    # Function to calculate overlap at a given position
    # --------------------------------------------------
def create_overlap(mask_a, mask_b, desired_overlap,
                   coarse_step=10,
                   fine_radius=10):

    mask_a = (mask_a > 0).astype(np.uint8)
    mask_b = (mask_b > 0).astype(np.uint8)

    if not 0 <= desired_overlap <= 1:
        raise ValueError("desired_overlap must be between 0 and 1")

    # Bounding boxes
    x_a, y_a, w_a, h_a = cv2.boundingRect(mask_a)
    x_b, y_b, w_b, h_b = cv2.boundingRect(mask_b)

    # Crop leaves
    leaf_a = mask_a[y_a:y_a+h_a, x_a:x_a+w_a]
    leaf_b = mask_b[y_b:y_b+h_b, x_b:x_b+w_b]

    area_b = np.count_nonzero(leaf_b)

    def calculate_overlap(dx, dy):

        a_y1 = max(0, dy)
        a_y2 = min(h_a, dy + h_b)

        if a_y1 >= a_y2:
            return 0

        b_y1 = max(0, -dy)
        b_y2 = b_y1 + (a_y2 - a_y1)

        a_x1 = max(0, dx)
        a_x2 = min(w_a, dx + w_b)

        if a_x1 >= a_x2:
            return 0

        b_x1 = max(0, -dx)
        b_x2 = b_x1 + (a_x2 - a_x1)

        region_a = leaf_a[a_y1:a_y2, a_x1:a_x2]
        region_b = leaf_b[b_y1:b_y2, b_x1:b_x2]

        intersection = np.count_nonzero(region_a & region_b)

        return intersection / area_b

    # -------------------------
    # Coarse search
    # -------------------------

    best_difference = float("inf")
    best_dx = 0
    best_dy = 0
    best_overlap = 0

    for dy in range(-h_b, h_a + 1, coarse_step):

        for dx in range(-w_b, w_a + 1, coarse_step):

            overlap = calculate_overlap(dx, dy)

            difference = abs(overlap - desired_overlap)

            if difference < best_difference:
                best_difference = difference
                best_dx = dx
                best_dy = dy
                best_overlap = overlap

    # -------------------------
    # Fine search
    # -------------------------

    coarse_dx = best_dx
    coarse_dy = best_dy

    best_difference = float("inf")

    for dy in range(
        coarse_dy - fine_radius,
        coarse_dy + fine_radius + 1
    ):

        for dx in range(
            coarse_dx - fine_radius,
            coarse_dx + fine_radius + 1
        ):

            overlap = calculate_overlap(dx, dy)

            difference = abs(overlap - desired_overlap)

            if difference < best_difference:
                best_difference = difference
                best_dx = dx
                best_dy = dy
                best_overlap = overlap

    # -------------------------
    # Create final masks
    # -------------------------

    padding_x = w_a + w_b
    padding_y = h_a + h_b

    canvas_w = w_a + w_b + 2 * padding_x
    canvas_h = h_a + h_b + 2 * padding_y

    positioned_a = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.uint8
    )

    positioned_b = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.uint8
    )

    centre_x = padding_x
    centre_y = padding_y

    # Leaf A
    positioned_a[
        centre_y:centre_y+h_a,
        centre_x:centre_x+w_a
    ] = leaf_a

    # Leaf B
    bx = centre_x + best_dx
    by = centre_y + best_dy

    positioned_b[
        by:by+h_b,
        bx:bx+w_b
    ] = leaf_b

    return positioned_a, positioned_b, best_overlap, best_dx, best_dy
#---------------------------------------------------------------Load Model and Images-----------------------------------------------------------------
model = YOLO("runs/segment/train/weights/best.pt")#YOLO model path

leaf_A = cv2.imread("Leaves/leaf_0_sub_2.jpg")#image path
leaf_B = cv2.imread("Leaves/leaf_0_sub_3.jpg")#image path

scale = 0.2
leaf_A = cv2.resize(leaf_A, None, fx=scale, fy=scale)
leaf_B = cv2.resize(leaf_B, None, fx=scale, fy=scale)

# Extract leaf masks
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

#-----------------------------------------------------------------Create Occlusion-----------------------------------------------------------------
positioned_a, positioned_b, overlap, dx, dy = create_overlap(leaf_A_mask, leaf_B_mask, 0.25)

print("Overlap:", overlap)
cv2.imshow("A", positioned_a * 255)
cv2.imshow("B", positioned_b * 255)

cv2.waitKey(0)
cv2.destroyAllWindows()

# A bounding box
x_a, y_a, w_a, h_a = cv2.boundingRect(leaf_A_mask)

# B bounding box
x_b, y_b, w_b, h_b = cv2.boundingRect(leaf_B_mask)

# New position for B
new_x = x_a + dx
new_y = y_a + dy

leaf_b_image = leaf_B[new_y:new_y+h_b, new_x:new_x+w_b]
leaf_b_mask = leaf_B_mask[new_y:new_y+h_b, new_x:new_x+w_b]

synthetic = leaf_A.copy()

# Region where B will go
region = synthetic[new_y:new_y+h_b,new_x:new_x+w_b]

# Paste B pixels
region[leaf_b_mask > 0] = leaf_b_image[leaf_b_mask > 0]

# Put region back
synthetic[new_y:new_y+h_b,new_x:new_x+w_b] = region

cv2.imshow("Synthetic", synthetic)
cv2.waitKey(0)
cv2.destroyAllWindows()