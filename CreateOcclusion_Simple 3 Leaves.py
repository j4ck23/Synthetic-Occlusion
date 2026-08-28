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
def create_overlap_3(
    image_a,
    image_b,
    image_c,
    mask_a,
    mask_b,
    mask_c,
    desired_overlap,
    step=5
):

    # Convert masks to binary
    mask_a = (mask_a > 0).astype(np.uint8)
    mask_b = (mask_b > 0).astype(np.uint8)
    mask_c = (mask_c > 0).astype(np.uint8)

    # -----------------------------------------
    # Leaf A bounding box
    # -----------------------------------------

    xa, ya, wa, ha = cv2.boundingRect(mask_a)

    # -----------------------------------------
    # Leaf B bounding box
    # -----------------------------------------

    xb, yb, wb, hb = cv2.boundingRect(mask_b)

    leaf_b = image_b[yb:yb+hb, xb:xb+wb]
    leaf_b_mask = mask_b[yb:yb+hb, xb:xb+wb]

    area_b = np.count_nonzero(leaf_b_mask)

    # -----------------------------------------
    # Find position for B
    # -----------------------------------------

    best_difference = float("inf")
    best_dx = 0
    best_dy = 0

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

            region_a = mask_a[
                new_y:new_y+hb,
                new_x:new_x+wb
            ]

            intersection = np.count_nonzero(
                region_a & leaf_b_mask
            )

            overlap = intersection / area_b

            difference = abs(
                overlap - desired_overlap
            )

            if difference < best_difference:

                best_difference = difference
                best_dx = dx
                best_dy = dy

    # -----------------------------------------
    # Put B onto A
    # -----------------------------------------

    output = image_a.copy()

    new_x = xa + best_dx
    new_y = ya + best_dy

    region = output[
        new_y:new_y+hb,
        new_x:new_x+wb
    ]

    region[leaf_b_mask > 0] = leaf_b[
        leaf_b_mask > 0
    ]

    # -----------------------------------------
    # Leaf C
    # -----------------------------------------

    xc, yc, wc, hc = cv2.boundingRect(mask_c)

    leaf_c = image_c[yc:yc+hc, xc:xc+wc]
    leaf_c_mask = mask_c[yc:yc+hc, xc:xc+wc]

    area_c = np.count_nonzero(leaf_c_mask)

    # -----------------------------------------
    # Find position for C
    # -----------------------------------------

    best_difference = float("inf")
    best_cx = 0
    best_cy = 0

    # Use the current image/masks as the target
    combined_mask = np.zeros_like(mask_a)

    combined_mask[mask_a > 0] = 1

    # Add B at its new position
    combined_mask[
        new_y:new_y+hb,
        new_x:new_x+wb
    ][leaf_b_mask > 0] = 1

    # Search around A
    for dy in range(-ha, ha, step):

        for dx in range(-wa, wa, step):

            new_cx = xa + dx
            new_cy = ya + dy

            if new_cx < 0 or new_cy < 0:
                continue

            if new_cx + wc > image_a.shape[1]:
                continue

            if new_cy + hc > image_a.shape[0]:
                continue

            region_existing = combined_mask[
                new_cy:new_cy+hc,
                new_cx:new_cx+wc
            ]

            intersection = np.count_nonzero(
                region_existing & leaf_c_mask
            )

            overlap = intersection / area_c

            difference = abs(
                overlap - desired_overlap
            )

            if difference < best_difference:

                best_difference = difference
                best_cx = dx
                best_cy = dy

    # -----------------------------------------
    # Put C onto image
    # -----------------------------------------

    final_x = xa + best_cx
    final_y = ya + best_cy

    region = output[
        final_y:final_y+hc,
        final_x:final_x+wc
    ]

    region[leaf_c_mask > 0] = leaf_c[
        leaf_c_mask > 0
    ]

    return output
#---------------------------------------------------------------Load Model and Images-----------------------------------------------------------------
model = YOLO("runs/segment/train/weights/best.pt")#YOLO model path

leaf_A = cv2.imread("Leaves/leaf_32.jpg")#image path
leaf_B = cv2.imread("Leaves/leaf_32_sub_2.jpg")#image path
leaf_C = cv2.imread("Leaves/leaf_32_sub_3.jpg")#image path

scale = 0.3
leaf_A = cv2.resize(leaf_A, None, fx=scale, fy=scale)
leaf_B = cv2.resize(leaf_B, None, fx=scale, fy=scale)
leaf_C = cv2.resize(leaf_C, None, fx=scale, fy=scale)

# Extract leaf masks
leaf_A_mask = get_masks(leaf_A, model)
leaf_B_mask = get_masks(leaf_B, model)
leaf_C_mask = get_masks(leaf_C, model)

isolated_leaf_A = cv2.bitwise_and(leaf_A,leaf_A,mask=leaf_A_mask)
leaf_A_rgba = cv2.cvtColor(isolated_leaf_A, cv2.COLOR_BGR2BGRA)
leaf_A_rgba[:, :, 3] = leaf_A_mask

isolated_leaf_B = cv2.bitwise_and(leaf_B,leaf_B,mask=leaf_B_mask)
leaf_B_rgba = cv2.cvtColor(isolated_leaf_B, cv2.COLOR_BGR2BGRA)
leaf_B_rgba[:, :, 3] = leaf_B_mask

isolated_leaf_C = cv2.bitwise_and(leaf_C,leaf_C,mask=leaf_C_mask)
leaf_C_rgba = cv2.cvtColor(isolated_leaf_C, cv2.COLOR_BGR2BGRA)
leaf_C_rgba[:, :, 3] = leaf_C_mask

#Test the output
cv2.imshow("Leaf A", leaf_A_rgba)
cv2.imshow("Leaf B", leaf_B_rgba)
cv2.imshow("Leaf C", leaf_C_rgba)
cv2.waitKey(0)

#set random angle and scale for augmentation
#angle = np.random.uniform(0, 360)
#scale = np.random.uniform(0.7, 1.3)
#Augmented_leaf_B = augment_leaf(leaf_B_rgba, 90, 1)

#-----------------------------------------------------------------Create Occlusion-----------------------------------------------------------------
result = create_overlap_3(
    leaf_A,
    leaf_B,
    leaf_C,
    leaf_A_mask,
    leaf_B_mask,
    leaf_C_mask,
    desired_overlap=0.25,
    step=5
)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()