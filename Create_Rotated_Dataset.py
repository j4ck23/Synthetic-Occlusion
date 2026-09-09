import os
import random
from CreateOcclusion_Simple import create_overlap, get_masks, augment_leaf
import cv2
import numpy as np

scale = 0.2
i=0
save_path = "C:/Users/Jack/Documents/Bits of Code/SyntheticOcclusion/Rotated_50"

for i in range(10):
    #Read in random leaf from directory
    Leaf_A = random.choice(os.listdir("Single_Leaves"))
    Leaf_A = cv2.imread(os.path.join("Single_Leaves", Leaf_A))
    Leaf_A = cv2.resize(Leaf_A, None, fx=scale, fy=scale)
    for file in os.listdir("Single_Leaves"):#Iterate through all leaves in the directory to create occlusions with Leaf_A
        Leaf_B = cv2.imread(os.path.join("Single_Leaves", file))
        Leaf_B = cv2.resize(Leaf_B, None, fx=scale, fy=scale)
        #set random angle and scale for augmentation
        angle = np.random.uniform(0, 360)
        #Augment Leaf_B (rotation and scaling)
        Leaf_B = augment_leaf(Leaf_B, angle, 1)
        leaf_B_mask = get_masks(Leaf_B)
        if leaf_B_mask is None:
            print(f"Mask not found for {file}. Skipping this leaf.")
            continue 
        angle = np.random.uniform(0, 360)
        Leaf_A = augment_leaf(Leaf_A, angle, 1)
        leaf_A_mask = get_masks(Leaf_A)
        if leaf_A_mask is None:
            print(f"Mask not found for {Leaf_A}. Skipping this leaf.")
            continue
        #create occlusion with Leaf_A and Leaf_B
        result, overlap = create_overlap(Leaf_A,
                                         Leaf_B,
                                         leaf_A_mask,
                                         leaf_B_mask,
                                         desired_overlap=0.50,
                                         step=5#time saving step
                                         )
        path = os.path.join(save_path, f"occlusion_result_{i}_{file}.png")
        cv2.imwrite(path, result)
        cv2.waitKey(1)

print("Completed")