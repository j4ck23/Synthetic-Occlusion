import os
import random
from CreateOcclusion_Simple import create_overlap, get_masks, augment_leaf
import cv2

scale = 0.2
i=0
save_path = "C:/Users/Jack/Documents/Bits of Code/SyntheticOcclusion/Occlusion_Dataset_75"

for i in range(10):
    #Read in random leaf from directory
    Leaf_A = random.choice(os.listdir("Single_Leaves"))
    Leaf_A = cv2.imread(os.path.join("Single_Leaves", Leaf_A))
    Leaf_A = cv2.resize(Leaf_A, None, fx=scale, fy=scale)

    #get leaf mask for Leaf_A
    leaf_A_mask = get_masks(Leaf_A)
    assert leaf_A_mask.shape[:2] == Leaf_A.shape[:2], f"MISMATCH: mask {leaf_A_mask.shape[:2]} vs image {Leaf_A.shape[:2]}"
    if leaf_A_mask is None:
        print(f"Mask not found for {Leaf_A}. Skipping this leaf.")
        continue
    isolated_leaf_A = cv2.bitwise_and(Leaf_A,Leaf_A,mask=leaf_A_mask)
    leaf_A_rgba = cv2.cvtColor(isolated_leaf_A, cv2.COLOR_BGR2BGRA)
    leaf_A_rgba[:, :, 3] = leaf_A_mask
    for file in os.listdir("Single_Leaves"):#Iterate through all leaves in the directory to create occlusions with Leaf_A
        Leaf_B = cv2.imread(os.path.join("Single_Leaves", file))
        Leaf_B = cv2.resize(Leaf_B, None, fx=scale, fy=scale)
        #Get mask for Leaf_B
        leaf_B_mask = get_masks(Leaf_B)
        if leaf_B_mask is None:
            print(f"Mask not found for {file}. Skipping this leaf.")
            continue 
        assert leaf_B_mask.shape[:2] == Leaf_B.shape[:2], f"MISMATCH: mask {leaf_B_mask.shape[:2]} vs image {Leaf_B.shape[:2]}"
        isolated_leaf_B = cv2.bitwise_and(Leaf_B,Leaf_B,mask=leaf_B_mask)
        leaf_B_rgba = cv2.cvtColor(isolated_leaf_B, cv2.COLOR_BGR2BGRA)
        leaf_B_rgba[:, :, 3] = leaf_B_mask
        #create occlusion with Leaf_A and Leaf_B
        result, overlap = create_overlap(Leaf_A,
                                         Leaf_B,
                                         leaf_A_mask,
                                         leaf_B_mask,
                                         desired_overlap=0.75,
                                         step=5#time saving step
                                         )
        path = os.path.join(save_path, f"occlusion_result_{i}_{file}.png")
        cv2.imwrite(path, result)
        cv2.waitKey(1)

print("Completed")