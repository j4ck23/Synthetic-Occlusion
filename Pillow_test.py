from PIL import Image

leaf_A = Image.open("Leaves/leaf_0_sub_2.jpg")#image path
leaf_B = Image.open("Leaves/leaf_0_sub_3.jpg")#image path

background = leaf_A.convert("RGBA")
overlay = leaf_B.convert("RGBA")

new_img = Image.new("RGBA", background.size, (0,0,0,0))
new_img.paste(background, (0,0))
new_img.paste(overlay, (0,0), overlay)
new_img.save("new.png","PNG")