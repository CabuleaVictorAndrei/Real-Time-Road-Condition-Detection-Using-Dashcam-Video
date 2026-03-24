import os
from PIL import Image

def crop_center_custom_all(input_folder, output_folder, left_pixels, right_pixels, top_pixels, bottom_pixels):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)
            width, height = img.size

            center_x = width // 2
            center_y = height // 2

            left = center_x - left_pixels
            right = center_x + right_pixels
            top = center_y - top_pixels
            bottom = center_y + bottom_pixels

            left = max(left, 0)
            right = min(right, width)
            top = max(top, 0)
            bottom = min(bottom, height)

            cropped_img = img.crop((left, top, right, bottom))
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)
            print(f"Cropped {filename} -> saved to {output_path}")

crop_center_custom_all(
    "E:/Video/original",
    "E:/Video/crop",
    left_pixels=750,
    right_pixels=750,
    top_pixels=400,
    bottom_pixels=400
)

folder_path = r"E:\Video\crop"

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isdir(old_path):
        continue

    new_filename = "2912202510" + filename
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)
