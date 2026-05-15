import os
from PIL import Image

def crop_images_in_folder(input_folder, output_folder, crop_width, crop_height):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Process only image files (you can add more extensions if needed)
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                # Open the image
                image = Image.open(file_path)
                
                # Define the box to crop (left, upper, right, lower)
                box = (0, 0, crop_width, crop_height)
                cropped_image = image.crop(box)
                
                # Define the output file path
                output_file_path = os.path.join(output_folder, filename)
                
                # Save the cropped image
                cropped_image.save(output_file_path)
                # print(f"Image '{filename}' cropped and saved to '{output_folder}'")
            except Exception as e:
                print(f"Error processing '{filename}': {e}")

# Example usage:
input_folder = '/home/qingwen/workspace/LVSM/data/davis/images/hike-raw'  # Change this to your input folder path
output_folder = '/home/qingwen/workspace/LVSM/data/davis/images/hike'  # Change this to your output folder path
crop_width = 256  # Set the desired crop width
crop_height = 256  # Set the desired crop height

crop_images_in_folder(input_folder, output_folder, crop_width, crop_height)
