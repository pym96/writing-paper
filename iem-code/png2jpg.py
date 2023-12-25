from PIL import Image
import os

def convert_png_to_jpeg(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(source_folder, filename))
            rgb_img = img.convert('RGB')  # Convert to RGB
            target_file = os.path.splitext(filename)[0] + '.jpg'
            rgb_img.save(os.path.join(target_folder, target_file))

source_folder = r'C:\Users\pym66\Documents\文献\writing_paper\code\iem-code\datasets'
target_folder = r'C:\Users\pym66\Documents\文献\writing_paper\code\iem-code\datasets_jpg'
convert_png_to_jpeg(source_folder, target_folder)
