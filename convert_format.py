from PIL import Image
import os

data_dir = 'data'
train_img_dir = os.path.join(data_dir, 'TestIJCNN2013Download')
png_img_dir = os.path.join(data_dir, 'test')

if not os.path.exists(png_img_dir):
    os.makedirs(png_img_dir)

for img_name in os.listdir(train_img_dir):
    if img_name[-3:] == "ppm":
        img_path = os.path.join(train_img_dir, img_name)
        img = Image.open(img_path)
        img = img.resize((416, 416), Image.ANTIALIAS)
        png_path = os.path.join(png_img_dir, img_name[:-3]+'png')
        print(png_path)
        img.save(png_path)
