import os
from PIL import Image, ImageSequence
import shutil
folder = '/home/zubairirshad/zero123/objaverse-rendering/partnet_mobility_fixed_camera/10040'

new_dir = '/home/zubairirshad/zero123/objaverse-rendering/test_art'

os.makedirs(new_dir, exist_ok=True)
subfolders = os.listdir(folder)

subfolders.sort()

image_name = '014.png'
all_images = []
for subfolder in subfolders:
    image_dir = os.path.join(folder, subfolder, subfolder)
    print("image_dir", image_dir)
    image_path = os.path.join(image_dir, image_name)
    print("image_path", image_path)
    image = Image.open(image_path)
    all_images.append(image)

    new_path = os.path.join(new_dir, subfolder+image_name)
    shutil.copy(image_path, new_path)

    #copy all images to a new directory


# first_image = all_images[0]
# size = first_image.size
# mode = first_image.mode

# gif = Image.new(mode, size)

# for image in all_images:
#     # frame = Image.open(image_path)
#     gif.paste(image)

# gif.save('partnet.gif', save_all=True, append_images=list(ImageSequence.Iterator(gif)), duration=500, loop=0)

# #make a gif
all_images[0].save('partnet.gif',
               save_all=True, append_images=all_images[1:], optimize=False, duration=100, loop=1)

# all_images[0].save('partnet.gif', save_all=True, append_images=all_images[1:])
    
    