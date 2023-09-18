from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np

from os.path import join


def read_metrics(dir_path):

    side_texts = []

    ap_filepath = join(dir_path, "Metrics", "AP.txt")
    ar_filepath = join(dir_path, "Metrics", "AR.txt")
    IoU_filepath = join(dir_path, "Metrics", "Pixel_IoU.txt")

    file_ap = open(ap_filepath, "r")
    file_ar = open(ar_filepath, "r")
    file_IoU = open(IoU_filepath, "r")

    files = [file_ap, file_ar, file_IoU]
    metric_type = ["AP", "AR", "Mean IoU"]

    for i, file in enumerate(files):
        data = file.read()
        values = data.split(",")
        values.pop(-1)
        values = [metric_type[i] + ": " + str('%.3f' % float(val.split(":")[1])) for val in values]
        side_texts.append(values)

    side_texts = np.array(side_texts).transpose()

    return side_texts


def read_images(dir_path,num_images):

    all_images = []

    for i in range(num_images):

        images=[]

        real_img = Image.open(join(dir_path, "real_img", str(i) + ".png"))
        images.append(real_img)

        semantic_mask = Image.open(join(dir_path, "semantic_predicted", str(i) + ".png"))
        images.append(semantic_mask)

        instance_predicted = Image.open(join(dir_path, "instance_predicted", str(i) + ".png"))
        images.append(instance_predicted)

        instance_target = Image.open(join(dir_path, "instance_target", str(i) + ".png"))
        images.append(instance_target)

        all_images.append(images)

    return all_images


if __name__ == "__main__":
    canvas_width = 4  # Number of columns
    canvas_height = 4  # Number of rows
    image_width = 320  # Width of each image
    image_height = 320  # Height of each image
    margin = 20  # Margin between images
    text_padding = 30
    text_top = 100
    text_side = 320
    font = ImageFont.truetype("/home/endrit/geo-scene/data/Arial.ttf", size=30)
    dir_path = "/home/endrit/geo-scene/results/predictions/DBSCAN_projected_good_images/1_1"
    top_texts = ["Image", "Semantic mask", "Predicted Instances", "Target Instances"]
    side_texts = read_metrics(dir_path)
    images = read_images(dir_path,canvas_height)


    # Calculate the size of the canvas
    canvas_width_px = (
            canvas_width * (image_width+margin) + text_side
    )
    canvas_height_px = (
            canvas_height * (image_height+margin) + text_top
    )

    # Create a blank canvas with a white background
    canvas = Image.new("RGB", (canvas_width_px, canvas_height_px), "white")
    draw = ImageDraw.Draw(canvas)



    for i in range(canvas_height):
        for j in range(3):
            text_position = (text_padding, (i) * (image_height + margin) + text_top + (j + 2) * text_padding)
            draw.text(text_position, side_texts[i][j], fill="black", font=font)

    for i, text in enumerate(top_texts):
        text_position = ((i + 1) * (image_width + margin) + text_padding, text_padding)
        draw.text(text_position, text, fill="black", font=font)


    for i in range(canvas_height):
        for j in range(canvas_width):
            image_position = (image_width + (image_width+margin)*i,  text_top + + j*(image_height+margin) )
            canvas.paste(images[j][i],image_position)


    canvas.save("/home/endrit/Desktop/"+dir_path.split("/")[-1]+dir_path.split("/")[-2]+".png")




