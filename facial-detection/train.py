from skimage import io, util, color, transform
import numpy as np

import random
import os

### Configuration ###
HEIGHT = 90.0
WIDTH = 60.0

NEG_FACE_COUNT = 10
### End Configuration ###

def get_labels(path):
    """Get labels from a label file

    Args:
        path: The path of the file

    Returns:
        list: The return list of labels

        The items of labels are index, i, j, h, l

    """
    f_label = open(path)

    datas = list()

    line = f_label.readline()
    while line:
        line = line.replace("\n", "")   # Remove newline
        data = line.split(" ")          # Split data
        datas.append(data)

        line = f_label.readline()

    f_label.close()
    return datas


def transform_to_label_dictionary(labels):
    """Transform labels list to labels lists dictionary

    Args:
        labels: the list of labels returned bt get_labels function

    Returns:
        dict: The return dictionary of lists of labels
    """
    d = dict()
    for label in labels:
        if label[0] not in d.keys():
            l = list()
            l.append(label)
            d.update({label[0]: l})
        else:
            d[label[0]].append(label)
    
    return d


def get_cover_rate(rect1, rect2):
    x1, y1, h1, l1 = (int(rect1[0]), int(rect1[1]), int(rect1[2]), int(rect1[3]))
    x2, y2, h2, l2 = (int(rect2[0]), int(rect2[1]), int(rect2[2]), int(rect2[3]))
    p1_x, p1_y = np.max((x1, x2)), np.max((y1, y2))
    p2_x, p2_y = np.min((x1 + h1, x2 + h2)), np.min((y1 + l1, y2 + l2))
    
    if ( p2_y-p1_y==y1 and p2_x-p1_x==x1 )  or ( p2_y-p1_y==y2 and p2_x-p1_x==x2 ) :
        return 1
        
    a_join = 0
    if (min(x1 + h1, x2 + h2) >= max(x1, x2)) and (min(y1 + l1, y2 + l2) >= max(y1, y2)):
        a_join = (p2_x - p1_x) * (p2_y - p1_y)
    a_union = h1 * l1 + l2 * h2 - a_join
    return a_join / a_union


def prepare_train_labels(input_file, output_file):
    ratio_h_w = 1.5

    labels = get_labels(input_file)

    cliped_labels_file = open(output_file, "w")

    for label in labels:
        x = float(label[2])
        y = float(label[1])
        width = float(label[4])
        height = float(label[3])

        if width * ratio_h_w > height:
            # Width is larger
            new_width = height / ratio_h_w
            new_x = x + width / 2.0 - new_width / 2.0
            cliped_labels_file.write("{} {} {} {} {}\n".format(int(label[0]), int(y), int(new_x), int(height), int(new_width)))
        else:
            # Height is larger
            new_height = width * ratio_h_w
            new_y = y + height / 2.0 - new_height / 2.0
            cliped_labels_file.write("{} {} {} {} {}\n".format(int(label[0]), int(new_y), int(x), int(new_height), int(width)))

    cliped_labels_file.close()


def read_image(filename):
    return util.img_as_float(color.rgb2gray(io.imread(filename)))


def generate_pos_image(labels, path):
    """
    Generate positive image
    """
    i_pos = 0
    for label in labels:
        index = int(float(label[0]))

        image = color.rgb2gray(io.imread("%s/%04d.jpg"%(path, index)))
        
        image_f = util.img_as_float(image)

        face_cliped = image_f[int(label[1]):(int(label[1]) + int(label[3])), int(label[2]):(int(label[2]) + int(label[4]))]

        face_scaled = transform.resize(face_cliped, (HEIGHT, WIDTH))

        io.imsave("%s/pos/%06d.jpg"%(path, i_pos), face_scaled)

        i_pos+=1


# Position
def generate_neg_position(x_max, y_max, width, height):
    x = random.randint(0, x_max - width)
    y = random.randint(0, y_max - height)

    return (x, y)


def generate_neg_image(labels, path):
    label_dict = transform_to_label_dictionary(labels)
    index = 0
    for img_num in label_dict.keys():
        max_iteration, count =  10000, 0
        img = util.img_as_float(color.rgb2gray(io.imread('%s/%04d.jpg'%(path, int(img_num)))))
        img_h, img_l = np.shape(img)[0], np.shape(img)[1]
        
        number_of_face = len(label_dict[img_num])

        for visage, visage_index in zip(label_dict[img_num], range(len(label_dict[img_num]))):
            thresold = 10/number_of_face if visage_index < len(label_dict[img_num]) - 1 \
                else NEG_FACE_COUNT/number_of_face + NEG_FACE_COUNT%number_of_face

            h_window, l_window = (int(visage[3]), int(visage[4]))
            
            while count < thresold:
                iteration = 0

                x = 0
                y = 0
                flag = True
                while iteration < max_iteration:
                    x = random.randint(0, img_h - h_window)
                    y = random.randint(0, img_l - l_window)
                    
                    flag = True
                    for visage_to_compare in label_dict[img_num]:
                        if get_cover_rate((x, y, h_window, l_window), visage_to_compare[1:]) >= 0.5:
                            flag = False
                            break
                    
                    if flag:
                        img_visage = img[x:x + h_window, y:y + l_window]
                        img_resize = transform.resize(img_visage, (HEIGHT, WIDTH))
                        io.imsave('%s/neg/%06d.jpg' % (path, index),img_resize)
                        index= index +1
                        break
                    
                    iteration += 1
                
                count = count + 1


def get_data(path):
    images = os.listdir(path)

    images_data = np.zeros((len(images), int(HEIGHT), int(WIDTH)))

    for image, index in zip(images, range(len(images))):
        im = util.img_as_float(io.imread(os.path.join(path, image)))

        images_data[index, :, :] = im

    return (len(images_data), images_data)


if __name__ == "__main__":
    #prepare_train_labels("label.txt", "clip_label.txt")

    labels = get_labels("clip_label.txt")   # Read labels

    generate_pos_image(labels, "train")     # Generate pos

    generate_neg_image(labels, "train")     # Generate neg

    #pos_length, pos_images = get_data("train/pos")   # Get positive image data

    #print(pos_length)
    #print(pos_images)

    #neg_length, neg_images = get_data("train/neg")   # Get negative image data




