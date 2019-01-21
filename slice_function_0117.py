from PIL import Image
import numpy as np
import cv2
import random

def del_middle (middle, x, y, window):
    #side = window//2
    side = window
    xmax = middle.shape[0]
    ymax = middle.shape[1]

    xlh = max(x-side, 0)
    xrh = min(x+side+1, xmax)

    ylh = max(y-side, 0)
    yrh = min(y+side+1, ymax)

    middle[xlh:xrh, ylh:yrh] = 0

def del_middle_new (middle, del_map, x, y, window):
    #side = window//2
    side = window
    xmax = middle.shape[0]
    ymax = middle.shape[1]

    xlh = max(x-side, 0)
    xrh = min(x+side+1, xmax)

    ylh = max(y-side, 0)
    yrh = min(y+side+1, ymax)

    middle[xlh:xrh, ylh:yrh, :] = del_map[xlh:xrh, ylh:yrh, :]



def criterion (cost, infer_map, pix, threshold=0.000001, ratio=0.1):
    num = 0
    # number of pixel does not exceed the threshold
    for i in range(infer_map.shape[0]):
        for j in range(infer_map.shape[1]):
            if infer_map[i,j] > cost*threshold:
                num += 1

    # if it out numbers the total pixel, increase the window size
    if (num/(infer_map.shape[0]*infer_map.shape[1]-pix)) > ratio:
        return True
    else:
        return False

def min_infer (infer, slice_map):
	# return index of minimum from infer, except for the one's with zero slice_map value
    min_list = []
    min_i = -1
    min_j = -1
    min_val = float('inf')

    for i in range(infer.shape[0]):
        for j in range(infer.shape[1]):
            if slice_map[i,j] == 0:
                continue
            if min_val == infer[i,j]:
                min_list.append((i,j))
                continue
            if min_val > infer[i,j]:
                min_list = [(i,j)]
                min_val = infer[i,j]


    return random.choice(min_list)

def mask_save (image, slice_map, ind, recover=False, save=True):
    # saves the image masked with slice_map. If recover option is on, each pixels in slice_map will make
    # 5x5 patch.
    mask_original = image.copy()

    #nslice_map = cv2.resize (slice_map, dsize=image.shape[:-1], interpolation=cv2.INTER_AREA)
    
    nslice_map = cv2.resize (slice_map, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(mask_original.astype(np.uint8))
    im = im.convert('RGB')
    im.save("./mimages_new/masked_tmp.jpeg") 
    if recover:
        temp_map = np.zeros (slice_map.shape)
        temp_map = np.pad (temp_map, [(2,2),(2,2)], mode='constant')
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 1:
                    temp_map[i:i+5,j:j+5] += 1
        temp_map = temp_map[2:-2,2:-2]
        temp_map = cv2.resize (temp_map, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        temp_map = nslice_map
    print('saving...  ')
    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            if temp_map[i,j] == 0:
                mask_original[i,j] = [0,0,0]
                # mask_original[i,j,0] =0
                # mask_original[i,j,1] =0
                # mask_original[i,j,2] =0
            
    if save:
        im = Image.fromarray(mask_original.astype(np.uint8))
        im.convert('RGB')
        im.save("./mimages_new/masked_"+str(ind)+".jpeg")
    return mask_original



def slice_save (image, slice_map, ind):
    mask_original = image.copy ()
	
    #nslice_map = cv2.resize (slice_map, dsize=image.shape[:-1], interpolation=cv2.INTER_AREA)
    nslice_map = cv2.resize (slice_map,  dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    temp_map = nslice_map

    remain_pix = 0
    avg_rgb = [0,0,0]

    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            if temp_map[i,j] != 0:
                    avg_rgb += mask_original[i,j]
                    remain_pix += 1

    avg_rgb /= remain_pix


    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            if temp_map[i,j] == 0:
                mask_original[i,j] = avg_rgb
                # mask_original[i,j,0] =0
                # mask_original[i,j,1] =0
                # mask_original[i,j,2] =0
            

    im = Image.fromarray(mask_original.astype(np.uint8))
    im.save("./sliced_images_new/sliced_"+str(ind)+".jpeg")
    return mask_original

