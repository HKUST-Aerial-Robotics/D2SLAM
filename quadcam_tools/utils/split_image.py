def splitImage(img, num_subimages = 4):
    #Split image vertically
    h, w = img.shape[:2]
    sub_w = w // num_subimages
    sub_imgs = []
    for i in range(num_subimages):
        sub_imgs.append(img[:, i*sub_w:(i+1)*sub_w])
    return sub_imgs
