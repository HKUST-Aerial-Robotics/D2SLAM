from fisheye_undist import *
class StereoGen:
    def __init__(self, undist_l:FisheyeUndist, undist_r:FisheyeUndist, R, T, idx_l = 1, idx_r = 0):
        self.undist_l = undist_l
        self.undist_r = undist_r
        self.R = R
        self.T = T
        self.idx_l = idx_l
        self.idx_r = idx_r
    
    def genStereo(self, img_l, img_r):
        if img_l.shape[2] == 3:
            img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        if img_r.shape[2] == 3:
            img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        img_l = self.undist_l.undist(img_l, self.idx_l)
        img_r = self.undist_r.undist(img_r, self.idx_r)
        return img_l, img_r

    def calibPhotometric(self, img_l, img_r, blur_size=31):
        if img_l.shape[2] == 3:
            img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        if img_r.shape[2] == 3:
            img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        img_l = self.undist_l.undist(img_l, self.idx_l)
        img_r = self.undist_r.undist(img_r, self.idx_r)
        #Gaussain blur
        img_l = cv.GaussianBlur(img_l, (blur_size, blur_size), 0)
        img_r = cv.GaussianBlur(img_r, (blur_size, blur_size), 0)
        #Calculate min max
        min_l = np.min(img_l)
        max_l = np.max(img_l)
        min_r = np.min(img_r)
        max_r = np.max(img_r)

        show_img_l = cv.cvtColor(img_l, cv.COLOR_GRAY2BGR)
        show_img_r = cv.cvtColor(img_r, cv.COLOR_GRAY2BGR)
        #print min max on image
        cv.putText(show_img_l, f"min: {min_l}, max: {max_l}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(show_img_r, f"min: {min_r}, max: {max_r}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #if max=255, then need to puttext 'lower the expousre'
        if max_l == 255:
            cv.putText(show_img_l, "Please lower the exposure", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if max_r == 255:
            cv.putText(show_img_r, "Please lower the exposure", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return show_img_l, show_img_r