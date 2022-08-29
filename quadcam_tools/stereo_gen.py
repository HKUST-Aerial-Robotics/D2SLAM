from fisheye_undist import *
class StereoGen:
    def __init__(self, undist_l:FisheyeUndist, undist_r:FisheyeUndist, cam_idx_a=0,  cam_idx_b=0, idx_l = 1, idx_r = 0):
        self.undist_l = undist_l
        self.undist_r = undist_r
        self.idx_l = idx_l
        self.idx_r = idx_r
        self.cam_idx_a = cam_idx_a
        self.cam_idx_b = cam_idx_b
    
    def initRectify(self, K1, D1, K2, D2, size, R, T):
        print("Init rectify")
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv.stereoRectify(K1, D1, K2, D2, size, R, T)
        self.mapl0, self.mapl1 = cv.initUndistortRectifyMap(K1, D1, self.R1, self.P1, size, cv.CV_32FC1)
        self.mapr0, self.mapr1 = cv.initUndistortRectifyMap(K2, D2, self.R2, self.P2, size, cv.CV_32FC1)
    
    def genStereo(self, img_l, img_r):
        img_l = self.undist_l.undist(img_l, self.idx_l)
        img_r = self.undist_r.undist(img_r, self.idx_r)
        return img_l, img_r

    def genRectStereo(self, img_l, img_r):
        s_img_l, s_img_r = self.genStereo(img_l, img_r)
        r_img_l = cv.remap(s_img_l, self.mapl0, self.mapl1, cv.INTER_LINEAR)
        r_img_r = cv.remap(s_img_r, self.mapr0, self.mapr1, cv.INTER_LINEAR)
        return r_img_l, r_img_r
    
    def genDisparity(self, img_l, img_r):
        if len(img_l.shape) > 2 and img_l.shape[2] == 3:
            img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        if len(img_r.shape) > 2 and img_r.shape[2] == 3:
            img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        img_l, img_r = self.genRectStereo(img_l, img_r)
        blockSize = 11
        stereo = cv.StereoSGBM.create(minDisparity=0, numDisparities=128, 
            blockSize=blockSize, P1=8 * 3 * blockSize ** 2, P2=32 * 3 * blockSize ** 2, 
            disp12MaxDiff=2, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
        disparity = stereo.compute(img_l, img_r)
        return disparity

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