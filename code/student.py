import numpy as np
import matplotlib
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

from numpy import linalg as LA # eigenvalue find
from scipy import signal

#################################### HW#3 basic function #HOG Feature

def build_vocabulary(image_paths, vocab_size):
    '''
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    to do this. Note that this can take a VERY LONG TIME to complete (upwards
    of ten minutes for large numbers of features and large max_iter), so set
    the max_iter argument to something low (we used 100) and be patient. You
    may also find success setting the "tol" argument (see documentation for
    details)
    '''
    bag_of_features = []
    bag_hog=[]
    print("Extract features")
    for path in image_paths:
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (128, 128))
        bag_hog = hog(gray_img, cells_per_block = (4, 4), pixels_per_cell = (4, 4), feature_vector = True)
        bag_of_features = np.concatenate((bag_of_features,bag_hog),axis=None)
    bag_of_features = bag_of_features.reshape(-1,4*4*9)
    print("Compute vocab")
    list_voca = KMeans(n_clusters=vocab_size,max_iter = 100, random_state=0).fit(bag_of_features)
    vocab=list_voca.cluster_centers_
    return vocab

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''
    
    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')
    #TODO: Implement this function!
    bag_of_word = []
    bag_hog=[]
    image_feats = np.zeros((len(image_paths),len(vocab)))
    print("Extract features")
    for i, path in enumerate(image_paths):
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (128, 128))
        bag_hog = hog(gray_img, cells_per_block = (4, 4), pixels_per_cell = (4, 4), feature_vector = True)
        bag_of_word = bag_hog.reshape(-1,4*4*9)
        dist = cdist(vocab, bag_of_word, 'euclidean')
        mdist = np.argmin(dist, axis = 0)
        histo, bins = np.histogram(mdist, range(len(vocab)+1))
        if np.linalg.norm(histo) == 0:
            image_feats[i, :] = histo
        else:
            image_feats[i, :] = histo / np.linalg.norm(histo)
    return image_feats

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    We suggest you look at the sklearn.svm module, including the LinearSVC
    class. With the right arguments, you can get a 15-class SVM as described
    above in just one call! Be sure to read the documentation carefully.
    '''

    # TODO: Implement this function!

    svc = LinearSVC(random_state = 0)
    #svc = SVC(random_state=0)
    param_C = [0.001 , 0.01 , 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_grid = [{'C': param_C}]

    gs = GridSearchCV(estimator = svc,
                      param_grid= param_grid,
                      scoring='accuracy',
                     )
    
    gs = gs.fit(train_image_feats, train_labels)
    
    print(f'Best Training Score = {gs.best_score_:.3f} with parameters {gs.best_params_}')
    
    classifier = gs.best_estimator_
    classifier.fit(train_image_feats, train_labels)
    
    pred_label = classifier.predict(test_image_feats)
    return pred_label

#################################### SIFT FEATURE


def sift_bocabulary(image_paths, vocab_size):
    bag_of_features = []
    bag_sift=[]
    print("Extract features")
    for path in image_paths:
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (64, 64))
        bag_sift = sift_like_features(gray_img)
        bag_of_features = np.concatenate((bag_of_features,bag_sift),axis=None)
    bag_of_features = bag_of_features.reshape(-1,4*4*8)
    print("Compute vocab")
    list_voca = KMeans(n_clusters=vocab_size,max_iter = 100, random_state=0).fit(bag_of_features)
    vocab=list_voca.cluster_centers_
    return vocab

def get_sift(image_paths):
    vocab = np.load('vocab_sift.npy')
    print('Loaded sift vocab from file.')
    #TODO: Implement this function!
    bag_sift=[]
    image_feats = np.zeros((len(image_paths),len(vocab)))
    print("Extract features")
    for i, path in enumerate(image_paths):
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (64, 64))
        bag_sift = sift_like_features(gray_img)
        bag_sift = np.array(bag_sift).reshape(-1,4*4*8)
        dist = cdist(vocab, bag_sift, 'euclidean')
        mdist = np.argmin(dist, axis = 0)
        histo, bins = np.histogram(mdist, range(len(vocab)+1))
        if np.linalg.norm(histo) == 0:
            image_feats[i, :] = histo
        else:
            image_feats[i, :] = histo / np.linalg.norm(histo)
    return image_feats

def sift_like_features(img):
    # make harris corner detector (key point)
    I_x = grad_x(img)
    I_y = grad_y(img)
    I_xx = I_x**2
    I_xy = I_x*I_y
    I_yy = I_y**2

    G1=signal.convolve2d(I_xx, gaus_f(5, 1.4), mode='same')
    G23=signal.convolve2d(I_xy, gaus_f(5, 1.4), mode='same')
    G4=signal.convolve2d(I_yy, gaus_f(5, 1.4), mode='same')

    height=img.shape[0]
    width=img.shape[1]
    corner_point=np.zeros((img.shape[0],img.shape[1]), np.int16)
    interest_point=np.zeros((img.shape[0],img.shape[1]), np.int16)
    M=np.zeros((2,2), np.float32)

    offset=2
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            M1 = np.sum(G1[y-offset:y+offset, x-offset:x+offset])
            M23 = np.sum(G23[y-offset:y+offset, x-offset:x+offset])
            M4 = np.sum(G4[y-offset:y+offset, x-offset:x+offset])

            M[0,0]=M1
            M[0,1]=M23
            M[1,0]=M23
            M[1,1]=M4

            EV = LA.eig(M)[0]
            corner_point[y,x]=EV[0]*EV[1]-0.04*(EV[0]+EV[1])**2
    th_value=0.1*corner_point.max()
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            if (corner_point[y,x]>th_value):
                interest_point[y,x]=1        # value 1 : key point, 0 : o.w
# make sift_like feature
    img_blur = signal.convolve2d(img, gaus_f(16, 1.4), mode='same')
    G_mag, G_theta = ext_Gth(img_blur)

    # check number of key point
    num_point=0
    for y in range(8, height-8):
        for x in range(8, width-8):
            if (interest_point[y,x]==1):
                num_point+=1        
    Rot_theta=np.zeros((height,width,num_point),np.float64)
    key_theta=np.zeros((num_point),np.float64)
    t=0
    for y in range(8, height-8):
        for x in range(8, width-8):
            if (interest_point[y,x]==1 and t<num_point):
                key_theta[t]=G_theta[y,x]
                t+=1
    for k in range(0, num_point):
        for y in range(0, height):
            for x in range(0, width):
                Rot_theta[y,x,k]=G_theta[y,x]-key_theta[k]
                if (Rot_theta[y,x,k]<0):
                    Rot_theta[y,x,k]=Rot_theta[y,x,k]+360
                if (Rot_theta[y,x,k]>360):
                    Rot_theta[y,x,k]=Rot_theta[y,x,k]-360
    Histog=np.zeros((16,8),np.float16)
    descrip=[]
    k=-1
    for y in range(8, height-8):
        for x in range(8, width-8):
            if (interest_point[y,x]==1 and k<num_point-1):
                k+=1
                # make 16 histogram
                for m in range (4,8): # 1st histogram
                    for n in range (4,8):
                        if (Rot_theta[y-n][x-m][k]>=0 & 45>Rot_theta[y-n][x-m][k]):# 1st bin
                            Histog[0,0] = Histog[0,0]+1
                        elif (Rot_theta[y-n][x-m][k]>=45 & 90>Rot_theta[y-n][x-m][k]):# 2rd bin
                            Histog[0,1] = Histog[0,1]+1
                        elif (Rot_theta[y-n][x-m][k]>=90 & 135>Rot_theta[y-n][x-m][k]):# 3rd bin
                            Histog[0,2] = Histog[0,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x-m][k]):# 4th bin
                            Histog[0,3] = Histog[0,3]+1
                        elif (Rot_theta[y-n][x-m][k]>=180 & 225>Rot_theta[y-n][x-m][k]):# 5th bin
                            Histog[0,4] = Histog[0,4]+1
                        elif (Rot_theta[y-n][x-m][k]>=225 & 270>Rot_theta[y-n][x-m][k]):# 6th bin
                            Histog[0,5] = Histog[0,5]+1
                        elif (Rot_theta[y-n][x-m][k]>=270 & 315>Rot_theta[y-n][x-m][k]):# 7th bin
                            Histog[0,6] = Histog[0,6]+1
                        elif (Rot_theta[y-n][x-m][k]>=315 & 360>Rot_theta[y-n][x-m][k]):# 8th bin
                            Histog[0,7] = Histog[0,7]+1
                for m in range (0,4): # 2nd histogram
                    for n in range (4,8):
                        if (Rot_theta[y-n][x-m][k]>=0 & 45>Rot_theta[y-n][x-m][k]):# 1st bin
                            Histog[1,0] = Histog[1,0]+1
                        elif (Rot_theta[y-n][x-m][k]>=45 & 90>Rot_theta[y-n][x-m][k]):# 2rd bin
                            Histog[1,1] = Histog[1,1]+1
                        elif (Rot_theta[y-n][x-m][k]>=90 & 135>Rot_theta[y-n][x-m][k]):# 3rd bin
                            Histog[1,2] = Histog[1,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x-m][k]):# 4th bin
                            Histog[1,3] = Histog[1,3]+1
                        elif (Rot_theta[y-n][x-m][k]>=180 & 225>Rot_theta[y-n][x-m][k]):# 5th bin
                            Histog[1,4] = Histog[1,4]+1
                        elif (Rot_theta[y-n][x-m][k]>=225 & 270>Rot_theta[y-n][x-m][k]):# 6th bin
                            Histog[1,5] = Histog[1,5]+1
                        elif (Rot_theta[y-n][x-m][k]>=270 & 315>Rot_theta[y-n][x-m][k]):# 7th bin
                            Histog[1,6] = Histog[1,6]+1
                        elif (Rot_theta[y-n][x-m][k]>=315 & 360>Rot_theta[y-n][x-m][k]):# 8th bin
                            Histog[1,7] = Histog[1,7]+1
                for m in range (1,5): # 3rd histogram
                    for n in range (4,8):
                        if (Rot_theta[y-n][x+m][k]>=0 & 45>Rot_theta[y-n][x+m][k]):# 1st bin
                            Histog[2,0] = Histog[2,0]+1
                        elif (Rot_theta[y-n][x+m][k]>=45 & 90>Rot_theta[y-n][x+m][k]):# 2rd bin
                            Histog[2,1] = Histog[2,1]+1
                        elif (Rot_theta[y-n][x+m][k]>=90 & 135>Rot_theta[y-n][x+m][k]):# 3rd bin
                            Histog[2,2] = Histog[2,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x+m][k]):# 4th bin
                            Histog[2,3] = Histog[3,3]+1
                        elif (Rot_theta[y-n][x+m][k]>=180 & 225>Rot_theta[y-n][x+m][k]):# 5th bin
                            Histog[2,4] = Histog[3,4]+1
                        elif (Rot_theta[y-n][x+m][k]>=225 & 270>Rot_theta[y-n][x+m][k]):# 6th bin
                            Histog[2,5] = Histog[3,5]+1
                        elif (Rot_theta[y-n][x+m][k]>=270 & 315>Rot_theta[y-n][x+m][k]):# 7th bin
                            Histog[2,6] = Histog[3,6]+1
                        elif (Rot_theta[y-n][x+m][k]>=315 & 360>Rot_theta[y-n][x+m][k]):# 8th bin
                            Histog[2,7] = Histog[3,7]+1
                for m in range (5,9): # 4th histogram
                    for n in range (4,8):
                        if (Rot_theta[y-n][x+m][k]>=0 & 45>Rot_theta[y-n][x+m][k]):# 1st bin
                            Histog[3,0] = Histog[3,0]+1
                        elif (Rot_theta[y-n][x+m][k]>=45 & 90>Rot_theta[y-n][x+m][k]):# 2rd bin
                            Histog[3,1] = Histog[3,1]+1
                        elif (Rot_theta[y-n][x+m][k]>=90 & 135>Rot_theta[y-n][x+m][k]):# 3rd bin
                            Histog[3,2] = Histog[3,2]+1
                        elif (Rot_theta[y-n][x+m][k]>=135 & 180>Rot_theta[y-n][x+m][k]):# 4th bin
                            Histog[3,3] = Histog[3,3]+1
                        elif (Rot_theta[y-n][x+m][k]>=180 & 225>Rot_theta[y-n][x+m][k]):# 5th bin
                            Histog[3,4] = Histog[3,4]+1
                        elif (Rot_theta[y-n][x+m][k]>=225 & 270>Rot_theta[y-n][x+m][k]):# 6th bin
                            Histog[3,5] = Histog[3,5]+1
                        elif (Rot_theta[y-n][x+m][k]>=270 & 315>Rot_theta[y-n][x+m][k]):# 7th bin
                            Histog[3,6] = Histog[3,6]+1
                        elif (Rot_theta[y-n][x+m][k]>=315 & 360>Rot_theta[y-n][x+m][k]):# 8th bin
                            Histog[3,7] = Histog[3,7]+1
                for m in range (4,8): # 5th histogram
                    for n in range (0,4):
                        if (Rot_theta[y-n][x-m][k]>=0 & 45>Rot_theta[y-n][x-m][k]):# 1st bin
                            Histog[4,0] = Histog[4,0]+1
                        elif (Rot_theta[y-n][x-m][k]>=45 & 90>Rot_theta[y-n][x-m][k]):# 2rd bin
                            Histog[4,1] = Histog[4,1]+1
                        elif (Rot_theta[y-n][x-m][k]>=90 & 135>Rot_theta[y-n][x-m][k]):# 3rd bin
                            Histog[4,2] = Histog[4,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x-m][k]):# 4th bin
                            Histog[4,3] = Histog[4,3]+1
                        elif (Rot_theta[y-n][x-m][k]>=180 & 225>Rot_theta[y-n][x-m][k]):# 5th bin
                            Histog[4,4] = Histog[4,4]+1
                        elif (Rot_theta[y-n][x-m][k]>=225 & 270>Rot_theta[y-n][x-m][k]):# 6th bin
                            Histog[4,5] = Histog[4,5]+1
                        elif (Rot_theta[y-n][x-m][k]>=270 & 315>Rot_theta[y-n][x-m][k]):# 7th bin
                            Histog[4,6] = Histog[4,6]+1
                        elif (Rot_theta[y-n][x-m][k]>=315 & 360>Rot_theta[y-n][x-m][k]):# 8th bin
                            Histog[4,7] = Histog[4,7]+1
                for m in range (0,4): # 6th histogram
                    for n in range (0,4):
                        if (Rot_theta[y-n][x-m][k]>=0 & 45>Rot_theta[y-n][x-m][k]):# 1st bin
                            Histog[5,0] = Histog[5,0]+1
                        elif (Rot_theta[y-n][x-m][k]>=45 & 90>Rot_theta[y-n][x-m][k]):# 2rd bin
                            Histog[5,1] = Histog[5,1]+1
                        elif (Rot_theta[y-n][x-m][k]>=90 & 135>Rot_theta[y-n][x-m][k]):# 3rd bin
                            Histog[5,2] = Histog[5,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x-m][k]):# 4th bin
                            Histog[5,3] = Histog[5,3]+1
                        elif (Rot_theta[y-n][x-m][k]>=180 & 225>Rot_theta[y-n][x-m][k]):# 5th bin
                            Histog[5,4] = Histog[5,4]+1
                        elif (Rot_theta[y-n][x-m][k]>=225 & 270>Rot_theta[y-n][x-m][k]):# 6th bin
                            Histog[5,5] = Histog[5,5]+1
                        elif (Rot_theta[y-n][x-m][k]>=270 & 315>Rot_theta[y-n][x-m][k]):# 7th bin
                            Histog[5,6] = Histog[5,6]+1
                        elif (Rot_theta[y-n][x-m][k]>=315 & 360>Rot_theta[y-n][x-m][k]):# 8th bin
                            Histog[5,7] = Histog[5,7]+1
                for m in range (1,5): # 7th histogram
                    for n in range (0,4):
                        if (Rot_theta[y-n][x+m][k]>=0 & 45>Rot_theta[y-n][x+m][k]):# 1st bin
                            Histog[6,0] = Histog[6,0]+1
                        elif (Rot_theta[y-n][x+m][k]>=45 & 90>Rot_theta[y-n][x+m][k]):# 2rd bin
                            Histog[6,1] = Histog[6,1]+1
                        elif (Rot_theta[y-n][x+m][k]>=90 & 135>Rot_theta[y-n][x+m][k]):# 3rd bin
                            Histog[6,2] = Histog[6,2]+1
                        elif (Rot_theta[y-n][x-m][k]>=135 & 180>Rot_theta[y-n][x+m][k]):# 4th bin
                            Histog[6,3] = Histog[6,3]+1
                        elif (Rot_theta[y-n][x+m][k]>=180 & 225>Rot_theta[y-n][x+m][k]):# 5th bin
                            Histog[6,4] = Histog[6,4]+1
                        elif (Rot_theta[y-n][x+m][k]>=225 & 270>Rot_theta[y-n][x+m][k]):# 6th bin
                            Histog[6,5] = Histog[6,5]+1
                        elif (Rot_theta[y-n][x+m][k]>=270 & 315>Rot_theta[y-n][x+m][k]):# 7th bin
                            Histog[6,6] = Histog[6,6]+1
                        elif (Rot_theta[y-n][x+m][k]>=315 & 360>Rot_theta[y-n][x+m][k]):# 8th bin
                            Histog[6,7] = Histog[6,7]+1
                for m in range (5,9): # 8th histogram
                    for n in range (0,4):
                        if (Rot_theta[y-n][x+m][k]>=0 & 45>Rot_theta[y-n][x+m][k]):# 1st bin
                            Histog[7,0] = Histog[7,0]+1
                        elif (Rot_theta[y-n][x+m][k]>=45 & 90>Rot_theta[y-n][x+m][k]):# 2rd bin
                            Histog[7,1] = Histog[7,1]+1
                        elif (Rot_theta[y-n][x+m][k]>=90 & 135>Rot_theta[y-n][x+m][k]):# 3rd bin
                            Histog[7,2] = Histog[7,2]+1
                        elif (Rot_theta[y-n][x+m][k]>=135 & 180>Rot_theta[y-n][x+m][k]):# 4th bin
                            Histog[7,3] = Histog[7,3]+1
                        elif (Rot_theta[y-n][x+m][k]>=180 & 225>Rot_theta[y-n][x+m][k]):# 5th bin
                            Histog[7,4] = Histog[7,4]+1
                        elif (Rot_theta[y-n][x+m][k]>=225 & 270>Rot_theta[y-n][x+m][k]):# 6th bin
                            Histog[7,5] = Histog[7,5]+1
                        elif (Rot_theta[y-n][x+m][k]>=270 & 315>Rot_theta[y-n][x+m][k]):# 7th bin
                            Histog[7,6] = Histog[7,6]+1
                        elif (Rot_theta[y-n][x+m][k]>=315 & 360>Rot_theta[y-n][x+m][k]):# 8th bin
                            Histog[7,7] = Histog[7,7]+1
                for m in range (4,8): # 9th histogram
                    for n in range (1,5):
                        if (Rot_theta[y+n][x-m][k]>=0 & 45>Rot_theta[y+n][x-m][k]):# 1st bin
                            Histog[8,0] = Histog[8,0]+1
                        elif (Rot_theta[y+n][x-m][k]>=45 & 90>Rot_theta[y+n][x-m][k]):# 2rd bin
                            Histog[8,1] = Histog[8,1]+1
                        elif (Rot_theta[y+n][x-m][k]>=90 & 135>Rot_theta[y+n][x-m][k]):# 3rd bin
                            Histog[8,2] = Histog[8,2]+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x-m][k]):# 4th bin
                            Histog[8,3] = Histog[8,3]+1
                        elif (Rot_theta[y+n][x-m][k]>=180 & 225>Rot_theta[y+n][x-m][k]):# 5th bin
                            Histog[8,4] = Histog[8,4]+1
                        elif (Rot_theta[y+n][x-m][k]>=225 & 270>Rot_theta[y+n][x-m][k]):# 6th bin
                            Histog[8,5] = Histog[8,5]+1
                        elif (Rot_theta[y+n][x-m][k]>=270 & 315>Rot_theta[y+n][x-m][k]):# 7th bin
                            Histog[8,6] = Histog[8,6]+1
                        elif (Rot_theta[y+n][x-m][k]>=315 & 360>Rot_theta[y+n][x-m][k]):# 8th bin
                            Histog[8,7] = Histog[8,7]+1
                for m in range (0,4): # 10th histogram
                    for n in range (1,5):
                        if (Rot_theta[y+n][x-m][k]>=0 & 45>Rot_theta[y+n][x-m][k]):# 1st bin
                            Histog[9,0] = Histog[9,0]+1
                        elif (Rot_theta[y+n][x-m][k]>=45 & 90>Rot_theta[y+n][x-m][k]):# 2rd bin
                            Histog[9,1] = Histog[9,1]+1
                        elif (Rot_theta[y+n][x-m][k]>=90 & 135>Rot_theta[y+n][x-m][k]):# 3rd bin
                            Histog[9,2] = Histog[9,2],+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x-m][k]):# 4th bin
                            Histog[9,3] = Histog[9,3],+1
                        elif (Rot_theta[y+n][x-m][k]>=180 & 225>Rot_theta[y+n][x-m][k]):# 5th bin
                            Histog[9,4] = Histog[9,4]+1
                        elif (Rot_theta[y+n][x-m][k]>=225 & 270>Rot_theta[y+n][x-m][k]):# 6th bin
                            Histog[9,5] = Histog[9,5]+1
                        elif (Rot_theta[y+n][x-m][k]>=270 & 315>Rot_theta[y+n][x-m][k]):# 7th bin
                            Histog[9,6] = Histog[9,6]+1
                        elif (Rot_theta[y+n][x-m][k]>=315 & 360>Rot_theta[y+n][x-m][k]):# 8th bin
                            Histog[9,7] = Histog[9,7]+1
                for m in range (1,5): # 11th histogram
                    for n in range (1,5):
                        if (Rot_theta[y+n][x+m][k]>=0 & 45>Rot_theta[y+n][x+m][k]):# 1st bin
                            Histog[10,0] = Histog[10,0]+1
                        elif (Rot_theta[y+n][x+m][k]>=45 & 90>Rot_theta[y+n][x+m][k]):# 2rd bin
                            Histog[10,1] = Histog[10,1]+1
                        elif (Rot_theta[y+n][x+m][k]>=90 & 135>Rot_theta[y+n][x+m][k]):# 3rd bin
                            Histog[10,2] = Histog[10,2]+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x+m][k]):# 4th bin
                            Histog[10,3] = Histog[10,3]+1
                        elif (Rot_theta[y+n][x+m][k]>=180 & 225>Rot_theta[y+n][x+m][k]):# 5th bin
                            Histog[10,4] = Histog[10,4]+1
                        elif (Rot_theta[y+n][x+m][k]>=225 & 270>Rot_theta[y+n][x+m][k]):# 6th bin
                            Histog[10,5] = Histog[10,5]+1
                        elif (Rot_theta[y+n][x+m][k]>=270 & 315>Rot_theta[y+n][x+m][k]):# 7th bin
                            Histog[10,6] = Histog[10,6]+1
                        elif (Rot_theta[y+n][x+m][k]>=315 & 360>Rot_theta[y+n][x+m][k]):# 8th bin
                            Histog[10,7] = Histog[10,7]+1
                for m in range (5,9): # 12th histogram
                    for n in range (1,5):
                        if (Rot_theta[y+n][x+m][k]>=0 & 45>Rot_theta[y+n][x+m][k]):# 1st bin
                            Histog[11,0] = Histog[11,0]+1
                        elif (Rot_theta[y+n][x+m][k]>=45 & 90>Rot_theta[y+n][x+m][k]):# 2rd bin
                            Histog[11,1] = Histog[11,1]+1
                        elif (Rot_theta[y+n][x+m][k]>=90 & 135>Rot_theta[y+n][x+m][k]):# 3rd bin
                            Histog[11,2] = Histog[11,2]+1
                        elif (Rot_theta[y+n][x+m][k]>=135 & 180>Rot_theta[y+n][x+m][k]):# 4th bin
                            Histog[11,3] = Histog[11,3]+1
                        elif (Rot_theta[y+n][x+m][k]>=180 & 225>Rot_theta[y+n][x+m][k]):# 5th bin
                            Histog[11,4] = Histog[11,4]+1
                        elif (Rot_theta[y+n][x+m][k]>=225 & 270>Rot_theta[y+n][x+m][k]):# 6th bin
                            Histog[11,5] = Histog[11,5]+1
                        elif (Rot_theta[y+n][x+m][k]>=270 & 315>Rot_theta[y+n][x+m][k]):# 7th bin
                            Histog[11,6] = Histog[11,6]+1
                        elif (Rot_theta[y+n][x+m][k]>=315 & 360>Rot_theta[y+n][x+m][k]):# 8th bin
                            Histog[11,7] = Histog[11,7]+1
                for m in range (4,8): # 13th histogram
                    for n in range (5,9):
                        if (Rot_theta[y+n][x-m][k]>=0 & 45>Rot_theta[y+n][x-m][k]):# 1st bin
                            Histog[12,0] = Histog[12,0]+1
                        elif (Rot_theta[y+n][x-m][k]>=45 & 90>Rot_theta[y+n][x-m][k]):# 2rd bin
                            Histog[12,1] = Histog[12,1]+1
                        elif (Rot_theta[y+n][x-m][k]>=90 & 135>Rot_theta[y+n][x-m][k]):# 3rd bin
                            Histog[12,2] = Histog[12,2]+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x-m][k]):# 4th bin
                            Histog[12,3] = Histog[12,3]+1
                        elif (Rot_theta[y+n][x-m][k]>=180 & 225>Rot_theta[y+n][x-m][k]):# 5th bin
                            Histog[12,4] = Histog[12,4]+1
                        elif (Rot_theta[y+n][x-m][k]>=225 & 270>Rot_theta[y+n][x-m][k]):# 6th bin
                            Histog[12,5] = Histog[12,5]+1
                        elif (Rot_theta[y+n][x-m][k]>=270 & 315>Rot_theta[y+n][x-m][k]):# 7th bin
                            Histog[12,6] = Histog[12,6]+1
                        elif (Rot_theta[y+n][x-m][k]>=315 & 360>Rot_theta[y+n][x-m][k]):# 8th bin
                            Histog[12,7] = Histog[12,7]+1
                for m in range (0,4): # 14th histogram
                    for n in range (5,9):
                        if (Rot_theta[y+n][x-m][k]>=0 & 45>Rot_theta[y+n][x-m][k]):# 1st bin
                            Histog[13,0] = Histog[13,0]+1
                        elif (Rot_theta[y+n][x-m][k]>=45 & 90>Rot_theta[y+n][x-m][k]):# 2rd bin
                            Histog[13,1] = Histog[13,1]+1
                        elif (Rot_theta[y+n][x-m][k]>=90 & 135>Rot_theta[y+n][x-m][k]):# 3rd bin
                            Histog[13,2] = Histog[13,2]+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x-m][k]):# 4th bin
                            Histog[13,3] = Histog[13,3]+1
                        elif (Rot_theta[y+n][x-m][k]>=180 & 225>Rot_theta[y+n][x-m][k]):# 5th bin
                            Histog[13,4] = Histog[13,4]+1
                        elif (Rot_theta[y+n][x-m][k]>=225 & 270>Rot_theta[y+n][x-m][k]):# 6th bin
                            Histog[13,5] = Histog[13,5]+1
                        elif (Rot_theta[y+n][x-m][k]>=270 & 315>Rot_theta[y+n][x-m][k]):# 7th bin
                            Histog[13,6] = Histog[13,6]+1
                        elif (Rot_theta[y+n][x-m][k]>=315 & 360>Rot_theta[y+n][x-m][k]):# 8th bin
                            Histog[13,7] = Histog[13,7]+1
                for m in range (1,5): # 15th histogram
                    for n in range (5,9):
                        if (Rot_theta[y+n][x+m][k]>=0 & 45>Rot_theta[y+n][x+m][k]):# 1st bin
                            Histog[14,0] = Histog[14,0]+1
                        elif (Rot_theta[y+n][x+m][k]>=45 & 90>Rot_theta[y+n][x+m][k]):# 2rd bin
                            Histog[14,1] = Histog[14,1]+1
                        elif (Rot_theta[y+n][x+m][k]>=90 & 135>Rot_theta[y+n][x+m][k]):# 3rd bin
                            Histog[14,2] = Histog[14,2]+1
                        elif (Rot_theta[y+n][x-m][k]>=135 & 180>Rot_theta[y+n][x+m][k]):# 4th bin
                            Histog[14,3] = Histog[14,3]+1
                        elif (Rot_theta[y+n][x+m][k]>=180 & 225>Rot_theta[y+n][x+m][k]):# 5th bin
                            Histog[14,4] = Histog[14,4]+1
                        elif (Rot_theta[y+n][x+m][k]>=225 & 270>Rot_theta[y+n][x+m][k]):# 6th bin
                            Histog[14,5] = Histog[14,5]+1
                        elif (Rot_theta[y+n][x+m][k]>=270 & 315>Rot_theta[y+n][x+m][k]):# 7th bin
                            Histog[14,6] = Histog[14,6]+1
                        elif (Rot_theta[y+n][x+m][k]>=315 & 360>Rot_theta[y+n][x+m][k]):# 8th bin
                            Histog[14,7] = Histog[14,7]+1
                for m in range (5,9): # 16th histogram
                    for n in range (5,9):
                        if (Rot_theta[y+n][x+m][k]>=0 & 45>Rot_theta[y+n][x+m][k]):# 1st bin
                            Histog[15,0] = Histog[15,0]+1
                        elif (Rot_theta[y+n][x+m][k]>=45 & 90>Rot_theta[y+n][x+m][k]):# 2rd bin
                            Histog[15,1] = Histog[15,1]+1
                        elif (Rot_theta[y+n][x+m][k]>=90 & 135>Rot_theta[y+n][x+m][k]):# 3rd bin
                            Histog[15,2] = Histog[15,2]+1
                        elif (Rot_theta[y+n][x+m][k]>=135 & 180>Rot_theta[y+n][x+m][k]):# 4th bin
                            Histog[15,3] = Histog[15,3]+1
                        elif (Rot_theta[y+n][x+m][k]>=180 & 225>Rot_theta[y+n][x+m][k]):# 5th bin
                            Histog[15,4] = Histog[15,4]+1
                        elif (Rot_theta[y+n][x+m][k]>=225 & 270>Rot_theta[y+n][x+m][k]):# 6th bin
                            Histog[15,5] = Histog[15,5]+1
                        elif (Rot_theta[y+n][x+m][k]>=270 & 315>Rot_theta[y+n][x+m][k]):# 7th bin
                            Histog[15,6] = Histog[15,6]+1
                        elif (Rot_theta[y+n][x+m][k]>=315 & 360>Rot_theta[y+n][x+m][k]):# 8th bin
                            Histog[15,7] = Histog[15,7]+1
                descrip=np.concatenate((descrip, Histog), axis=None)
    return descrip
##############################################################sub function



def sift_bocabulary(image_paths, vocab_size):
    bag_of_features = []
    bag_sift=[]
    print("Extract features")
    for path in image_paths:
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (64, 64))
        bag_sift = sift_like_features(gray_img)
        bag_of_features = np.concatenate((bag_of_features,bag_sift),axis=None)
    bag_of_features = bag_of_features.reshape(-1,4*4*8)
    print("Compute vocab")
    list_voca = KMeans(n_clusters=vocab_size,max_iter = 100, random_state=0).fit(bag_of_features)
    vocab=list_voca.cluster_centers_
    return vocab

def get_sift(image_paths):
    vocab = np.load('vocab_sift.npy')
    print('Loaded sift vocab from file.')
    #TODO: Implement this function!
    bag_sift=[]
    image_feats = np.zeros((len(image_paths),len(vocab)))
    print("Extract features")
    for i, path in enumerate(image_paths):
        img = imread(path)
        gray_img = rgb2grey(img)
        gray_img = resize(gray_img, (64, 64))
        bag_sift = sift_like_features(gray_img)
        bag_sift = np.array(bag_sift).reshape(-1,4*4*8)
        dist = cdist(vocab, bag_sift, 'euclidean')
        mdist = np.argmin(dist, axis = 0)
        histo, bins = np.histogram(mdist, range(len(vocab)+1))
        if np.linalg.norm(histo) == 0:
            image_feats[i, :] = histo
        else:
            image_feats[i, :] = histo / np.linalg.norm(histo)
    return image_feats

def sift_like_features(img):
    # make harris corner detector (key point)
    I_x = grad_x(img)
    I_y = grad_y(img)
    I_xx = I_x**2
    I_xy = I_x*I_y
    I_yy = I_y**2

    G1=signal.convolve2d(I_xx, gaus_f(5, 1.4), mode='same')
    G23=signal.convolve2d(I_xy, gaus_f(5, 1.4), mode='same')
    G4=signal.convolve2d(I_yy, gaus_f(5, 1.4), mode='same')


##############################################################sub function
# define kernel of filer
def grad_x(img):
  # derivate operator kernel (sobel)
  kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  return signal.convolve2d(img, kernel_x, mode='same')
def grad_y(img):
  # derivate operator kernel (sobel)
  kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  return signal.convolve2d(img, kernel_y, mode='same')
def gaus_f(size, sigma=1):
  # blur gaussian filter kernel
  size = int(size) //2 #몫
  x, y = np.mgrid[-size:size+1, -size:size+1]
  normal = 1 / (2.0*np.pi*sigma**2)
  g = np.exp(-((x**2+y**2)/(2.0*sigma**2)))*normal
  return g

def ext_Gth(img):
  kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  Ix=signal.convolve2d(img, kernel_x, mode='same')
  Iy=signal.convolve2d(img, kernel_y, mode='same')
  G=np.hypot(Ix,Iy)
  G=G/G.max()*255
  theta = (np.arctan2(Iy,Ix)+np.pi)*180/np.pi # radian to degree
  return (G,theta)
