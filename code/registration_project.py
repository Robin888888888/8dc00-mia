"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output


def intensity_based_registration(image1,image2,type='rigid',evaluation_metric='corr',learning_rate=0.001,number_of_iterations=150):

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/'+image1+'.tif')
    Im = plt.imread('../data/image_data/'+image2+'.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    if type=='rigid':
        x = np.array([0, 0, 0])
    if type=='affine':
        x = np.array([0, 1, 1, 0, 0, 0, 0])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation

    if type=='rigid':
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)
    elif type=='affine':
        fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)

    # the learning rate
    mu = learning_rate

    # number of iterations
    num_iter = number_of_iterations

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)
    if type=='affine':
        params = np.full((num_iter, 7), np.nan)
    elif type=='corr':
        params = np.full((num_iter,3),np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()
    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x = x+g*mu
 
        if type=='rigid':
            S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)
        elif type=='affine':
            S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
        if evaluation_metric=='mi':
            p=reg.joint_histogram(I,Im_t)
            MI=reg.mutual_information_e(p)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        if evaluation_metric=='mi':
            similarity[k] = MI
        elif evaluation_metric=='corr':
            similarity[k] = S
        params[k]=x
        learning_curve.set_ydata(similarity)


        display(fig)

    max_similarity=max(similarity)
    iteration_max,k=np.where(similarity==max_similarity)
    print('best iteration:\n{}'.format(iteration_max))
    print('highest similarity:\n{}'.format(max_similarity))
    print('best parameters:\n{}'.format(params[iteration_max,:]))