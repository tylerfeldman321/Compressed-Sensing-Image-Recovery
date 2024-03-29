# This is for ECE580: Intro to machine learning Spring 2020 in Duke
# This is translated to Python from show_chanWeights.m file provided by Prof. Li by 580 TAs

# import ext libs
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.signal import medfilt2d, convolve2d
from tqdm import tqdm
from scipy.ndimage import uniform_filter


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


RESULTS_DIR = 'Output'


def img_read(filename):
    """
    load the input image into a matrix
    :param filename: name of the input file
    :return: a matrix of the input image
    Examples: imgIn = imgRead('lena.bmp')
    """
    imgIn = plt.imread(filename)
    return np.float64(imgIn)


def img_show(img_out, title):
    """
    show the image saved in a matrix
    :param img_out: a matrix containing the image to show
    :param title: title of plot
    :return: None
    """
    img_out = np.uint8(img_out)
    img = plt.imshow(img_out, cmap='gray')
    if title:
        plt.title(title)
    plt.show()


def img_save(img_out, title):
    """
    show the image saved in a matrix
    :param img_out: a matrix containing the image to show
    :param title: title of plot
    :return: None
    """
    plt.figure()
    img_out = np.uint8(img_out)
    plt.imshow(img_out, cmap='gray')
    if title:
        plt.title(title)
    plt.savefig(os.path.join(RESULTS_DIR, title + '.png'))


def split_img_into_blocks(img, k):
    """
    Splits an image into n kxk blocks
    :param img: 2D numpy array
    :param k: The side length of the square of the blocks to extract from the image. Both image dimensions must be
    divisible by this number.
    :return: A 3D numpy array that has shape (n x k x k), where n is the number of blocks that are able to be extracted
    from the image. The blocks are ordered by column and then row. For example, the zeroth block is in the top left of
    the image and the 1st block is shifted by one column to the right compared with the zeroth block.
    """
    assert img.shape[0] % k == 0, "Can't divide img rows into sections of k"
    assert img.shape[1] % k == 0, "Can't divide img columns into sections of k"
    img_blocks = img.reshape(img.shape[0] // k, k, -1, k).swapaxes(1, 2).reshape(-1, k, k)
    return img_blocks


def sample_pixels_from_block(block, S, copy=False):
    """
    Sample pixels from a square block of pixels from the original image
    :param block: 2D numpy array
    :param S: How many pixels to sample from the block
    :param copy: Whether to make a copy of the block before modifying. If no, then it will modify the original datastructure
    :return: Returns block with only the sampled pixels, and everything else as 0. If not copy, then will modify in
    place as well.
    """
    assert S <= block.size, 'S should not be larger than the number of pixels in the block'
    num_pixels_to_remove = block.size - S
    block_ravel = block.ravel() if not copy else block.copy().ravel()

    indices = np.arange(0, block_ravel.shape[0], 1)
    np.random.shuffle(indices)
    unknown_pixels_indices = indices[0:num_pixels_to_remove]
    unknown_pixels_indices = np.sort(unknown_pixels_indices)

    unknown_pixels = block_ravel[unknown_pixels_indices].copy()

    sampled_pixel_indices = [index for index in indices if index not in unknown_pixels_indices]
    sampled_pixel_indices = np.sort(sampled_pixel_indices)
    sampled_pixels = block_ravel[sampled_pixel_indices]

    block_ravel[unknown_pixels_indices] = np.min(block_ravel) # 0
    block_for_viewing = block_ravel.reshape(block.shape)

    return block_for_viewing, sampled_pixels, sampled_pixel_indices, unknown_pixels, unknown_pixels_indices


def combine_block_to_get_image(img_blocks, shape):
    rows, cols = shape
    n, nrows, ncols = img_blocks.shape
    img = img_blocks.reshape(rows // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(rows, cols)
    return img


def make_T(n=8):
    T = np.zeros((n*n, n*n))
    P, Q = n, n
    for y in np.arange(1, n+1):
        for x in np.arange(1, n+1):
            for u in np.arange(1, n+1):
                for v in np.arange(1, n+1):
                    T[(y-1) * n + (x-1), (u-1) * n + (v-1)] = get_val_T(x, y, u, v, P, Q)
    T[:, 0] = 1
    return T


def estimate_dct_coeffs(T, pixel_values, lam):
    clf = Lasso(alpha=lam)
    clf.fit(T[:, 1:], pixel_values)
    coeffs = clf.coef_
    intercept = clf.intercept_
    coeffs = np.concatenate((intercept, coeffs), axis=None)
    return coeffs


def cos_basis_function(x_y, u_v, P_Q):
    return np.cos((np.pi * (2*x_y - 1) * (u_v - 1)) / (2 * P_Q))


def get_val_T(x, y, u, v, P, Q):
    alpha, beta = calc_alpha(u, P), calc_beta(v, Q)
    return alpha * beta * np.cos((np.pi * (2*x - 1) * (u -1)) / (2 * P)) * np.cos((np.pi * (2*y - 1) * (v -1)) / (2 * Q))


def calc_beta(v, Q):
    return np.sqrt(1 / Q) if (v == 1) else np.sqrt(2 / Q)


def calc_alpha(u, P):
    return np.sqrt(1 / P) if (u == 1) else np.sqrt(2 / P)


def calc_alpha_vec(u, P):
    return [np.sqrt(1 / P) if (u_val == 1) else np.sqrt(2 / P) for u_val in u]


def calc_beta_vec(v, Q):
    return [np.sqrt(1 / Q) if (v_val == 1) else np.sqrt(2 / Q) for v_val in v]


# Read image
# Split into blocks
# For each block:
#   Sample the test pixels from each block, store them somewhere
#   Determine DCT coeffs from by using cross validation with the remaining pixels
#   Use all S samples to find DCT coeffs
#   Recover block
# Combine them
# Median filter


class ImageRecover():
    def __init__(self, img_path='data/fishing_boat.bmp', block_size=8, S_values=np.arange(30, 61, 10), lambda_val_list=np.logspace(-7, 5, num=30), num_cv_folds=10):
        self.T = make_T(block_size)
        self.block_size = block_size
        self.num_cv_folds = num_cv_folds
        self.S_values = S_values
        self.S = S_values[0]
        self.img = img_read(img_path)
        self.lambda_val_list = lambda_val_list

    def recover_block(self, block, verbose=False, plot_lambdas=False):
        block_for_viewing, sampled_pixels, sampled_pixel_indices_in_T, unknown_pixels, unknown_pixels_indices = sample_pixels_from_block(block, self.S, copy=True)
        sampled_block = block_for_viewing.copy()

        indicies = np.arange(len(sampled_pixels))

        lambda_val_errors = {}
        for lambda_val in self.lambda_val_list:

            fold_size = self.S // self.num_cv_folds

            cv_fold_errors = []
            for fold in range(self.num_cv_folds):
                np.random.shuffle(indicies)
                train_indices = indicies[fold_size:]
                val_indices = indicies[:fold_size]

                train_pixel_indices_in_T = sampled_pixel_indices_in_T[train_indices]
                val_pixel_indices_in_T = sampled_pixel_indices_in_T[val_indices]

                train_pixels = sampled_pixels[train_indices]
                val_pixels = sampled_pixels[val_indices]

                T_sampled_train = self.T[train_pixel_indices_in_T, :]
                dct_coeffs = estimate_dct_coeffs(T_sampled_train, train_pixels, lambda_val)

                T_sampled_val = self.T[val_pixel_indices_in_T, :]
                val_pixels_estimated = np.matmul(T_sampled_val, np.expand_dims(dct_coeffs, axis=0).T)[:, 0]
                val_pixels_estimated = np.array(list(map(int, val_pixels_estimated)))

                cv_error = mse(val_pixels, val_pixels_estimated)
                cv_fold_errors.append(cv_error)

            lambda_val_avg_error = np.array(cv_fold_errors).sum() / len(cv_fold_errors)
            lambda_val_errors[lambda_val] = lambda_val_avg_error

        best_lambda_value = float(min(lambda_val_errors, key=lambda_val_errors.get))

        if verbose:
            print(f'\nBest Lambda Value: {best_lambda_value}')

        if plot_lambdas:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(list(lambda_val_errors.keys()), list(lambda_val_errors.values()))
            ax.plot(best_lambda_value, lambda_val_errors[best_lambda_value], 'ro', label=f'Best Lambda Value; MSE {lambda_val_errors[best_lambda_value]}')
            ax.set_xscale('log')
            ax.grid(True)
            ax.set_xlabel('Lambda')
            ax.set_ylabel('10-fold Cross Validation MSE')
            plt.legend(loc='best')
            plt.title('MSE vs Lambda for Chosen Block in Fishing Boat')
            fig.tight_layout()
            plt.show()

        # Estimate the unknown (test) pixels
        T_sampled = self.T[sampled_pixel_indices_in_T, :]
        dct_coeffs = estimate_dct_coeffs(T_sampled, sampled_pixels, best_lambda_value)

        T_unknown = self.T[unknown_pixels_indices, :]
        unknown_pixels_estimated = np.matmul(T_unknown, np.expand_dims(dct_coeffs, axis=0).T)[:, 0]
        unknown_pixels_estimated = np.maximum(0, unknown_pixels_estimated)
        unknown_pixels_estimated = np.array(list(map(int, unknown_pixels_estimated)))

        block_for_viewing_flattened = block_for_viewing.ravel()
        block_for_viewing_flattened[unknown_pixels_indices] = unknown_pixels_estimated
        block_for_viewing = np.reshape(block_for_viewing_flattened, block_for_viewing.shape)
        return block_for_viewing, sampled_block

    def recover_image(self):
        errors = {}
        errors_before_filtering = {}
        errors_mean = {}
        errors_gaussian3 = {}
        for S_value in self.S_values:
            self.S = S_value
            img_blocks = split_img_into_blocks(self.img, self.block_size)

            recovered_blocks, sampled_blocks = img_blocks.copy(), img_blocks.copy()

            recovered_blocks[0], sampled_blocks[0] = self.recover_block(img_blocks[0])
            img_show(sampled_blocks[0], 'Sampled Block in Fishing Boat')
            img_show(img_blocks[0], 'Block in Fishing Boat')

            for i in tqdm(range(len(img_blocks))):
                recovered_blocks[i], sampled_blocks[i] = self.recover_block(img_blocks[i])

            # Combine recovered blocks, apply median filter, and show recovered image
            img_sampled = combine_block_to_get_image(sampled_blocks, self.img.shape)

            recovered_img = combine_block_to_get_image(recovered_blocks, self.img.shape)
            mse_error_before_filtering = mse(self.img, recovered_img)
            img_save(recovered_img, f'Recovered Image Before Filtering for S={self.S}, MSE={mse_error_before_filtering:.2f}')
            print('MSE for Recovered Image Before Filtering:', mse_error_before_filtering)

            recovered_img_filtered = median_filter(recovered_img)
            mse_error = mse(self.img, recovered_img_filtered)
            img_save(recovered_img_filtered, f'Recovered Image After Median Filtering for S={self.S}, MSE={mse_error:.2f}')
            print('MSE for Recovered Image After Filtering:', mse_error)

            errors_before_filtering[self.S] = mse_error_before_filtering
            errors[self.S] = mse_error

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(list(errors.keys()), list(errors.values()), label='MSE After Median Filtering')
        ax.plot(list(errors_before_filtering.keys()), list(errors_before_filtering.values()), label='MSE Before Median Filtering')
        ax.grid(True)
        ax.set_xlabel('Value of S')
        ax.set_ylabel('MSE Values Between Recovered Image and True Image')
        plt.title('MSE vs S')
        fig.tight_layout()
        plt.legend(loc='best')
        plt.show()


def median_filter(image, kernel_size=3):
    return medfilt2d(image, kernel_size=kernel_size)


def mean_filter(image, kernel_size=3):
    return uniform_filter(image, size=3, mode='nearest')


def gaussian_filter(image, kernel_size=3):
    kernel = (1/16) * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])

    return convolve2d(image, kernel, mode='same')


def mse(y_hat, y):
    return np.sum(np.square(y - y_hat)) / y.size


if __name__ == '__main__':
    ir = ImageRecover()
    ir.recover_image()
