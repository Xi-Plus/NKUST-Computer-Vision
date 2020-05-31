# https://github.com/MarcoForte/bayesian-matting

import argparse

import cv2
import numpy as np
from numba import jit


class Node(object):
    def __init__(self, matrix, w):
        W = np.sum(w)
        self.w = w
        self.X = matrix
        self.left = None
        self.right = None
        self.mu = np.einsum('ij,i->j', self.X, w) / W
        diff = self.X - np.tile(self.mu, [np.shape(self.X)[0], 1])
        t = np.einsum('ij,i->ij', diff, np.sqrt(w))
        self.cov = (t.T @ t) / W + 1e-5 * np.eye(3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.cov)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]


def clustFunc(S, w, minVar=0.05):
    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))

    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.cov)

    return np.array(mu), np.array(sigma)


def split(nodes):
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[np.logical_not(idx)], C_i.w[np.logical_not(idx)])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes


def gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


@jit(nopython=True, cache=True)
def get_window(img, h, w, size):
    H, W, c = img.shape
    half_N = size // 2
    result = np.zeros((size, size, c))
    h_min = max(0, h - half_N)
    h_max = min(H, h + (half_N + 1))
    w_min = max(0, w - half_N)
    w_max = min(W, w + (half_N + 1))
    r_h_min = half_N - (h - h_min)
    r_h_max = half_N + (h_max - h)
    r_w_min = half_N - (w - w_min)
    r_w_max = half_N + (w_max - w)

    result[r_h_min:r_h_max, r_w_min:r_w_max] = img[h_min:h_max, w_min:w_max]
    return result


@jit(nopython=True, cache=True)
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, sigma_C, alpha_init, maxIter, minLike):
    '''
    Solves for F,B and alpha that maximize the sum of log
    likelihoods at the given pixel C.
    input:
    mu_F - means of foreground clusters (for RGB, of size 3x#Fclusters)
    Sigma_F - covariances of foreground clusters (for RGB, of size
    3x3x#Fclusters)
    mu_B,Sigma_B - same for background clusters
    C - observed pixel
    alpha_init - initial value for alpha
    maxIter - maximal number of iterations
    minLike - minimal change in likelihood between consecutive iterations

    returns:
    F,B,alpha - estimate of foreground, background and alpha
    channel (for RGB, each of size 3x1)
    '''
    I = np.eye(3)
    FMax = np.zeros(3)
    BMax = np.zeros(3)
    alphaMax = 0
    maxlike = - np.inf
    invsgma2 = 1 / sigma_C**2
    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])
        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308
            while True:
                # solve for F,B
                A11 = invSigma_Fi + I * alpha**2 * invsgma2
                A12 = I * alpha * (1 - alpha) * invsgma2
                A22 = invSigma_Bj + I * (1 - alpha)**2 * invsgma2
                A = np.vstack((np.hstack((A11, A12)), np.hstack((A12, A22))))
                b1 = invSigma_Fi @ mu_Fi + C * (alpha) * invsgma2
                b2 = invSigma_Bj @ mu_Bj + C * (1 - alpha) * invsgma2
                b = np.atleast_2d(np.concatenate((b1, b2))).T

                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                # solve for alpha

                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T - B).T @ (F - B)) / np.sum((F - B)**2)))[0, 0]
                # # calculate likelihood
                L_C = - np.sum((np.atleast_2d(C).T - alpha * F - (1 - alpha) * B)**2) * invsgma2
                L_F = (- ((F - np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F - np.atleast_2d(mu_Fi).T)) / 2)[0, 0]
                L_B = (- ((B - np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B - np.atleast_2d(mu_Bj).T)) / 2)[0, 0]
                like = (L_C + L_F + L_B)
                #like = 0

                if like > maxlike:
                    alphaMax = alpha
                    maxLike = like
                    FMax = F.ravel()
                    BMax = B.ravel()

                if myiter >= maxIter or abs(like - lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
    return FMax, BMax, alphaMax


def main(path_in, path_trimap, path_out):
    N = 25
    sigma = 8
    minN = 1

    print('path_in', path_in)
    print('path_trimap', path_trimap)
    print('path_out', path_out)

    img_in = cv2.imread(path_in)
    h, w, channel = img_in.shape
    print('img_in shape', h, w, channel)
    img_in = img_in / 255

    img_trimap = cv2.imread(path_trimap, cv2.IMREAD_GRAYSCALE)
    print('img_trimap shape', img_trimap.shape)

    mask_fg = (img_trimap == 255)
    mask_bg = (img_trimap == 0)
    mask_unknown = np.logical_not(np.logical_or(mask_fg, mask_bg))

    img_fg = img_in * np.repeat(mask_fg[:, :, np.newaxis], 3, axis=2)
    img_bg = img_in * np.repeat(mask_bg[:, :, np.newaxis], 3, axis=2)

    alpha = np.zeros((h, w))
    alpha[mask_fg] = 1
    alpha[mask_bg] = 0
    alpha[mask_unknown] = np.nan
    cnt_unknown = np.sum(mask_unknown)
    print('cnt_unknown', cnt_unknown)

    F = np.zeros(img_in.shape)
    B = np.zeros(img_in.shape)

    mask_temp = mask_unknown.astype(np.uint8)

    gaussian_weights = gauss2d((N, N), sigma)
    gaussian_weights = gaussian_weights / np.max(gaussian_weights)

    cnt_done = 0
    # while cnt_done < cnt_unknown:
    for i in range(100):
        if cnt_done == cnt_unknown:
            break
        mask_temp = cv2.erode(mask_temp, np.ones((3, 3)), iterations=1)
        mask_run = np.logical_and(np.logical_not(mask_temp), mask_unknown)  # 未知區域最靠近前景/背景的區域

        H, W = np.nonzero(mask_run)
        print('to_run', len(H), cnt_done)
        for h, w in zip(H, W):
            # if cnt_done % 100 == 1:
            #     print(cnt_done, cnt_unknown)
            pixel = img_in[h, w]

            a = get_window(alpha[:, :, np.newaxis], h, w, N)[:, :, 0]

            f_pixels = get_window(img_fg, h, w, N)
            f_weights = (a**2 * gaussian_weights).ravel()

            f_pixels = np.reshape(f_pixels, (N * N, 3))
            posInds = np.nan_to_num(f_weights) > 0
            f_pixels = f_pixels[posInds, :]
            f_weights = f_weights[posInds]

            # Take surrounding foreground pixels
            b_pixels = get_window(img_bg, h, w, N)
            b_weights = ((1 - a)**2 * gaussian_weights).ravel()

            b_pixels = np.reshape(b_pixels, (N * N, 3))
            posInds = np.nan_to_num(b_weights) > 0
            b_pixels = b_pixels[posInds, :]
            b_weights = b_weights[posInds]

            # print(f_weights)
            # print(b_weights)

            # if not enough data, return to it later...
            if len(f_weights) < minN or len(b_weights) < minN:
                # print('skip')
                continue
            # Partition foreground and background pixels to clusters (in a weighted manner)
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            # print(mu_f)
            # print(sigma_f)
            # exit()
            alpha_init = np.nanmean(a.ravel())
            # Solve for F,B for all cluster pairs
            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, pixel, 0.01, alpha_init, 50, 1e-6)
            img_fg[h, w] = f.ravel()
            img_bg[h, w] = b.ravel()
            alpha[h, w] = alphaT
            mask_unknown[h, w] = 0
            cnt_done += 1

    alpha *= 255
    # alpha[alpha < 128] = 0
    # alpha[alpha >= 128] = 255
    cv2.imwrite(path_out, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('trimap')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args.input, args.trimap, args.output)
