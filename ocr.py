import numpy as np
import cv2
import imutils
import pytesseract
import os
from skimage.filters import threshold_local
from pytesseract import Output
from matplotlib import pyplot as plt

TESSERACT_PATH = os.environ['TESSERACT_PATH']


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def performDocumentAlignment(image):
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 2
    image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # edge detection
    cv2.imwrite('./output_imgs/edged.png', edged)
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # find contours
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imwrite('./output_imgs/contour.png', image)
    warped = four_point_transform(
        orig, screenCnt.reshape(4, 2) * ratio
    )
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    warped = cv2.dilate(warped, kernel, iterations=2)
    warped = cv2.erode(warped, kernel, iterations=2)
    T = threshold_local(warped, 11, offset=10, method='gaussian')
    warped = (warped > T).astype('uint8') * 255
    image = warped

    # perspective transform
    cv2.imwrite('./output_imgs/scanned.png', warped)
    return cv2.imread('./output_imgs/scanned.png')


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def ocr(image):
    gray = get_grayscale(image)
    warped = thresholding(gray)
    warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method='gaussian')
    warped = (warped > T).astype('uint8') * 255
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(warped, config=custom_config)
    return text


def recognizeText(path, alignment=False):
    image = cv2.imread(path)
    if(alignment):
        image = performDocumentAlignment(image)
    return ocr(image)
