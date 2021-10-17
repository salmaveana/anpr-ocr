# Import the necessary packages
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class ANPR:

    def __init__(self, minAR=1.3, maxAR=3, debug=False):

        # Store minimum and maximum rectangular aspect ratio
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    # Display results
    def debug_imshow(self, title, image, waitKey=False):

        # Check to see if debug mode is enabled
        if self.debug:
            
            # Show image with the supplied title
            cv2.imshow(title, image)

            # Check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    # Find license plate candidates
    # Receives: grayscale image
    #           number of plates to store
    def locate_license_plate_candidates(self, gray, keep=5):

        # Perform blackhat morphological operation to allow
        # reveal of dark regions (text) on light backgrounds (plate)
        # Aspect ratio 12x7 is ideal for mexican plates
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (12,7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)

        # Find light regions
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)

        # Compute Scharr gradient representation of blackhat image
        # in the x-direction
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                dx=1, dy=0, ksize=-1)
        # Scale result back to range [0, 255]
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)

        # Blur gradient representation
        # Apply a close operation
        # Threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)

        # Perform erosions and dilatations to clean up the
        # thresholded image (denoise)
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Grad Erode/Dilate", thresh)

        # Take the bitwise "AND" between the threshold result 
        # and the light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Final", thresh, waitKey=True)

        # Find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sort them by size in descending order
        # Keep only the largest ones
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        # Return list of contours
        return cnts

    # Find the most likely contour containig a license plate
    # Receives: grayscale image
    #           license plate contour candidates
    #           boolean to eliminate contours that touch the edge
    def locate_license_plate(self, gray, candidates, clearBorder=False):

        # Initialize the license plate contour and ROI
        lpCnt = None
        roi = None

        # Loop over license plate candidates
        for i, c in enumerate(candidates):

            print("\n")
            print("[INFO] Candidate: ", i)

            # Compute bounding box of the contour
            # Use bounding box to derive aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            print("Width: ", w)
            print("Height: ", h)
            ar = w / float(h)

            print("[INFO] Aspect ratio")
            print(str(ar) + " >= " + str(self.minAR))
            print(str(ar) + " <= " + str(self.maxAR)) 
            print("[INFO] coordinates...")
            print("Start x", x)
            print("Start y", y)
            print("End x", str(x+w))
            print("End y", str(y+h))

            # Check is aspect ratio is rectangular
            if w > h and ar >= self.minAR and ar <= self.maxAR:
            #if ar >= self.minAR:

                print("[INFO] Match aspect ratio...")

                # Store license plate contour
                lpCnt = c

                # Extract the license plate from grayscale image
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # Check if any foreground pixels touching the edge
                # should be cleared
                if clearBorder: 
                    roi = clear_border(roi)

                # Display debbuging info
                self.debug_imshow("License Plate", licensePlate)
                self.debug_imshow("ROI", roi, waitKey=True)

                # Break from loop
                break

        # Return a 2-tuple:
        #   license plate ROI 
        #   contour associated with the ROI
        return (roi, lpCnt)

    # Send image through Tesseract OCR
    # Set Tesseract operation mode as 7 (image as single text line)
    def build_tesseract_options(self, psm=7):

        # Use Tesseract only with OCR alphanumerics characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)

        # Set PSM mode
        options += " --psm {}".format(psm)

        # Return build options string
        return options

    # Bring all components together
    # Instantiate ANPR object
    # Making a single function call
    # Receives: 3-channel color image
    #           tesseract page segmentation mode 
    #           flag to eliminate contours that touch the edge
    def find_and_ocr(self, image, psm=7, clearBorder=False):

        # Initialize the license plate text
        lpText = None 

        # Convert input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Locate all license plate candidates 
        candidates = self.locate_license_plate_candidates(gray)

        # Process candidates getting *actual* license plate
        (lp, lpCnt) = self.locate_license_plate(gray, candidates,
                clearBorder=clearBorder)

        # Only OCR the license plate if the license plate ROI is not
        # empty
        if lp is not None:

            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)

        # Return a 2-tuple
        # The OCR'd license plate text
        # The contour associated with the license plate region
        return (lpText, lpCnt)
        
