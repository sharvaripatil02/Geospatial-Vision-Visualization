from skimage.filters import threshold_adaptive
import os
import sys
import numpy as np
import imutils, cv2
import math as m


def find_smear(current_dir):
    all_images = os.listdir(current_dir)

    list_image = [filename for filename in all_images]

    # Calculate mean of all the given images and store in mean_images
    mean_images = calculate_mean_images(current_dir, list_image)

    final_image = cv2.imread(os.path.join(current_dir, list_image[0]))

    final_image = imutils.resize(final_image, width=600)

    mean_images = np.array(np.round(mean_images), dtype=np.uint8)

    # Display image before starting processing
    cv2.imshow('Mean Image', mean_images);
    cv2.waitKey(0)
    cv2.imwrite('mean_img.jpg', mean_images)

    # Convert the mean image found to greyscale
    grey_mean_images = cv2.cvtColor(mean_images, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Greyscaled Mean Image", grey_mean_images);
    cv2.waitKey(0)
    cv2.imwrite('greyscale_img.jpg', grey_mean_images)

    # Applying threshold adaptive algorithm to the greyscaled image
    w = threshold_adaptive(grey_mean_images, 251, offset=10)
    w_image = w.astype("uint8") * 255
    # Use of Canny Edge Detection Algorithm
    canny_image = cv2.Canny(w_image, 75, 200)
    cv2.imshow('Canny Edge Detected Image',canny_image);cv2.waitKey(0)
    cv2.imwrite('Canny_Edge_Detected_img.jpg', canny_image)

    # Finding contours in the image gained after Canny Edge Detection and sorting the contours
    (_, contour_list, _) = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = sorted(contour_list, key=cv2.contourArea, reverse=True)

    final_res = False
    mask_current_image = np.zeros((600, 600, 1), np.float)

    # By using Douglas Pecker algorithm identify if the contour is a smear and mark it on the final_image
    for c in contour_list:

        poly_val = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if abs(cv2.contourArea(c) - 3.14 * m.pow(radius, 2)) < 400 and cv2.contourArea(c) > 400:
            cv2.drawContours(final_image, [poly_val], -1, (0, 255, 0), 2)
            cv2.drawContours(mask_current_image, [poly_val], -1, (255, 255, 255), -1)
            final_res = True

    # Display final image with smear marked
    cv2.imshow('Final Image', final_image);
    cv2.waitKey(0)

    # Display masked image with smear
    cv2.imshow('Masked Final Image', mask_current_image);
    cv2.waitKey(0)

    if (not final_res):
        return False
    cv2.imwrite('FinalResult.jpg', final_image)
    cv2.imwrite('masked_final_img.jpg', mask_current_image)

    return True

# Function to calculate mean image among all given images
def calculate_mean_images(current_dir, list_image):
    mean_images = np.zeros((600, 600, 3), np.float)
    for i in list_image:
        current_image = cv2.imread(os.path.join(current_dir, i))
        current_image = imutils.resize(current_image, width=600)
        current_image = cv2.GaussianBlur(current_image, (3, 3), 0)
        array_image = np.array(current_image, dtype=np.float)
        mean_images = mean_images + array_image / len(list_image)
    return mean_images


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args[0]:
        print "Directory not provided."
        sys.exit(1)

    print ("Directory loaded. Starting Smear detection.")

    smear_detected = find_smear(args[0])

    if (smear_detected):
        print("Detected Smear. Output present in masked_final_image.jpg and FinalResult.jpg")
    else:
        print("No smear present.")
