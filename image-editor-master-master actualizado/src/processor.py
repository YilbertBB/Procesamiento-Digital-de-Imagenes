import numpy as np
import cv2
from matplotlib import pyplot as plt


class Processor:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.actions_history = []

    def _open_image(self, filename=None):
        """Open an image and return it"""
        try:
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        except:
            raise Exception("Could not open image: {}".format(filename))

    def open_image(self, filename=None):
        """Open a image to edit it"""
        try:
            _image = self._open_image(filename)
            if _image is not None:
                self.image = _image
                self.original_image = self.image.copy()
                self.actions_history.insert(0, self.image)
            else:
                raise Exception("Could not open image: {}".format(filename))
        except:
            raise Exception("Could not open image: {}".format(filename))

    def revert_image(self):
        """Set image to default"""
        self.actions_history.append(self.image)
        self.image = self.original_image.copy()

    def undo_image(self):
        """Undo image action"""
        if len(self.actions_history) > 0:
            self.image = self.actions_history.pop(-1)

    def save_image(self, location):
        """Save image in a image file"""
        try:
            cv2.imwrite(location, self.image)
        except:
            raise Exception("Specify the file format!")

    def flip_image(self, option):
        """Flip the imagen"""
        self.actions_history.append(self.image)

        if option == 'HORIZONTAL':
            self.image = cv2.flip(self.image, 1)
        elif option == 'VERTICAL':
            self.image = cv2.flip(self.image, 0)

    def grayscale_filter(self):
        """Escala de grises"""
        self.actions_history.append(self.image)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    def brightness_filter(self, bias):
        """Modify the brightness"""
        self.actions_history.append(self.image)
        self.image = cv2.addWeighted(self.image, 1, np.zeros(
            self.image.shape, self.image.dtype), 0, bias*500-250)

    def contrast_filter(self, gain):
        """Modify the contrast"""
        self.actions_history.append(self.image)
        self.image = cv2.addWeighted(
            self.image, gain*10, np.zeros(self.image.shape, self.image.dtype), 0, 0)

    def negative_filter(self):
        """Set image to negative"""
        self.actions_history.append(self.image)

        self.image = 255 - self.image

    def rotate_image(self, degrees):
        """Rotate the imagen"""
        try:
            temp_img = self.image.copy()
            height = temp_img.shape[0]
            width = temp_img.shape[1]
            m = cv2.getRotationMatrix2D((width // 2, height // 2), degrees, 1)
            self.actions_history.append(self.image)
            self.image = cv2.warpAffine(temp_img, m, (width, height))
        except:
            raise Exception("Degrees need to be a real number!")

    def move_image(self, option, amount):
        """Move the imagen"""
        try:
            height = self.image.shape[0]
            width = self.image.shape[1]
            if option == 'HORIZONTAL':
                m = np.float32([[1, 0, amount], [0, 1, 0]])
            elif option == 'VERTICAL':
                m = np.float32([[1, 0, 0], [0, 1, amount]])
            else:
                m = np.float32([[1, 0, 0], [0, 1, 0]])

            self.actions_history.append(self.image)
            self.image = cv2.warpAffine(self.image, m, (width, height))
        except:
            raise Exception("Amount need to be a real number!")

    def resize_image(self, option, amount):
        """Resize the imagen"""
        try:
            h, w, _ = self.image.shape
            if option == 'WIDTH':
                w += amount
            elif option == 'HEIGHT':
                h += amount

            try:
                self.actions_history.append(self.image)
                self.image = cv2.resize(
                    self.image, (w, h), interpolation=cv2.INTER_CUBIC)
            except:
                raise Exception("The size must be greater than 0!")
        except:
            raise Exception("Amount need to be a real number!")

    def operations(self, option, another=None):
        """Do operations"""
        if another is None:
            return
        another = self._open_image(another)
        h, w, _ = self.image.shape
        another = cv2.resize(another, (w, h), interpolation=cv2.INTER_CUBIC)
        self.actions_history.append(self.image)
        if option == 'Sum':
            self.image = cv2.add(self.image, another)
        elif option == 'Subtract':
            self.image = cv2.subtract(self.image, another)
        elif option == 'AND':
            self.image = cv2.bitwise_and(self.image, another)
        elif option == 'OR':
            self.image = cv2.bitwise_or(self.image, another)

    def suavizado(self, option):
        """Do Smoothing"""
        self.actions_history.append(self.image)
        if option == 'Paso Bajo':
            kernel = np.ones((5, 5), np.float32)/25
            self.image = cv2.filter2D(self.image, -1, kernel)
            plt.subplot(121), plt.imshow(self.image), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(self.image), plt.title('Promediada')
            plt.show()
            plt.xticks([]), plt.yticks([])

        elif option == 'Promedio':
            self.image = cv2.blur(self.image, (3, 3))
            plt.subplot(121), plt.imshow(self.image), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(self.image), plt.title('Difuminada')
            plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Gaussiano':
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
            plt.subplot(121), plt.imshow(self.image), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(self.image), plt.title('Difuminada')
            plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Mediana':
            self.image = cv2.medianBlur(self.image, 5)
            plt.subplot(121), plt.imshow(self.image), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(self.image), plt.title('Difuminada')
            plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Sobel':
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)

            plt.subplot(2, 2, 1), plt.imshow(self.image, cmap='gray')
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 3), plt.imshow(self.image, cmap='gray')
            plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 4), plt.imshow(self.image, cmap='gray')
            plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Laplaciano':
            self.image = cv2.Laplacian(self.image, cv2.CV_64F)

            plt.subplot(2, 2, 1), plt.imshow(self.image, cmap='gray')
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 2), plt.imshow(self.image, cmap='gray')
            plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
            plt.show()

    def transformacionesM(self, option):
        """Do Morphological Transformations"""
        self.actions_history.append(self.image)
        if option == 'Erosión':
            kernel = np.ones((7, 7), np.uint8)
            self.image = cv2.erode(self.image, kernel, iterations=1)

        elif option == 'Dilatación':
            kernel = np.ones((7, 7), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=1)

        elif option == 'Apertura':
            kernel = np.ones((7, 7), np.uint8)
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

        elif option == 'Cierre':
            kernel = np.ones((7, 7), np.uint8)
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

    def umbralizacion(self, option):
        """Do Thresholding"""
        self.actions_history.append(self.image)
        if option == 'Fija':
            ret, thresh1 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_BINARY_INV)
            ret, thresh3 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_TRUNC)
            ret, thresh4 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_TOZERO)
            ret, thresh5 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_TOZERO_INV)

            titles = ['Original Image', 'BINARY',
                      'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
            images = [self.image, thresh1, thresh2, thresh3, thresh4, thresh5]
            miArray = np.arange(6)
            for i in miArray:
                plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Adaptativa':
            self.image = cv2.medianBlur(self.image, 5)

            ret, th1 = cv2.threshold(
                self.image, 127, 255, cv2.THRESH_BINARY)
            th2 = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
            th3 = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

            titles = ['Original Image', 'Global Thresholding (v = 127)',
                      'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
            images = [self.image, th1, th2, th3]
            miArray = np.arange(4)
            for i in miArray:
                plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

        elif option == 'Binarización de Otsu':
            # global thresholding
            ret1, th1 = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)

            # Otsu's thresholding
            ret2, self.image = cv2.threshold(
                self.image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(self.image, (5, 5), 0)
            ret3, th3 = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # plot all the images and their histograms
            images = [self.image, 0, th1, self.image, 0, th2, blur, 0, th3]
            titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
                      'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
                      'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
            miArray = np.arange(3)
            for i in miArray:
                plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
                plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
                plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
                plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
                plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
                plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
            plt.show()

    def connected_component_label(self):
        self.actions_history.append(self.image)
        # Converting those pixels with values 1-127 to 0 and others to 1
        self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents()
        num_labels, labels = cv2.connectedComponents(self.image)

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        # Showing Original Image
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Orginal Image")
        plt.show()

        # Showing Image after Component Labeling
        plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Image after Component Labeling")
        plt.show()
