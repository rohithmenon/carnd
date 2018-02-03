import cv2
PRE_PROCESS_IMAGE_SIZE = 64


class Preprocessor(object):
    def process(self, images):
        for image in images:
            yield cv2.resize(
                image,
                (PRE_PROCESS_IMAGE_SIZE, PRE_PROCESS_IMAGE_SIZE),
                interpolation=cv2.INTER_AREA)
