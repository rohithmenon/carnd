class ImageFilter(object):
    """
    Image filter apply function. Default implementation is no-op. Subclasses
    will override this method to implement different filters.
    """
    def apply(self, image, context):
        return image

    def __repr__(self):
        return self.__class__.__name__
