import numpy as np
from filters.image_filter import ImageFilter


class BinaryAndFilter(ImageFilter):
    def __init__(self, filters):
        self.filters = filters

    def apply(self, image, context):
        processed = []
        for image_filter in self.filters:
            if context and context.debug_fn:
                context.debug_fn.start_chain()
            processed_img = image_filter.apply(image, context)
            processed.append(processed_img)
            if context and context.debug_fn:
                context.debug_fn.processed(processed_img, image_filter)
            if context and context.debug_fn:
                context.debug_fn.end_chain()
        mask = np.all(np.stack(processed), 0)
        binary_output = np.zeros_like(mask, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        filter_str = ', '.join([str(f) for f in self.filters])
        return 'OR({})'.format(filter_str)


class BinaryOrFilter(ImageFilter):
    def __init__(self, filters):
        self.filters = filters

    def apply(self, image, context):
        processed = []
        for image_filter in self.filters:
            if context and context.debug_fn:
                context.debug_fn.start_chain()
            processed_img = image_filter.apply(image, context)
            processed.append(processed_img)
            if context and context.debug_fn:
                context.debug_fn.processed(processed_img, image_filter)
            if context and context.debug_fn:
                context.debug_fn.end_chain()
        mask = np.any(np.stack(processed), 0)
        binary_output = np.zeros_like(mask, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        filter_str = ', '.join([str(f) for f in self.filters])
        return 'OR({})'.format(filter_str)


class BinaryWeightedAndFilter(ImageFilter):
    def __init__(self, filters, fraction):
        self.filters = filters
        self.threshold = fraction * len(filters)

    def apply(self, image, context):
        processed = []
        for image_filter in self.filters:
            if context and context.debug_fn:
                context.debug_fn.start_chain()
            processed_img = image_filter.apply(image, context)
            processed.append(processed_img)
            if context and context.debug_fn:
                context.debug_fn.processed(processed_img, image_filter)
            if context and context.debug_fn:
                context.debug_fn.end_chain()
        mask = np.sum(np.stack(processed), 0) >= self.threshold
        binary_output = np.zeros_like(mask, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        filter_str = ', '.join([str(f) for f in self.filters])
        return 'WeightedAnd({})'.format(filter_str)


# Weighted AND of all filters. This allows for a flexible binary operation. If fraction is 1.0, then
# we get AND of all filters and fraction = 1/len(filters) will make it OR of all filters.
def weighted_and_filters(filters, fraction):
    return BinaryWeightedAndFilter(filters, fraction)


# AND of all filters.
def and_filters(filters):
    return BinaryAndFilter(filters)


# OR of all filters.
def or_filters(filters):
    return BinaryOrFilter(filters)
