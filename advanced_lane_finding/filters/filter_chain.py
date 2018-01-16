from filters.image_filter import ImageFilter


class FilterChain(ImageFilter):
    """
    Chain multiple filters to create a composite filter. Runs the filters
    one after the other feeding output of the previous filter to next.
    """
    def __init__(self, filters):
        self.filters = filters

    def apply(self, image, context):
        processed = image
        for image_filter in self.filters:
            if context and context.debug_fn:
                context.debug_fn.start_chain()
            processed = image_filter.apply(processed, context)
            if context and context.debug_fn:
                context.debug_fn.processed(processed, image_filter)
            if context and context.debug_fn:
                context.debug_fn.end_chain()
        return processed
