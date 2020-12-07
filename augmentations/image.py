import tensorflow as tf

class ImageAugmentation:
    @staticmethod
    def random_flip_horizontal(image):
        return tf.image.random_flip_left_right(image)
    @staticmethod
    def random_flip_vertical(image):
        return tf.image.random_flip_up_down(image)
    @staticmethod
    def random_brightness(image):
        return tf.image.random_brightness(image, max_delta=0.1)
    @staticmethod
    def random_hue(image):
        return tf.image.random_hue(image, 0.08)
    @staticmethod
    def random_saturation(image):
        return tf.image.random_saturation(image, 0.6, 1.6)
    @staticmethod
    def random_contrast(image):
        return tf.image.random_contrast(image, 0.7, 1.3)
    @staticmethod
    def random_zoom(image):
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))
        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]
        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, 
                box_ind=np.zeros(len(scales)), crop_size=(32, 32))
            # Return a random crop
            return crops[tf.random_uniform(shape=[], minval=0, 
                maxval=len(scales), dtype=tf.int32)]
        choice = tf.random_uniform(shape=[], minval=0., maxval=1., 
            dtype=tf.float32)
        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(image))