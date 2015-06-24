import numpy as np
import numba as nb

bags = ['bag1', 'bag2', 'bag3']
bag_of_word_dtype = np.dtype([(bags[0], np.int32),
                 (bags[1], np.int32),
                 (bags[2], np.int32)])
numba_bag_of_word_dtype = nb.from_dtype(bag_of_word_dtype)