import numpy as np
v = np.load('data/mnist550+letters.logit.npy')
np.save('data/mnist550_part.logit', v[:550, :10])