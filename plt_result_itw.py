import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import savefig
import json

connMat = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])
num_frame = 360

if __name__ == '__main__':
    for i in xrange(309, num_frame + 1):
        dataFile = 'output/dslr_dance1/{:04d}.json' .format(i)
        with open(dataFile) as f:
            joints3d = np.array(json.load(f))   # discard the last joint

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        x = joints3d[:, 0]
        y = -joints3d[:, 2]
        z = joints3d[:, 1]

        ax.scatter(x, y, z, c='r')
        # for j in xrange(17):
        #     ax.text(x[j], y[j], z[j], str(j))
        for conn in connMat:
            ax.plot(x[conn], y[conn], z[conn], c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim([-800, 800])
        ax.set_ylim([-800, 800])
        ax.set_zlim([-800, 800])
        ax.view_init(elev=-162, azim=-147)
        plt.tight_layout()

        savefig('output/dslr_dance1/side_{:04d}.png'.format(i))
        # plt.show()
        plt.close(fig)
