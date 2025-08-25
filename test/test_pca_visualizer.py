import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np

from utils.pca_visualizer import PCAVisualizer


class TestPCAVisualizer(unittest.TestCase):
    def test_fit_transform_shape(self):
        data = np.random.randn(100, 5)
        pca = PCAVisualizer(n_components=2)
        proj = pca.fit_transform(data)
        self.assertEqual(proj.shape, (100, 2))

    def test_visualize_saves_file(self):
        data = np.random.randn(50, 4)
        pca = PCAVisualizer(n_components=2)
        out_path = Path('result/test/pca_vis.png')
        pca.visualize(data, save_path=str(out_path))
        self.assertTrue(out_path.is_file())


if __name__ == '__main__':
    unittest.main()
