import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union


class PCAVisualizer:
    """Simple PCA dimensionality reduction and visualization utility.

    This class computes principal components using Singular Value Decomposition
    and provides helper methods to transform data and visualize the result.
    """

    def __init__(self, n_components: int = 2) -> None:
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3 for visualization")
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, data: Union[np.ndarray, list]) -> "PCAVisualizer":
        """Fit PCA on data.

        Args:
            data (Union[np.ndarray, list]): Input data of shape (N, D).

        Returns:
            PCAVisualizer: self
        """
        data = np.asarray(data)
        self.mean_ = data.mean(axis=0)
        X = data - self.mean_
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        return self

    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """Project data onto principal components.

        Args:
            data (Union[np.ndarray, list]): Data to project.

        Returns:
            np.ndarray: Projected data of shape (N, n_components).
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCAVisualizer must be fitted before calling transform")
        data = np.asarray(data)
        X = data - self.mean_
        return np.dot(X, self.components_.T)

    def fit_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """Fit PCA on data and return projected result."""
        return self.fit(data).transform(data)

    def visualize(
        self,
        data: Union[np.ndarray, list],
        labels: Optional[Union[np.ndarray, list]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Fit PCA on data and create a scatter plot.

        Args:
            data (Union[np.ndarray, list]): Input data of shape (N, D).
            labels (Optional[Union[np.ndarray, list]], optional): Labels for coloring points. Defaults to None.
            save_path (Optional[Union[str, Path]], optional): If provided, save the plot to this path instead of showing it.

        Returns:
            np.ndarray: Projected data of shape (N, n_components).
        """
        proj = self.fit_transform(data)

        if self.n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if labels is None:
                ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2])
            else:
                sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=labels, cmap="viridis")
                fig.colorbar(sc)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        else:
            plt.figure()
            if labels is None:
                plt.scatter(proj[:, 0], proj[:, 1])
            else:
                sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="viridis")
                plt.colorbar(sc)
            plt.xlabel("PC1")
            plt.ylabel("PC2")

        plt.title("PCA Visualization")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return proj
