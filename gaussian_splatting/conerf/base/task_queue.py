# pylint: disable=[W0621,W0707,W0718,W1203]

import os
import queue
import threading
import time
import logging
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import torch
import numpy as np


class ImageReaderError(Exception):
    """Custom exception for image reader errors"""
    pass


def read_image(
    image_path: str,
    num_channels: int = 3,
    depth_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    normals_path: Optional[str] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Read an image with optional depth, mask, and normals.

    Args:
        image_path: Path to RGB image
        num_channels: Expected number of channels (3 or 4)
        depth_path: Optional path to depth image
        mask_path: Optional path to mask image
        normals_path: Optional path to normals image

    Returns:
        Tuple of (image, depth, mask, normals) tensors
    """
    try:
        # Read RGB image
        image = imageio.imread(image_path)
        if not isinstance(image, np.ndarray):
            raise ImageReaderError(f"Failed to read image at {image_path}")

        image = torch.from_numpy(image).to(torch.uint8)
        image = (image / 255.0).clamp(0.0, 1.0)

        # Handle alpha channel if needed
        if num_channels == 4:
            if image.shape[-1] != 4:
                raise ImageReaderError(
                    f"Expected 4 channels but got {image.shape[-1]} at {image_path}")
            background = torch.tensor([0., 0., 0.], device=image.device)
            image = image[..., :3] * image[..., 3:] + \
                background * (1 - image[..., 3:])

        # Read depth if provided
        depth = None
        if depth_path is not None:
            depth = imageio.imread(depth_path)
            depth = torch.from_numpy(depth).float()
            if depth.ndim != 2:
                depth = depth[..., 0]
            depth = depth[None] / float(2 ** 16)

        # Read mask if provided
        mask = None
        if mask_path is not None:
            mask = imageio.imread(mask_path)
            mask = torch.from_numpy(mask).to(torch.uint8)
            if mask.ndim == 3:
                mask = mask[..., 0]  # Take first channel if multi-channel

        # Read normals if provided
        normals = None
        if normals_path is not None:
            normals = imageio.imread(normals_path)
            normals = torch.from_numpy(normals).float()
            if normals.shape[-1] != 3:
                raise ImageReaderError(
                    f"Normals must have 3 channels, got {normals.shape[-1]} at {normals_path}")
            normals = (normals / 255.0) * 2 - 1  # Convert [0,255] to [-1,1]

        return image, depth, mask, normals

    except Exception as e:
        raise ImageReaderError(f"Error processing {image_path}: {str(e)}")


class TaskQueue(queue.Queue):
    """Thread-safe task queue with worker threads"""

    def __init__(self, max_size: int = 100, max_num_threads: int = 8):
        super().__init__(maxsize=max_size)
        self.max_num_threads = max_num_threads
        self.threads = []
        self.lock = threading.Lock()
        self.start_workers()
        self._shutdown = False

    def worker(self):
        """Worker thread that processes tasks from the queue"""
        while True:
            try:
                # Get task with timeout to allow graceful shutdown
                try:
                    item, args, kwargs = self.get(timeout=0.1)
                except queue.Empty:
                    if self._shutdown:
                        break
                    continue

                if item is None:  # Sentinel value for shutdown
                    self.task_done()
                    break

                try:
                    item(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Task failed: {str(e)}", exc_info=True)
                finally:
                    self.task_done()

            except Exception as e:
                logging.error(f"Worker thread error: {str(e)}", exc_info=True)

    def start_workers(self):
        """Start worker threads"""
        with self.lock:
            for _ in range(self.max_num_threads):
                t = threading.Thread(target=self.worker, daemon=True)
                t.start()
                self.threads.append(t)

    def safe_shutdown(self):
        """Gracefully shutdown all workers"""
        self._shutdown = True
        # Add sentinel values for each worker
        for _ in range(self.max_num_threads):
            self.put((None, None, None))

        # Wait for workers to finish
        for t in self.threads:
            t.join(timeout=1.0)

        self.threads.clear()


class ImageReader(TaskQueue):
    """Multi-threaded image reader supporting RGB, depth, normals, and masks"""

    def __init__(
        self,
        max_size: int = 100,
        max_num_threads: int = 8,
        num_channels: int = 3,
        image_list: Optional[List[str]] = None,
        depth_list: Optional[List[str]] = None,
        mask_list: Optional[List[str]] = None,
        normals_list: Optional[List[str]] = None,
    ):
        super().__init__(max_size, max_num_threads)

        # Validate input lists
        if image_list is None:
            raise ValueError("image_list cannot be None")

        if depth_list is not None and len(depth_list) != len(image_list):
            raise ValueError("depth_list must have same length as image_list")

        if mask_list is not None and len(mask_list) != len(image_list):
            raise ValueError("mask_list must have same length as image_list")

        if normals_list is not None and len(normals_list) != len(image_list):
            raise ValueError(
                "normals_list must have same length as image_list")

        self.image_list = image_list
        self.depth_list = depth_list
        self.mask_list = mask_list
        self.normals_list = normals_list
        self.num_channels = num_channels

        # Use a thread-safe queue for results
        self.result_queue = queue.Queue(maxsize=max_size*2)
        logging.warning(f"ImageReader initialized with {max_size*2} capacity. "
                        "Ensure this fits your memory constraints.")

        self._expected_count = 0  # Total images we expect to process
        # Images that finished processing (success or failure)
        self._completed_count = 0
        self._lock = threading.Lock()  # For thread-safe counters

    def start_loading(self):
        """Start loading all images"""

        with self._lock:
            self._expected_count = len(self.image_list)
            self._completed_count = 0

        for i, image_path in enumerate(self.image_list):
            depth_path = self.depth_list[i] if self.depth_list is not None else None
            mask_path = self.mask_list[i] if self.mask_list is not None else None
            normals_path = self.normals_list[i] if self.normals_list is not None else None

            self.put((
                read_image,
                (image_path, self.num_channels,
                 depth_path, mask_path, normals_path),
                {'index': i}
            ))

    def worker(self):
        """Custom worker that puts results in the result queue"""
        while True:
            try:
                item, args, kwargs = self.get(timeout=0.1)
                if item is None:
                    self.task_done()
                    break

                index = kwargs.get('index')
                try:
                    result = item(*args)
                    self.result_queue.put((index, *result))
                except Exception as e:
                    logging.error(
                        f"Failed to load image at index {index}: {str(e)}")
                    self.result_queue.put((index, None, None, None, None))
                finally:
                    with self._lock:
                        self._completed_count += 1
                    self.task_done()

            except queue.Empty:
                if self._shutdown:
                    break
                continue
            except Exception as e:
                logging.error(f"Worker error: {str(e)}", exc_info=True)

    def get_next(self, timeout: Optional[float] = None):
        """
        Improved get_next that properly handles cases where:
        - Images are still being processed
        - Some images failed to load
        """
        try:
            # First try to get an available result
            result = self.result_queue.get(timeout=timeout)
            self.result_queue.task_done()
            return result
        except queue.Empty:
            # Check if we should expect more results
            with self._lock:
                if self._completed_count >= self._expected_count:
                    return None  # All images processed, none available

            # If we get here, images are still being processed
            if timeout is not None and timeout > 0:
                # Wait a bit more if caller specified a timeout
                time.sleep(0.01)
                return self.get_next(timeout=timeout - 0.01)
            return None

    def has_next(self):
        """
        More reliable check for remaining images by:
        1. Checking if we expect more images to come
        2. OR if there are ready results in the queue
        """
        # If we have results ready, return True immediately
        if not self.result_queue.empty():
            return True

        with self._lock:
            return self._completed_count < self._expected_count

    def remaining_images(self) -> int:
        """
        Returns the accurate count of images still to be processed or available.
        """
        with self._lock:
            remaining = self._expected_count - self._completed_count
        return remaining + self.result_queue.qsize()

    def safe_shutdown(self):
        """Gracefully shutdown the reader"""
        super().safe_shutdown()
        # Clear the result queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                break


def test_one_loop(image_list: List, depth_list: List):
    reader = ImageReader(
        image_list=image_list,
        depth_list=depth_list,
        max_num_threads=16,
    )

    reader.start_loading()

    print(f'total images: {reader.remaining_images()}')

    while reader.has_next():
        index, image, depth, mask, normals = reader.get_next()
        if image is not None:
            # Process the image
            print(
                f'image index: {index}, image shape: {image.shape}, ' +
                f'depth shape: {depth.shape}, ' +
                f'num images in queue: {reader.remaining_images()}')

    reader.safe_shutdown()


def test_multi_loops(image_list: List, depth_list: List):
    reader = None
    itr = 0
    for itr in range(200):
        if (itr % len(image_list) == 0) or (reader is None) or (not reader.has_next()):
            if reader is not None:
                if reader.has_next():
                    print(f'[Warning] Reader still has {reader.remaining_images()} ' + \
                          'images remaining before restart!')
                else:
                    print('[INFO] Reader has finished all images before restart.')

            if reader is None:
                reader = ImageReader(
                    image_list=image_list,
                    depth_list=depth_list,
                    max_num_threads=16,
                )
            print(f'New loop {itr}, restarting reader...')
            reader.start_loading()

        print(f'total images: {reader.remaining_images()}')

        assert reader.has_next(), "Reader should have next image!"

        index, image, depth, mask, normals = reader.get_next()
        if image is not None:
            # Process the image
            print(
                f'image index: {index}, image shape: {image.shape}, ' +
                f'depth shape: {depth.shape}, ' +
                f'num images in queue: {reader.remaining_images()}')

        itr += 1

    reader.safe_shutdown()


if __name__ == "__main__":
    image_root_dir = "/nfs/camera2/yuchen/datasets/mipnerf360/test"
    image_list, depth_list = [], []

    NUM_IMAGES = 23
    for i in range(NUM_IMAGES):
        image_name = f"{i:03d}.png"
        image_path = os.path.join(image_root_dir, "rgb", image_name)
        depth_path = os.path.join(image_root_dir, "depth", image_name)
        image_list.append(image_path)
        depth_list.append(depth_path)

    test_one_loop(image_list, depth_list)

    test_multi_loops(image_list, depth_list)
