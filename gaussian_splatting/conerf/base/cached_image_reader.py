# pylint: disable=[W0621,W0707,W0718,W1203]

import os
import queue
import threading
import time
import logging
import random

from typing import List, Optional, Tuple, Dict, Union, Iterator
from dataclasses import dataclass
from enum import Enum
import copy

import imageio.v2 as imageio
import torch
import torch.utils.data
import numpy as np


class ImageReaderError(Exception):
    """Custom exception for image reader errors"""
    pass


def read_image(
    image_path: str,
    num_channels: int = 3,
    depth_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    normals_path: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Read an image with optional depth, mask, and normals.
    Returns CPU tensors.
    """
    try:
        # Read RGB image
        image = imageio.imread(image_path)
        if not isinstance(image, np.ndarray):
            raise ImageReaderError(f"Failed to read image at {image_path}")

        image = torch.from_numpy(image).float() / 255.0

        # Handle alpha channel if needed
        if num_channels == 4:
            if image.shape[-1] != 4:
                raise ImageReaderError(
                    f"Expected 4 channels but got {image.shape[-1]} at {image_path}")
            background = torch.tensor([0., 0., 0.])
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
            mask = torch.from_numpy(mask).float() / 255.0
            if mask.ndim == 3:
                mask = mask[..., 0]

        # Read normals if provided
        normals = None
        if normals_path is not None:
            normals = imageio.imread(normals_path)
            normals = torch.from_numpy(normals).float()
            if normals.shape[-1] != 3:
                raise ImageReaderError(
                    f"Normals must have 3 channels, got {normals.shape[-1]} at {normals_path}")
            normals = (normals / 255.0) * 2 - 1

        return image, depth, mask, normals

    except Exception as e:
        raise ImageReaderError(f"Error processing {image_path}: {str(e)}")


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset that loads images from disk.
    Always returns CPU tensors.
    """
    
    def __init__(
        self,
        image_list: List[str],
        num_channels: int = 3,
        depth_list: Optional[List[str]] = None,
        mask_list: Optional[List[str]] = None,
        normals_list: Optional[List[str]] = None,
    ):
        self.image_list = image_list
        self.depth_list = depth_list
        self.mask_list = mask_list
        self.normals_list = normals_list
        self.num_channels = num_channels
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        """Returns (image, depth, mask, normals) as CPU tensors"""
        image_path = self.image_list[idx]
        depth_path = self.depth_list[idx] if self.depth_list is not None else None
        mask_path = self.mask_list[idx] if self.mask_list is not None else None
        normals_path = self.normals_list[idx] if self.normals_list is not None else None
        
        return read_image(
            image_path, 
            self.num_channels, 
            depth_path, 
            mask_path, 
            normals_path
        )


class CachedImageDataLoader:
    """
    A DataLoader-like image reader with caching support.
    
    Features:
    - Multi-threaded loading like PyTorch DataLoader
    - Optional RAM caching of loaded images
    - Optional VRAM caching for frequently accessed images
    - Maintains index mapping correctly
    """
    
    def __init__(
        self,
        image_list: List[str],
        num_channels: int = 3,
        depth_list: Optional[List[str]] = None,
        mask_list: Optional[List[str]] = None,
        normals_list: Optional[List[str]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        ram_cache_size: int = 200,  # Number of images to keep in RAM
        vram_cache_size: int = 50,   # Number of images to keep in VRAM
        pin_memory: bool = True,
        device: str = 'cuda',
    ):
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.ram_cache_size = ram_cache_size
        self.vram_cache_size = vram_cache_size
        self.pin_memory = pin_memory
        self.device = device
        
        # Create dataset
        self.dataset = ImageDataset(
            image_list=image_list,
            num_channels=num_channels,
            depth_list=depth_list,
            mask_list=mask_list,
            normals_list=normals_list,
        )
        
        # Create indices
        self.indices = list(range(len(self.dataset)))
        if shuffle:
            self._shuffle_indices()
        
        self.current_idx = 0
        
        # Caches
        self.ram_cache: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self.vram_cache: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self.access_count: Dict[int, int] = {}
        self.cache_lock = threading.RLock()
        
        # Prefetch queue
        self.prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        self.prefetch_threads = []
        self.prefetch_running = True
        self._start_prefetch_threads()
        
        # For iteration
        self._iter_count = 0
        
        logging.warning(
            f"CachedImageDataLoader initialized with {len(self.dataset)} images, "
            f"RAM cache: {ram_cache_size}, VRAM cache: {vram_cache_size}"
        )
    
    def _shuffle_indices(self):
        """Shuffle indices for next epoch"""
        random.shuffle(self.indices)
        self.current_idx = 0
    
    def _start_prefetch_threads(self):
        """Start background threads for prefetching"""
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._prefetch_worker,
                name=f"Prefetch-{i}",
                daemon=True
            )
            t.start()
            self.prefetch_threads.append(t)
    
    def _prefetch_worker(self):
        """Worker thread that prefetches images from disk"""
        while self.prefetch_running:
            try:
                # Get next index to prefetch
                if self.current_idx >= len(self.indices):
                    time.sleep(0.1)
                    continue
                
                with threading.Lock():
                    if self.current_idx >= len(self.indices):
                        continue
                    idx = self.indices[self.current_idx]
                    self.current_idx += 1
                
                # Check if already in cache
                with self.cache_lock:
                    if idx in self.vram_cache or idx in self.ram_cache:
                        continue
                
                # Load from disk
                data = self.dataset[idx]
                
                # Add to RAM cache (with LRU eviction)
                with self.cache_lock:
                    self._add_to_ram_cache(idx, data)
                
                # Put in prefetch queue for main thread
                self.prefetch_queue.put((idx, data), timeout=1.0)
                
            except queue.Full:
                continue
            except Exception as e:
                logging.error(f"Prefetch worker error: {str(e)}")
    
    def _add_to_ram_cache(self, idx: int, data: Tuple[torch.Tensor, ...]):
        """Add data to RAM cache with LRU eviction"""
        if idx in self.ram_cache:
            return
        
        # Evict if needed
        if len(self.ram_cache) >= self.ram_cache_size:
            # Find least recently accessed
            lru_idx = min(
                self.ram_cache.keys(),
                key=lambda x: self.access_count.get(x, 0)
            )
            del self.ram_cache[lru_idx]
        
        # Pin memory if requested
        if self.pin_memory:
            data = tuple(d.pin_memory() if d is not None else None for d in data)
        
        self.ram_cache[idx] = data
    
    def _promote_to_vram(self, idx: int):
        """Promote image from RAM to VRAM"""
        with self.cache_lock:
            if idx not in self.ram_cache or idx in self.vram_cache:
                return
            
            # Evict from VRAM if needed
            if len(self.vram_cache) >= self.vram_cache_size:
                lru_idx = min(
                    self.vram_cache.keys(),
                    key=lambda x: self.access_count.get(x, 0)
                )
                del self.vram_cache[lru_idx]
            
            # Move to VRAM
            data = self.ram_cache[idx]
            vram_data = tuple(
                d.to(self.device, non_blocking=True) if d is not None else None 
                for d in data
            )
            
            self.vram_cache[idx] = vram_data
    
    def __iter__(self):
        """Return iterator"""
        self._iter_count = 0
        return self
    
    def __next__(self):
        """Get next item"""
        if self._iter_count >= len(self.indices):
            self._shuffle_indices()
            raise StopIteration
        
        idx = self.indices[self._iter_count]
        self._iter_count += 1
        
        # Update access count
        with self.cache_lock:
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
        
        # Try to get from VRAM cache first
        with self.cache_lock:
            if idx in self.vram_cache:
                data = self.vram_cache[idx]
                # Return copies to prevent modification
                return (idx,) + tuple(
                    d.clone() if d is not None else None for d in data
                )
        
        # Try to get from RAM cache
        with self.cache_lock:
            if idx in self.ram_cache:
                data = self.ram_cache[idx]
                # Promote frequently accessed to VRAM
                if self.access_count[idx] > 3:
                    self._promote_to_vram(idx)
                
                # Move to device and return copies
                device_data = tuple(
                    d.to(self.device, non_blocking=False).clone() 
                    if d is not None else None 
                    for d in data
                )
                return (idx,) + device_data
        
        # Not in cache, wait for prefetch or load directly
        try:
            # Wait for prefetch (with timeout)
            prefetched_idx, data = self.prefetch_queue.get(timeout=5.0)
            if prefetched_idx != idx:
                # Wrong index, put back and load directly
                self.prefetch_queue.put((prefetched_idx, data))
                data = self.dataset[idx]
            # Add to cache
            with self.cache_lock:
                self._add_to_ram_cache(idx, data)
        except queue.Empty:
            # Prefetch timeout, load directly
            data = self.dataset[idx]
            with self.cache_lock:
                self._add_to_ram_cache(idx, data)
        
        # Move to device and return copies
        device_data = tuple(
            d.to(self.device, non_blocking=False).clone() 
            if d is not None else None 
            for d in data
        )
        return (idx,) + device_data
    
    def __len__(self):
        return len(self.dataset)
    
    def shutdown(self):
        """Clean shutdown"""
        self.prefetch_running = False
        for t in self.prefetch_threads:
            t.join(timeout=1.0)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'ram_cache_size': len(self.ram_cache),
                'ram_cache_capacity': self.ram_cache_size,
                'vram_cache_size': len(self.vram_cache),
                'vram_cache_capacity': self.vram_cache_size,
                'total_images': len(self.dataset),
            }
    
    def clear_cache(self):
        """Clear all caches"""
        with self.cache_lock:
            self.ram_cache.clear()
            self.vram_cache.clear()
            self.access_count.clear()


# For backward compatibility
ImageReader = CachedImageDataLoader