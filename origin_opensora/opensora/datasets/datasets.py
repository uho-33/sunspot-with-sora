import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.transforms import Compose, RandomHorizontalFlip

from opensora.datasets.utils import video_transforms
from opensora.registry import DATASETS

# from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
        tokenize_fn=None,
        bucket_class="Bucket",
        return_path=False,
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.tokenize_fn = tokenize_fn
        self.bucket_class = bucket_class
        self.return_path = return_path

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {
            "video": video,
            "num_frames": self.num_frames,
            "height": self.image_size[0],
            "width": self.image_size[1],
            "ar": 1.0,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
            if self.tokenize_fn is not None:
                ret.update({k: v.squeeze(0) for k, v in self.tokenize_fn(ret["text"]).items()})
        if self.return_path:
            ret["path"] = path
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        tokenize_fn=None,
    ):
        super().__init__(
            data_path, num_frames, frame_interval, image_size, transform_name=None, tokenize_fn=tokenize_fn
        )
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def get_path(self, index):
        try:
            index, num_frames, height, width = [int(val) for val in index.split("-")]
            return self.data.iloc[index]["path"]
        except Exception as e:
            print(f"data {index}: {e}")
            return index

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "path": path,
        }
        if self.get_text:
            ret["text"] = sample["text"]
            if self.tokenize_fn is not None:
                ret.update({k: v.squeeze(0) for k, v in self.tokenize_fn(ret["text"]).items()})
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            path = self.get_path(index)
            print(f"data {path}: {e}")
            return None


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret


@DATASETS.register_module()
class VideoClasssificationDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(224, 224),
        data_split="train",
        label_key="preference_mean",
        problem_type="regression",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.label_key = label_key
        self.problem_type = problem_type

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if data_split == "train":
            self.transforms = Compose(
                [
                    video_transforms.ToTensorVideo(),  # TCHW
                    video_transforms.ResizeCrop(image_size),
                    transforms.Normalize(mean, std, inplace=True),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.transforms = Compose(
                [
                    video_transforms.ToTensorVideo(),  # TCHW
                    video_transforms.ResizeCrop(image_size),
                    transforms.Normalize(mean, std, inplace=True),
                ]
            )

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]

        label = None
        if sample.get(self.label_key):  # # label processing, if test won't have it
            label = sample[self.label_key]
            if self.label_key != "label_idx":
                label = label - 1  # normalize score to 0 - 4
            if self.problem_type == "single_label_classification":
                label = round(label)

        # loading
        vframes, vinfo = read_video(path, backend="av")
        video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

        if (
            self.frame_interval == -1
        ):  # instead of random sampling, we use all of the video by uniformly sampling num_frames from the video
            frame_indices = np.linspace(0, vframes[0] - 1, self.num_frames)
            video = vframes[frame_indices]
        else:
            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)  # Sampling video frames, TCHW

        # transform
        video = self.transforms(video)  # T C H W

        if label is not None:
            ret = {"video": video, "label": label, "fps": video_fps, "path": path, "index": index}
        else:
            ret = {"video": video, "fps": video_fps, "path": path, "index": index}

        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class SunObservationDataset(torch.utils.data.Dataset):
    """Dataset for Sun observation sequences with brightness data as text.
    
    Args:
        data_path (str, optional): Not used, kept for compatibility
        num_frames (int): Number of frames to include in each sequence
        frame_interval (int): Interval between frames (for compatibility)
        image_size (tuple): Target size for images (height, width)
        transform_name (str): Name of transform to apply
        tokenize_fn (callable, optional): Function to tokenize text (brightness data)
        time_series_dir (str): Directory containing time series image folders
        brightness_dir (str): Directory containing brightness NPZ files
    """
    
    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(240, 240),
        transform_name="center",
        tokenize_fn=None,
        time_series_dir="dataset/training/figure/360p/L16-S8/",
        brightness_dir="dataset/training/brightness/L16-S8/",
        return_path=False,
        return_filename=False,
    ):
        self.time_series_dir = os.path.expanduser(time_series_dir)
        self.brightness_dir = os.path.expanduser(brightness_dir)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.tokenize_fn = tokenize_fn
        self.return_path = return_path
        self.return_filename = return_filename
        
        # Find all sequence directories that have corresponding brightness data
        self.sequence_dirs = []
        for folder_name in os.listdir(self.time_series_dir):
            folder_path = os.path.join(self.time_series_dir, folder_name)
            brightness_path = os.path.join(self.brightness_dir, f"{folder_name}.npz")
            
            if os.path.isdir(folder_path) and os.path.exists(brightness_path):
                # Check if directory has enough images
                image_files = sorted([f for f in os.listdir(folder_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if len(image_files) >= num_frames:
                    self.sequence_dirs.append(folder_name)
        
        print(f"Found {len(self.sequence_dirs)} valid sun observation sequences")

    def getitem(self, index):
        sequence_name = self.sequence_dirs[index]
        sequence_path = os.path.join(self.time_series_dir, sequence_name)
        brightness_path = os.path.join(self.brightness_dir, f"{sequence_name}.npz")
        
        # Load brightness data from NPZ file
        brightness_data = np.load(brightness_path)['data'] 
        
        # Load image frames
        image_files = sorted([f for f in os.listdir(sequence_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure we have enough frames
        if len(image_files) < self.num_frames:
            raise RuntimeError(f"Not enough frames in {sequence_path}: {len(image_files)} < {self.num_frames}")
        
        # Select frames with interval
        selected_indices = range(0, min(len(image_files), self.num_frames * self.frame_interval), self.frame_interval)
        selected_indices = selected_indices[:self.num_frames]  # Limit to num_frames
        
        # Load the images but keep them as PIL images
        frames = []
        for idx in selected_indices:
            img_path = os.path.join(sequence_path, image_files[idx])
            image = pil_loader(img_path)
            frames.append(image)
        
        # Convert frames to a torch in the format expected by video transforms
        video_array = torch.from_numpy(np.stack([np.array(frame) for frame in frames]))  # [T, H, W, C]
        
        # Transpose from [T, H, W, C] to [T, C, H, W] for compatibility with video transforms
        video_array = video_array.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Apply transforms - this handles conversion to tensor with the right dtype
        transform = self.transforms["video"]
        video = transform(video_array)  # Outputs [T, C, H, W] tensor
        
        # TCHW -> CTHW (match VideoTextDataset format)
        video = video.permute(1, 0, 2, 3)
        
        # Create return dict similar to VideoTextDataset
        ret = {
            "video": video,
            "num_frames": self.num_frames,
            "height": self.image_size[0],
            "width": self.image_size[1],
            "ar": 1.0,
            "fps": 8,  # Using a standard fps value
            "text": brightness_data,  # Brightness data as text
        }
        
        # Apply tokenize function if provided
        if self.tokenize_fn is not None:
            ret.update({k: v.squeeze(0) for k, v in self.tokenize_fn(brightness_data).items()})
        
        if self.return_path:
            ret["path"] = sequence_path
        if self.return_filename:
            ret["filename"] = [image_files[idx] for idx in selected_indices]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                sequence_name = self.sequence_dirs[index]
                print(f"Error loading sequence {sequence_name}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.sequence_dirs)