import os
import argparse
import cv2

from typing import List, Dict, Tuple
from imaging_interview import (
    preprocess_image_change_detection,
    compare_frames_change_detection,
)
from dataclasses import dataclass


@dataclass
class OrganiseDataset:
    folder_path: str
    """
    The OrganiseDataset Class
    Args:
        folder_path (str): The path to folder containing images.
        In dataset, filenames  use  the  following  formatting:
        - c%camera_id%-%timestamp%.png
    Raises:
        FileNotFoundError: If the specified folder path does not exist.
    """
    def get_camera_id_indices(self) -> Tuple[Dict[str, List[int]], List[str]]:
        """
        Get camera ID indices for the images in the folder.
        Returns a tuple containing a dictionary mapping camera IDs (based on prefixes)
        to their corresponding indices in the sorted list of files

        Returns:
            Tuple[Dict[str, List[int]], List[str]]: A tuple containing a dictionary and a list.
                - The dictionary maps camera IDs (based on prefixes) to a list of indices
                  representing the positions of the corresponding filenames in the list.
                - The list contains the filenames sorted in ascending order.

        Raises:
            FileNotFoundError: If the specified folder path does not exist.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        Image_files = list()
        camera_id_indices = dict()
        camera_ids = set()
        for file in os.listdir(self.folder_path):
            if file.endswith(".png"):
                Image_files.append(file)
        Image_files = sorted(Image_files)
        for file_name in Image_files:
            if "-" in file_name:
                camera_ids.add(file_name.split("-")[0])
            elif "_" in file_name:
                camera_ids.add(file_name.split("_")[0])
        for ids in sorted(camera_ids):
            indices = [
                index
                for index, filename in enumerate(Image_files)
                if filename.startswith(ids)
            ]
            camera_id_indices[ids] = indices
        return camera_id_indices, Image_files

    def remove_duplicates(
        self,
        gaussian_blur_radius_list: List[int] = [3, 5, 7],
        min_contour_area: int = 2000,
        min_score: int = 25000,
    ) -> None:
        """
        This function removes duplicated images from the folder
        based on output of "compare_frames_change_detection" function.
        Args:
            gaussian_blur_radius_list (List[int]): List of Gaussian blur radii used
            for preprocessing images. Defaults to [3, 5, 7].
            min_contour_area (int): Minimum contour area threshold. Defaults to 1000.
            min_score (int): Minimum score threshold. Defaults to 15000.
        Returns:
            None
        Raises:
            FileNotFoundError: If the specified folder path does not exist.
        """
        camera_id_indices, Image_files = self.get_camera_id_indices()
        non_essential_data = dict()
        for camera_id, img_indices in camera_id_indices.items():
            remove_indices = list()
            for img_index in img_indices:
                file_path = os.path.join(self.folder_path, Image_files[img_index])
                read_image = cv2.imread(file_path)
                if read_image is None:
                    remove_indices.append(img_index)
                    continue
                if img_index == img_indices[0]:
                    prev_frame = preprocess_image_change_detection(
                        read_image, gaussian_blur_radius_list
                    )
                else:
                    next_frame = preprocess_image_change_detection(
                        read_image, gaussian_blur_radius_list
                    )
                    if prev_frame.shape != next_frame.shape:
                        prev_frame = cv2.resize(
                            prev_frame, (next_frame.shape[1], next_frame.shape[0])
                        )
                    score, _, _ = compare_frames_change_detection(
                        prev_frame, next_frame, min_contour_area
                    )
                    if score < min_score:
                        remove_indices.append(img_index)
                    else:
                        prev_frame = next_frame
            non_essential_data[camera_id] = remove_indices
        if non_essential_data:
            self.remove_ids(non_essential_data, Image_files)
        else:
            pass

    def remove_ids(self, non_essential_data, Image_files) -> None:
        """
        This function removes the non-essential images based on data in non_essential_data.

        Args:
            non_essential_data (dict): A dictionary of camera IDs,lists of image indices.
            filenames (list): The list of filenames.
        Returns:
            None
        Raises:
            FileNotFoundError: If the specified folder path does not exist.
        """
        for camera_id, remove_indices in non_essential_data.items():
            count = 0
            for img_index in remove_indices:
                file_path = os.path.join(self.folder_path, Image_files[img_index])
                os.remove(file_path)
                count += 1
            print(f"In camera:{camera_id} total {count} images are repeated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path", type=str, help="path to folder containing images"
    )
    args = parser.parse_args()
    Organise = OrganiseDataset(args.folder_path)
    Organise.remove_duplicates()
