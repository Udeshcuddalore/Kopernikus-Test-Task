import unittest
from organise_dataset import OrganiseDataset


class OrganiseDatasetTest(unittest.TestCase):
    def test_get_camera_id_indices(self) -> None:
        """
        This Test case to validate the 'get_camera_id_indices' method of the OrganiseDataset class.

        This method verifies that the 'get_camera_id_indices' method
        returns the expected camera ID indices dictionary.
        """
        Test_folder_path = "path to the folder"
        Organise = OrganiseDataset(Test_folder_path)
        camera_id_indices, _ = Organise.get_camera_id_indices()
        self.assertEqual(len(camera_id_indices.keys()), 4)


if __name__ == "__main__":
    unittest.main()
