import os
import glob
from data.base_dataset import BaseDataset, get_transform_3D

class AlignedDataset(BaseDataset):
    """
    This dataset class loads paired 3D medical images.
    It assumes that the directory '/path/to/data/train/A' and '/path/to/data/train/B' contain paired images.
    The pairing is done by assuming the filenames are identical in both directories.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # Get the sorted list of all NIfTI image paths for both domains
        self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot,"*", f'{opt.A_Prefix}.nii.gz')))
        self.B_paths = sorted(glob.glob(os.path.join(opt.dataroot,"*", f'{opt.B_Prefix}.nii.gz')))

        # Ensure that we have the same number of images in both domains for pairing
        assert len(self.A_paths) == len(self.B_paths), \
            f"The number of images in {self.A_paths} and {self.B_paths} must be the same for paired data."

        # Get the MONAI transform pipeline
        self.transform = get_transform_3D(self.opt, self.params)

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        Parameters:
            index (int) -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths, and B_paths
        """
        # Get the file paths for the paired images using the same index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # Create the dictionary that the MONAI transform pipeline expects
        data_dict = {'imgA': A_path, 'imgB': B_path}

        # Apply the MONAI transforms (loading, cropping, etc.)
        processed_dict = self.transform(data_dict)

        # The pix2pix model expects keys 'A' and 'B', so we format the output accordingly
        return processed_dict

    def __len__(self):
        """Return the total number of images (pairs) in the dataset."""
        return len(self.A_paths)

### How to Use It
"""
1.  **Save the Code**: Save the code above into a new file named `data/aligned_3d_dataset.py`.
2.  **Organize Your Data**: Make sure your paired data is structured correctly. For example:
    ```
    /path/to/your/data/
      └── train/
          ├── A/
          │   ├── patient_01.nii.gz
          │   └── patient_02.nii.gz
          └── B/
              ├── patient_01.nii.gz
              └── patient_02.nii.gz
    ```
3.  **Run the Training**: When you run your training script, you will now specify the new dataset mode and the pix2pix model:
    ```bash
    python ./train.py --dataroot /path/to/your/data --model pix2pix --dataset_mode aligned_3d [other_options] 
"""