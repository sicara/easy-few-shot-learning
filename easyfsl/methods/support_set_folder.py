
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class SupportSetFolder (ImageFolder):
    """
        This class aims to help you to create a support set based on a folder.

        The few shot classifiers need a support set that will be used for
        calculating the distance between the support set and the query image.

        To load the support we have used an ImageFolder Dataset, which needs to have the following structure:

        folder:
          |_ class_name_folder_1:
                 |_ image_1
                 |_  …
                 |_ image_n
          |_ class_name_folder_2:
                 |_ image_1
                 |_  …
                 |_ image_n

        The folder must contain the same number of images per class, being the total images (n_way * n_shot).

        There must be n_way folders with n_shot images per folder.

        Example of use:

        predict_transformation = tt.Compose([
            tt.Resize((224, 224)),
            tt.ToTensor()
        ])
        support_set = SupportSetFolder(root= path_to_support_images,
                                       transform=predict_transformation,
                                       device=DEVICE)
        with torch.no_grad():
            few_shot_classifier.eval()
            few_shot_classifier.process_support_set(support_set.get_images(), support_set.get_labels())
            predictions = few_shot_classifier(normed_torch_img.to(device))
            class_names = support_set.classes
            class_index = torch.max(predictions, 1)
            class_name = classes[class_index[1].item()]

    """
    def __init__(self, root:str, transform, device,  n_way, n_shot):

        """
            :param device: device to be executed
            :param root: path to creating a support set
            :param n_way: number of classes
            :param n_shot: number of images on each class

        """

        super().__init__(root = root,transform = transform)

        self.val_loader = DataLoader(
            self,
            batch_size=(n_way * n_shot),
            num_workers=1,
            pin_memory=True
        )
        self.device = device
        self.support_images, self.support_labels = next(iter(self.val_loader))

    def get_images(self):
        """
            Returns the support set images on the selected device
        """
        return self.support_images.to(self.device)

    def get_labels(self):
        """
            Returns the support set labels on the selected device
        """
        return self.support_labels.to(self.device)
