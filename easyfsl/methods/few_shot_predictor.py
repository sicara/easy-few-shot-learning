import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from easyfsl.methods import FewShotClassifier


class FewShotPredictor :
    """

        This class aims to implement a predictor for a Few-shot classifier.

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

    """

    def __init__(self ,
                 classifier: FewShotClassifier,
                 device,
                 path_to_support_images,
                 n_way,
                 n_shot,
                 input_size=224):

        """
            :param classifier: created and loaded model
            :param device: device to be executed
            :param path_to_support_images: path to creating a support set
            :param n_way: number of classes
            :param n_shot: number of images on each class
            :param input_size: size of image

        """
        self.classifier = classifier
        self.device = device

        self.predict_transformation = tt.Compose([
            tt.Resize((input_size, input_size)),
            tt.ToTensor()
        ])

        self.test_ds = ImageFolder(path_to_support_images, self.predict_transformation)

        self.val_loader = DataLoader(
            self.test_ds,
            batch_size= (n_way*n_shot),
            num_workers=1,
            pin_memory=True
        )

        self.support_images, self.support_labels = next(iter(self.val_loader))



    def predict (self, tensor_normalized_image):
        """

        :param tensor_normalized_image:
        Example of normalized image:

            pil_img = PIL.Image.open(img_dir)

            torch_img = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(pil_img)

            tensor_normalized_image = tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]


        :return:

        Return

        predict = tensor with prediction (mean distance of query image and support set)
        torch_max [1] = predicted class index

        """

        with torch.no_grad():
            self.classifier.eval()
            self.classifier.to(self.device)
            self.classifier.process_support_set(
                self.support_images.to(self.device), self.support_labels.to(self.device))
            pre_predict = self.classifier(tensor_normalized_image.to(self.device))
            predict = pre_predict.detach().data
            torch_max = torch.max(predict,1)
            class_name = self.test_ds.classes[torch_max[1].item()]
            return predict, torch_max[1], class_name
