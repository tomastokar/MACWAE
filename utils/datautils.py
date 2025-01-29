import os
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torchvision import transforms
from torchvision.datasets import CelebA
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from PIL import Image
from pathlib import Path
from typing import Union, Literal, Tuple
from torch.utils.data import Subset
from multivae.data.datasets import MultimodalBaseDataset, DatasetOutput, IncompleteDataset
from multivae.data.datasets.mmnist import MMNISTDataset

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

# from multivae.data.datasets.base import DatasetOutput, IncompleteDataset

def collate_instances(instances: list):

    mods = instances[0].data.keys()
    data = {}
    for m in mods:
        if isinstance(instances[0].data[m], dict):
            data[m] = {}
            keys = instances[0].data[m].keys()            
            for k in keys:
                data[m][k] = torch.stack(
                    [instance.data[m][k] for instance in instances]    
                )
        else:            
            data[m] = torch.stack(
                [instance.data[m] for instance in instances]
            )
            
    return DatasetOutput(data=data)
        
        
def set_inputs_to_device(inputs: DatasetOutput, device: Union[str, torch.device]):
    device_inputs = inputs
    
    if device != 'cpu':
        
        device_inputs.data = {
            k : d.to(device) for k, d in inputs.data.items()
        }
        
        if hasattr(inputs, 'masks'):
            device_inputs.masks = {
                k : m.to(device) for k, m in inputs.masks.items()
            }            
    
    return device_inputs
    

def load_celeba(dir = './data'):
    # Fetch data
    data = {}
    for split in ['train', 'test']:
        data[split] = CelebAttr(dir, split, download=False)

    # Fetch eval dataset
    data['eval'] = CelebAttr(dir, split='valid', download=False)    
    
    return data  


def load_polymnist(dir = './data'):
    # Fetch data
    data = {}
    for split in ['train', 'test']:
        data[split] = MMNISTDataset(dir, split, download=False)

    # Set number of samples
    # for eval and test set
    N = len(data['test'])
    k = N // 2

    # Split test to eval and test
    data['eval'] = Subset(data['test'], range(k))
    data['test'] = Subset(data['test'], range(k, N))  
    
    return data  


def load_mhd(dir = './data/', download = False):
            
    # Fetch data
    data = {}
    for split in ['train', 'test']:
        data[split] = MHD(dir, split = split, download=download)

    # Set number of samples
    # for eval and test set
    N = len(data['test'])
    k = N // 2

    # Split test to eval and test
    data['eval'] = Subset(data['test'], range(k))
    data['test'] = Subset(data['test'], range(k, N))  
    
    return data  


def load_cub(dir = './data/CUB_200_2011'):
    
    # # Image transform
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(64),
    #         transforms.CenterCrop(64),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
    #     ]
    # )
        
    # Fetch data
    data = {}
    for split in ['train', 'test']:
        data[split] = CUB(
            dir, 
            split, 
            # img_transform=transform
        )

    # Set number of samples
    # for eval and test set
    N = len(data['test'])
    k = N // 2

    # Split test to eval and test
    data['eval'] = Subset(data['test'], range(k))
    data['test'] = Subset(data['test'], range(k, N))  
    
    return data  


class CUB(MultimodalBaseDataset):  # pragma: no cover
    """

    A paired text img CUB dataset.

    Args:
        path (str) : The path where the data is saved.
        split (str) : Either 'train' or 'test'. Default: 'train'
        captions_per_image (int): The number of captions text per image. Default: 10
        max_words_in_caption (int): The number of words in the captions. Default: 18
        im_size (Tuple[int]): The desired size of the images. Default: (64, 64)
        img_transform (Transforms): The transformations to be applied to the images. If
            None, nothing is done. Default: None.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        captions_per_image: int = 10,
        max_words_in_caption: int = 18,
        im_size: Tuple[int] = (64, 64),
        img_transform=None,
    ):
        if split not in ["train", "test"]:
            raise AttributeError("Possible values for split are 'train' or 'test'")

        if captions_per_image > 10:
            raise AttributeError("Maximum number of captions per image is 10.")

        self.img_transform = img_transform
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.captions_per_img = captions_per_image
        self.data_path = data_path
        self.split = split
        self.imsize = im_size
        self.max_words_in_captions = max_words_in_caption
        self.tokenizer = RegexpTokenizer(r"\w+")

        self.train_test_split = self._load_train_test_split()
        filenames = self._load_filenames()
        labels = self._load_labels()

        # get train_test_filenames
        train_filenames = filenames[self.train_test_split.split == 1]
        test_filenames = filenames[self.train_test_split.split == 0]
        train_filenames.reset_index(drop=True, inplace=True)
        test_filenames.reset_index(drop=True, inplace=True)

        # get train_test_labels
        train_labels = labels[self.train_test_split.split == 1]
        test_labels = labels[self.train_test_split.split == 0]
        train_labels.reset_index(drop=True, inplace=True)
        test_labels.reset_index(drop=True, inplace=True)

        train_captions = self._load_captions(train_filenames.name)
        test_captions = self._load_captions(test_filenames.name)
        self.bbox = self._load_bbox()

        (
            train_captions_new,
            test_captions_new,
            idxtoword,
            wordtoidx,
            vocab_size,
        ) = self.build_vocab(train_captions, test_captions)

        self.train_filenames = train_filenames
        self.test_filenames = test_filenames
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_captions = train_captions_new
        self.test_captions = test_captions_new
        self.idxtoword = idxtoword
        self.wordtoidx = wordtoidx
        self.vocab_size = vocab_size
        self.labels = None

    def _load_train_test_split(self):
        train_test_split = pd.read_csv(
            os.path.join(self.data_path, "train_test_split.txt"),
            sep='\s+', #delim_whitespace=True,
            header=None,
            names=["id", "split"],
        )
        return train_test_split

    def _load_filenames(self):
        filenames = pd.read_csv(
            os.path.join(self.data_path, "images.txt"),
            sep='\s+', # delim_whitespace=True,
            header=None,
            names=["id", "name"],
        )
        filenames["name"] = filenames["name"].str.replace(".jpg", "")

        return filenames

    def _load_captions(self, filenames):
        all_captions = defaultdict(list)
        for filename in filenames:
            cap_path = os.path.join(f"{self.data_path}", "text", f"{filename}.txt")
            with open(cap_path, "r") as f:
                captions = f.read().splitlines()

                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions[filename].append(tokens_new)
                    cnt += 1
                    if cnt == self.captions_per_img:
                        break
                if cnt < self.captions_per_img:
                    logger.error(
                        "ERROR: the captions for %s less than %d" % (filename, cnt)
                    )

        return all_captions

    def _load_bbox(self):
        data_dir = self.data_path
        bbox_path = os.path.join(data_dir, "bounding_boxes.txt")
        df_bounding_boxes = pd.read_csv(
            bbox_path, 
            sep='\s+', #delim_whitespace=True, 
            header=None
        ).astype(int)

        filepath = os.path.join(data_dir, "images.txt")
        df_filenames = pd.read_csv(
            filepath, 
            sep='\s+', #delim_whitespace=True, 
            header=None
        )
        filenames = df_filenames[1].tolist()

        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox

    def get_imgs(self, img_path, bbox=None):
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if self.img_transform is not None:
            img = self.img_transform(img)

        re_img = transforms.Resize(self.imsize)(img)
        ret = self.normalize(re_img)

        return ret

    def _load_labels(self):
        labels = pd.read_csv(
            os.path.join(self.data_path, "image_class_labels.txt"),
            sep='\s+', #delim_whitespace=True,
            header=None,
            names=["label"],
        )

        labels.reset_index(drop=True, inplace=True)

        return labels

    def build_vocab(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = [cap for cap_list in train_captions.values() for cap in cap_list] + [
            cap for cap_list in test_captions.values() for cap in cap_list
        ]
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = "<end>"
        ixtoword[1] = "<pad>"
        wordtoix = {}
        wordtoix["<end>"] = 0
        wordtoix["<pad>"] = 1
        ix = 2
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = defaultdict(list)
        for cap_key in train_captions.keys():
            for t in train_captions[cap_key]:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                train_captions_new[cap_key].append(rev)

        test_captions_new = defaultdict(list)
        for cap_key in test_captions.keys():
            for t in test_captions[cap_key]:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                test_captions_new[cap_key].append(rev)

        return [
            train_captions_new,
            test_captions_new,
            ixtoword,
            wordtoix,
            len(ixtoword),
        ]

    def get_caption(self, sent_ix, captions):
        # a list of indices for a sentence
        sent_caption = np.asarray(captions[sent_ix]).astype("int64")
        if (sent_caption == 0).sum() > 0:
            logger.error("ERROR: do not need END (0) token", sent_caption)
        num_words = len(sent_caption)
        # pad with 1s (i.e., '<pad>')
        x = np.ones((self.max_words_in_captions, 1), dtype="int64")
        x_len = num_words
        if num_words < self.max_words_in_captions:
            x[:num_words, 0] = sent_caption
            x[num_words, 0] = 0
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[: self.max_words_in_captions]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.max_words_in_captions
        padding_mask = x == 1
        return x, x_len, ~padding_mask

    def __len__(self):
        if self.split == "train":
            return len(self.train_filenames.name)

        return len(self.test_filenames.name)

    def __getitem__(self, index):
        if self.split == "train":
            names = self.train_filenames.name[index]
            captions = self.train_captions[names]
            labels = self.train_labels.label[index]

        else:
            names = self.test_filenames.name[index]
            captions = self.test_captions[names]
            labels = self.test_labels.label[index]

        bbox = self.bbox[names]
        img_path = os.path.join(self.data_path, "images", f"{names}.jpg")
        imgs = self.get_imgs(img_path, bbox=bbox)

        sent_ix = np.random.randint(0, self.captions_per_img)

        # new_sent_ix = index * self.captions_per_img + sent_ix
        caps, cap_len, padding_mask = self.get_caption(sent_ix, captions)

        X = dict(
            img=imgs,
            text=dict(
                tokens=torch.tensor(caps).squeeze(-1),
                padding_mask=torch.FloatTensor(padding_mask).squeeze(-1),
            ),
            label=F.one_hot(torch.tensor(labels) - 1, num_classes=200).float() # Added !!!!
        )

        return DatasetOutput(data=X) #DatasetOutput(data=X, labels=labels)


class CelebAttr(MultimodalBaseDataset):  # pragma: no cover
    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        attributes: Literal["18", "40"] = "40",
        download=False,
    ):
        self.root = root

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

        self.torchvision_dataset = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        if attributes == "18":
            self.attributes_to_keep = [
                4,
                5,
                8,
                9,
                11,
                12,
                15,
                17,
                18,
                20,
                21,
                22,
                26,
                28,
                31,
                32,
                33,
                35,
            ]
        else:
            self.attributes_to_keep = range(40)

        self.attr_to_idx = {
            "Sideburns": 30,
            "Black_Hair": 8,
            "Wavy_Hair": 33,
            "Young": 39,
            "Heavy_Makeup": 18,
            "Blond_Hair": 9,
            "Attractive": 2,
            "5_o_Clock_Shadow": 0,
            "Wearing_Necktie": 38,
            "Blurry": 10,
            "Double_Chin": 14,
            "Brown_Hair": 11,
            "Mouth_Slightly_Open": 21,
            "Goatee": 16,
            "Bald": 4,
            "Pointy_Nose": 27,
            "Gray_Hair": 17,
            "Pale_Skin": 26,
            "Arched_Eyebrows": 1,
            "Wearing_Hat": 35,
            "Receding_Hairline": 28,
            "Straight_Hair": 32,
            "Big_Nose": 7,
            "Rosy_Cheeks": 29,
            "Oval_Face": 25,
            "Bangs": 5,
            "Male": 20,
            "Mustache": 22,
            "High_Cheekbones": 19,
            "No_Beard": 24,
            "Eyeglasses": 15,
            "Bags_Under_Eyes": 3,
            "Wearing_Necklace": 37,
            "Wearing_Lipstick": 36,
            "Big_Lips": 6,
            "Narrow_Eyes": 23,
            "Chubby": 13,
            "Smiling": 31,
            "Bushy_Eyebrows": 12,
            "Wearing_Earrings": 34,
        }

        self.idx_to_attr = {v: k for k, v in self.attr_to_idx.items()}

    def __getitem__(self, index):
        img, target = self.torchvision_dataset[index]
        
        out = dict(image=img)
        for i in self.attributes_to_keep:
            out[self.idx_to_attr[i]]=F.one_hot(target[i], num_classes=2).type(torch.float32)
        
        return DatasetOutput(data=out)

    def __len__(self):
        return self.torchvision_dataset.__len__()


def unstack_tensor(tensor, dim=0):
    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack


class MHD(IncompleteDataset):  # pragma: no cover
    """

    Dataset class for the MHD dataset introduced in the paper:
    'Leveraging hierarchy in multimodal generative models for effective
    cross-modality inference' (Vasco et al, 2021).'

    In this version of the dataset class, we add the possibility to
    simulate missingness in the data, depending on the dataclass.
    For that the missing_probabilities input provides probabilities of missingness for each class,
    for each modality. For instance,

    .. code-block:: python

        >>> missing_probabilities = {
        ...     image = np.zeros(10,).float(),
        ...     audio = np.zeros(10,).float(),
        ...     trajectory = [0.1,0.3,0.4,0.,0.,0.,0.,0.,0.,0.9]
        ... }

    will define a dataset with missing samples in the trajectory modality only in the classes
    0,1,2, et 9.

    Args:

        datapath (str) : Where the data is stored. It must contained the 'mhd_train.pt' file and
            'mhd_test.pt' file.
        split (Literal['train', 'test']) : Split of the data to use. Default to 'train'.
        modalities (list) :  The modalities to use among 'label', 'trajectory', 'image', 'audio'.
            By default, we use all.
        download (bool) : If the dataset is not present at the given path, wether to download it or not.
             Default to False.
        missing_probabilities (dict) : For each modality, the probabilities for each class
            to be missing in the created incomplete dataset. By default, we use no missing data.
        seed (int) : default to 0. You can change the seed to create a different incomplete dataset.


    """

    def __init__(
        self,
        datapath: str,
        split="train",
        modalities: list = ["label", "audio", "trajectory", "image"],
        download=False,
        missing_probabilities=dict(
            label=[0.0] * 10, audio=[0.0] * 10, trajectory=[0.0] * 10, image=[0.0] * 10
        ),
        seed=0,
    ):
        self.data_file = os.path.join(datapath, f"mhd_{split}.pt")
        self.modalities = modalities
        if not os.path.exists(self.data_file):
            if not download:
                raise RuntimeError(
                    f"Dataset not found at path {datapath} and download is set to False. "
                    "Please change the path or set download to True"
                )
            else:
                try:
                    self.__download__(split, datapath)

                except:
                    raise RuntimeError(
                        "gdown must be installed to download the dataset automatically."
                        "Install gdown with "
                        ' "pip install gdown" or download the dataset manually at the following url'
                        "train : https://docs.google.com/uc?export=download&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV"
                        "test : https://docs.google.com/uc?export=download&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6"
                    )

        (
            self._s_data,
            self._i_data,
            self._t_data,
            self._a_data,
            self._traj_normalization,
            self._audio_normalization,
        ) = torch.load(self.data_file)

        self.data = dict()
        if "image" in modalities:
            self.data["image"] = self._i_data
        if "label" in modalities:
            self.data["label"] = one_hot(self._s_data, num_classes=10).float()
        if "trajectory" in modalities:
            self.data["trajectory"] = self._t_data
        if "audio" in modalities:
            self.data["audio"] = self._a_data

        self.labels = self._s_data
        self.n_data = len(self._s_data)
        self.is_incomplete = (
            sum([sum(missing_probabilities[s]) for s in missing_probabilities]) != 0
        )

        if self.is_incomplete:
            # generate the masks
            self.masks = {}
            for i, mod in enumerate(self.data):
                # randomly define the missing samples.
                p = 1 - torch.tensor(missing_probabilities[mod])[self._s_data]
                self.masks[mod] = torch.bernoulli(
                    p, generator=torch.Generator().manual_seed(seed + i)
                ).bool()

            # To be sure, also erase the content of the masked samples
            for k in self.masks:
                reverse_dim_order = tuple(np.arange(len(self.data[k].shape))[::-1])
                self.data[k] = self.data[k].permute(*reverse_dim_order).float()
                # now the batch dimension is last
                self.data[k] *= self.masks[k].float()  # erase missing samples
                # put dimensions back in order
                self.data[k] = self.data[k].permute(*reverse_dim_order)

    def __download__(self, split, datapath):  # pragram : no cover
        import gdown

        if not os.path.exists(datapath):
            os.makedirs(Path(datapath), exist_ok=True)

        if split == "train":
            gdown.download(
                "https://docs.google.com/uc?export=download&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV",
                output=os.path.join(datapath, f"mhd_{split}.pt"),
            )
        else:
            gdown.download(
                "https://docs.google.com/uc?export=download&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6",
                output=os.path.join(datapath, f"mhd_{split}.pt"),
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (t_data, m_data, f_data)
        """

        data = {s: self.data[s][index] for s in self.data}

        if "audio" in data:
            # Audio modality is a 3x32x32 representation, need to unstack!
            audio = unstack_tensor(data["audio"]).unsqueeze(0)
            data["audio"] = audio.permute(0, 2, 1)

        if not self.is_incomplete:
            return DatasetOutput(data=data, labels=self._s_data[index])
        else:
            masks = {s: self.masks[s][index] for s in self.data}
            return DatasetOutput(data=data, labels=self._s_data[index], masks=masks)

    def __len__(self):
        return self.n_data

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization
