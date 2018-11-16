from __future__ import print_function
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

class DataInfo(object):
    
    def __init__(self):
        self.info = pd.read_csv("data/TrainIJCNN2013/gt.txt", sep=";", header = None)
        self.info.columns = ["img_name","topmost", "upmost", "rightmost",  "downmost", "class" ]
        self.signs = pd.read_csv("data/signs.txt", sep="=", header=None)
    
    def get_sign_name(self, ind):
        return self.signs.loc[ind][1]

    def class_info(self):
        counts = self.info.groupby(['class']).count().reset_index('class')
        print("Total Number of classes: ", counts.count()['class'])
        counts['sign_name'] = counts['class'].apply(self.get_sign_name)
        count_20 = counts.sort_values('class', ascending=False).head(20)
        print("Distribution of 20 classes with Number of samples")
        plt.figure(figsize=(10,5))
        sns.set(font_scale=1.3)
        sns.barplot(x='img_name',y='sign_name',data=count_20,orient='o')
        plt.xticks(rotation=90)
        plt.ylabel('Sign Name')
        plt.xlabel('Training Samples');
        plt.tight_layout()
        plt.show()
    
    def display_transforms_of_images(self, loaders):
        plt.figure()
        j=1
        for loader in loaders:
            for i, curr_data in enumerate(loader):
                curr_img = curr_data['img'][1]
                curr_img = np.transpose(curr_img.numpy(), (1, 2, 0))
                plt.subplot(2,3,j)
                j = j + 1
                plt.imshow(curr_img)
                #curr_img.show()
                #plt.pause(0.010)
                if i == 0:
                    break;
        plt.show()
        
    def display_label_info(self, loaders):
        labels = [] 
        #images = []
        for loader in loaders:
            for i, curr_data in enumerate(loader):
                curr_data_labels = curr_data['label'].numpy()
                #curr_data_imgs = curr_data['img'].numpy()
                for i in range(len(curr_data_labels)):
                    labels.append(curr_data_labels[i])
                    #images.append(curr_data_imgs[i])
        
        fig = plt.figure()
        fig.set_dpi(130)
        plt.bar([0,1],[labels.count(0), labels.count(1)])
        plt.xlabel('Classes')
        plt.xticks(range(0,2,1))
        plt.ylabel('Number of examples per class')
        plt.show()
        
class DataAugmentation(object):
    
    def __init__(self):
        pass
    
    def get_transforms(self):
        
        transform_orig = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.
                                                    transforms.ToTensor()
                                                ])
        transform_brightness = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.ColorJitter(brightness=np.random.random(1)),
                                                    transforms.ToTensor()
                                                ])
        transform_contrast = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.ColorJitter(contrast=np.random.random(1)),
                                                transforms.ToTensor()
                                                ])
        transform_saturation = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.ColorJitter(saturation=np.random.random(1)),
                                                    transforms.ToTensor()
                                                    ])
        transform_hue =  transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ColorJitter(hue=np.random.random(1)/2.0),
                                            transforms.ToTensor()
                                            ])    
        all_transforms = [transform_orig, transform_brightness, transform_contrast, transform_saturation, transform_hue]
        
        return all_transforms

class LoaderIter(object):
    
  def __init__(self, loader):
    self.loader = loader
    self.iterator = [iter(curr_loader) for curr_loader in self.loader.loaders]

  def __iter__(self):
    return self

  def __next__(self):
    batches = [loader_iter.next() for loader_iter in self.iterator]
    return self.loader.combine_batch(batches)

  next = __next__

  def __len__(self):
    return len(self.my_loader)

  
class FuseLoader(object):
    
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return LoaderIter(self)

  def __len__(self):
    return min([len(loader) for loader in self.loaders])

  def combine_batch(self, batches):
    return batches



