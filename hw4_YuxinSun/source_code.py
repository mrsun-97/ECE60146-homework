import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
from torch.utils.data.sampler import SubsetRandomSampler
import torch_directml
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import random
import os

# use directml to run codes on AMD GPU
dml = torch_directml.device()

class HW4Net1(nn.Module):
    def __init__(self):
        super(HW4Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HW4Net2(nn.Module):
    def __init__(self):
        super(HW4Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HW4Net3(nn.Module):
    def __init__(self):
        super(HW4Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10= nn.Conv2d(32, 32, 3, padding=1)
        self.conv11= nn.Conv2d(32, 32, 3, padding=1)
        self.conv12= nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    xform = tvt.Compose([
        tvt.ToTensor(),
        # transform to range [-1, 1]:
        tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # read COCO data from coco_dir, resize them and write to save_dir
    def __init__(self, *, coco_dir="./coco", save_dir="./resized", train_num=1500, test_num=500, \
            catType=['airplane','bus','cat','dog','pizza'], update=False
        ):
        dataDir  = coco_dir
        prefix   = 'instances'
        dataType = 'train2014'
        annFile  = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
        cocoGt   = COCO(annFile)

        self.train_num = train_num
        self.test_num  = test_num
        self.dir = save_dir

        # [17, 18, 5...] to [0, 1, 2...]
        self.catId_to_label = {cocoGt.getCatIds(catType[i])[0]: i  for i in range(len(catType))}

        # [0, 1, 2...] to ['cat', 'dog', ...]
        self.label_to_cat = {i: catType[i] for i in range(len(catType))}

        self.data, self.labels = self.gen_data_id(cocoGt, catType)
        self.gen_resized_image(cocoGt, dataDir, update=update)


    # return data and label(data are actually categoryIDs)
    def gen_data_id(self, cocoGt, catType):
        catIds = cocoGt.getCatIds(catNms=catType)
        sets = [set(cocoGt.getImgIds(catIds=catIds[i])) for i in range(len(catType))]
        rg = [(i, len(sets[i]), catIds[i]) for i in range(len(catType))]
        rg.sort(key=lambda x: x[1])

        cat_sets = []
        m = self.train_num
        k = self.test_num
        n = self.train_num+self.test_num
        labels = [0]*n*len(catType)
        data = [0]*len(labels)
        for (i, length, cat)  in rg:
            s = sets[i]
            for s2 in cat_sets:
                # subtraction between two set() object
                s = s - s2
            ids = random.sample(s,n)
            cat_sets.append(set(ids))
            labels[i*m:(i+1)*m] = [cat]*m
            data[i*m:(i+1)*m] = ids[:m]
            labels[i*k+len(catType)*m:(i+1)*k+len(catType)*m] = [cat]*m
            data[i*k+len(catType)*m:(i+1)*k+len(catType)*m] = ids[m:]
        return cocoGt.loadImgs(data), labels

    # resize data. if resized data exist and update=False, we will not overwrite images.
    def gen_resized_image(self, cocoGt, dataDir, new_size=(64,64), update=False):
        for im in self.data:
            path = '%s/%s'%(self.dir, im['file_name'])
            img = Image.open('%s/images/%s'%(dataDir, im['file_name']))
            if img.mode != 'RGB':
                # force update if it is not RGB
                img = img.convert('RGB')
            elif not update and os.path.exists(path):
                continue
            img = img.resize(new_size,resample=Image.Resampling.LANCZOS)
            img.save(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pil_image = Image.open('%s/%s'%(self.dir, self.data[index]['file_name']))
        tensor_img = self.xform(pil_image)
        tensor_lab = self.catId_to_label[self.labels[index]]
        return tensor_img, tensor_lab



classes = ['pizza','airplane','bus','cat','dog']
dataset = MyDataset(update=False, catType=classes)


train_num = 1500
test_num = 500
Net = [HW4Net1, HW4Net2, HW4Net3]
plt.figure()
confusion_matrices = []

for ni in [1,2,3]:
    print("running Net%d"%ni, "...")
    net = Net[ni-1]()
    net = net.to(dml)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.99)
    )

    split = train_num*len(classes)
    indices = list(range((train_num+test_num)*len(classes)))
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                            sampler=train_sampler)
    valid_data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                            sampler=valid_sampler)

    epochs = 12
    # epochs = 2
    loss_record = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(dml)
            labels = labels.to(dml)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1)%100 == 0:
                # print("[epoch: %d, batch: %5d] loss: %.3f" \
                #     % (epoch+1, i+1, running_loss/100)            
                # )
                loss_record.append(running_loss/100)
                running_loss = 0.0
    plt.plot(loss_record,label="Net%d"%ni)

    confusion_matrix = torch.zeros( (len(classes),len(classes)) )

    with torch.no_grad():
        for i, data in enumerate(valid_data_loader):
            images, labels = data
            images = images.to(dml)
            labels = labels.to(dml)
            outputs = net(images)
            _id, predicted = torch.max(outputs.data, 1)
            for label,prediction in zip(labels,predicted):
                confusion_matrix[label][prediction] += 1
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    confusion_matrices.append(confusion_matrix)
    print("Net%d "%ni, "done")
for m in confusion_matrices:
    print(m)
    print()
plt.legend()
plt.show()