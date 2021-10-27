import torch
import torch.nn as nn


# VOC
VOC_labels = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
# COCO
COCO_labels = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed',
                       'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus',
                       'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup',
                       'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe',
                       'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave',
                       'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant',
                       'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis',
                       'snowboard', 'spoon', 'sports ball', 'stop sign',
                       'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
                       'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


class M3TR(nn.Module):
    def __init__(self, vit, model, num_classes):
        super(M3TR, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = 768
        self.in_planes = 2048

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.vit_features = vit
        self.vit_classifier = nn.Linear(self.embedding_dim, num_classes)

        self.fc = nn.Conv2d(self.in_planes, num_classes, (1, 1), bias=True)
        self.GAP1d = nn.AdaptiveAvgPool1d(1)
        self.GMP1d = nn.AdaptiveMaxPool1d(1)

        self.fc_transform = nn.Linear(196, self.embedding_dim)
        self.sem_cls_transform = nn.Conv2d(self.in_planes, self.embedding_dim, (1, 1), bias=True)
        self.sem_classifier = nn.Linear(self.embedding_dim, num_classes)
        self.last_linear = nn.Linear(self.embedding_dim, num_classes)
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

        self.sem_embedding = self.get_word_embedding(self.num_classes).detach()
        self.embed_conv = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.GAP2d = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, num_classes)

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def get_semantic_token(self, x):
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        x = self.fc_transform(mask)
        return x

    def get_word_embedding(self, num_classes):
        embedding_path = './Bert/voc_embeddings.pt'
        labels = VOC_labels
        if num_classes == 80:
            # coco
            embedding_path = './Bert/coco_embeddings.pt'
            labels = COCO_labels

        loaded = torch.load(embedding_path)
        
        embedding_tensors = torch.zeros((1, num_classes, self.embedding_dim))
        for c_idx in range(num_classes):
            label = labels[c_idx]
            embedding_tensors[0, c_idx, :] = loaded[label]
        print('[Init Embedding tensors]:', embedding_tensors.shape)
        return embedding_tensors

    def forward_baseline(self, x):
        x = self.GAP2d(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.classifier(x)
        return x

    def forward(self, x):
        x_cnn = self.forward_feature(x).detach()
        sem_token = self.get_semantic_token(x_cnn)
        sem_cls1 = self.GAP1d(sem_token).squeeze()

        vit_tmp, x_vit = self.vit_features.forward_features(x, sem_token, self.sem_embedding.cuda())
        x_vit = x_vit.view(x_vit.shape[0], x_vit.shape[1])
        vit_cls = self.vit_classifier(x_vit)

        # semantic tokens
        sem_cls_final = vit_tmp[:, -(self.num_classes*2):-self.num_classes]
        sem_cls_final = self.last_linear(sem_cls_final)

        # embedding mask
        embed_mask = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                embed_mask[i, j] = torch.cosine_similarity(self.sem_embedding[:, i].cuda(), self.sem_embedding[:, j].cuda(), dim=1)

        embed_mask = embed_mask.cuda().detach()
        mask_mat = self.mask_mat.detach()
        sem_cls_final = (sem_cls_final * embed_mask).sum(-1)

        return vit_cls, sem_cls1, sem_cls_final

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters())) + list(map(id, self.vit_features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.vit_features.parameters(), 'lr': lr * lrp},
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

