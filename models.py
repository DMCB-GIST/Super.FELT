import torch
import torch.nn as nn

class Supervised_Encoder(nn.Module):
    def __init__(self,input_dim,output_dim,drop_rate):
        super(Supervised_Encoder, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate)
            )
    def forward(self, x):
        output = self.model(x)
        return output

class AutoEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,drop_rate):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            )
        self.decoder = torch.nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.Dropout(drop_rate),
            )
    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x

class Classifier(nn.Module):
    def __init__(self,input_dim,output_dim,drop_rate):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
            )
    def forward(self, x):
        output = self.model(x)
        return output

class OnlineTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTriplet, self).__init__()
        self.marg = marg
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets

class OnlineTestTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.marg = marg
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets
