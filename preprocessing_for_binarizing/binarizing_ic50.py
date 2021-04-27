import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import scipy.stats as st

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

class Binarizing_ic50():
    
    def __init__(self,ic50_list=None):
        self.ic50_list = ic50_list
        self.upsampled_ic50_list = None
        self.f_ic50 = None
        self.f_x =None
        self.x = None
        self.kde = None
        self.mu_ic50_index = None
        self.theta_ic50_index = None
        self.theta = None
        self.mu = None
        self.b = None
        self.drug = ic50_list.name
        self.max_ic50 = None
        self.NumSample = None
    
    def data_processing(self,max_ic50=15,min_ic50=-10, remove_inf=True):
        self.max_ic50 = max_ic50
        self.min_ic50 = min_ic50
        
        if remove_inf:
            self.ic50_list = self.ic50_list.replace([np.inf, -np.inf], np.nan).dropna()
            
        else:
            first_max_ic50 = -1000
            second_max_ic50 = -1000
            for ic50 in self.ic50_list:
                if ic50 < self.max_ic50 and ic50 > first_max_ic50:
                    second_max_ic50 = first_max_ic50
                    first_max_ic50 = ic50
                    
            for i in range(len(self.ic50_list)):
                if self.ic50_list[i] == np.inf:
                    self.ic50_list[i] = random.uniform(second_max_ic50,first_max_ic50)
                    
        self.ic50_list = [i for i in self.ic50_list if self.max_ic50 > i > self.min_ic50]
        return
        
    def upsampling(self, sigma=0.22,NumSample=100):
        self.NumSample = NumSample
        sample_list = []
        for ic50 in self.ic50_list:
            sample = np.random.normal(ic50, sigma, self.NumSample)
            sample_list.append(sample)
        samples = np.concatenate([sample_list],axis=1).reshape(len(self.ic50_list)*self.NumSample)
        samples.sort()
        self.upsampled_ic50_list = samples
        return
        
    def make_kernel(self,):
        max_x = max(self.ic50_list)
        min_x = min(self.ic50_list)
        self.x = np.linspace(min_x, max_x,abs(max_x-min_x)*1000)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.upsampled_ic50_list[:,None])
        self.f_ic50 = np.exp(self.kde.score_samples(self.upsampled_ic50_list[:,None]))
        self.f_x = np.exp(self.kde.score_samples(self.x[:,None]))
        return
    
    def find_index(self,y,fx,alpha=0.0005):
        for i in range(1,len(fx)):
            if fx[i-1]-alpha <= y <= fx[i]+alpha:                    
                return i
    
    def get_threta(self,):
        x_prob_d1 = np.gradient(self.f_x)
        x_prob_d2 = np.gradient(x_prob_d1)
        x_prob_d3 = np.gradient(x_prob_d2)
        
        mu_prob = np.max(self.f_x)
        mu_index = list(self.f_x).index(mu_prob)
        self.mu_ic50_index = self.find_index(self.f_x[mu_index],self.f_ic50)
        self.mu = self.upsampled_ic50_list[self.mu_ic50_index]
        
        ## i)
        d1_zero_index_list = argrelextrema(self.f_x, np.greater)[0]
        theta_list = []
        for index in d1_zero_index_list:
            integral = (self.f_x[:index-1] *(self.x[1:index]-self.x[:index-1])).sum()
            
            if self.x[index] < self.x[mu_index] and self.f_x[index]< 0.8* mu_prob and integral > 0.05:
                theta_list.append(index)
                
        if len(theta_list) != 0:
            theta_index = max(theta_list)
            self.theta_ic50_index = self.find_index(self.f_x[theta_index],self.f_ic50)
            print("first condition")
            return
        
        ## ii)
        d2_zero_index_list = argrelextrema(x_prob_d2, np.greater)[0] 
        theta_list = []
        for index in d2_zero_index_list:
            integral = (self.f_x[:index-1] *(self.x[1:index]-self.x[:index-1])).sum()
            
            if self.x[index] < self.x[mu_index] and x_prob_d3[index] >0 and self.f_x[index]< 0.8* mu_prob and integral > 0.05:
                theta_list.append(index)
        
        if len(theta_list) != 0:
            theta_index = max(theta_list)     
            self.theta_ic50_index = self.find_index(self.f_x[theta_index],self.f_ic50)
            print("second condition")
            return
        else:
        ## iii)
            self.theta_ic50_index = 0
            print("third condition")
        return 
    
    def get_binary_threshold(self,save_dir,show=True):
        
        self.theta = self.upsampled_ic50_list[self.theta_ic50_index]
        
        median = np.median(self.upsampled_ic50_list[self.theta_ic50_index:self.mu_ic50_index])
        sigma = abs(self.upsampled_ic50_list[self.mu_ic50_index] - median)
        mu = self.upsampled_ic50_list[self.mu_ic50_index]
        self.b  = norm(mu,sigma).ppf(0.05)
        
        normal_x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        
        if show:
            plt.axvline(x=self.b,color = 'green')
            plt.fill_between(self.upsampled_ic50_list, self.f_ic50, alpha=1)
            plt.plot(self.upsampled_ic50_list, np.full_like(self.upsampled_ic50_list, -0.01),'*',color='red', markeredgewidth=1)
            plt.plot(normal_x, st.norm.pdf(normal_x, mu, sigma),color='black')
            plt.xlabel(self.drug+' distribution, NumSample '+str(self.NumSample)+', threshold b: '+str(round(self.b,4)))
            plt.savefig(save_dir+self.drug+'_NumSample_'+str(self.NumSample)+'_max ic50_'+str(self.max_ic50)+'_distribution.png')
            plt.show()
        
        return self.b

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
