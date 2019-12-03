#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: yonghong, luo
"""
from __future__ import print_function
import sys
sys.path.append("..")
import WGAN_GRUI 
import tensorflow as tf
import argparse
import numpy as np
from PPMIData import readData
import os

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=100)
    parser.add_argument('--pretrain-epoch', type=int, default=5)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default="../PPMIData/")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--epoch', type=int, default=50)
    # feature dimension
    parser.add_argument('--n-inputs', type=int, default=134)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2) # not needed
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=0)
    parser.add_argument('--disc-iters',type=int,default=8)
    args = parser.parse_args()
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True

    #make the max step length of two datasett the same
    epochs=[30]
    g_loss_lambdas=[0.1]
    beta1s=[0.5]
    for beta1 in beta1s:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.beta1=beta1
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                ppmi = readData.ReadPPMIData(batch_size=args.batch_size, \
                                             data_path=args.data_path, isNormal=args.isNormal)
                dt_train = ppmi.read_train()
                dt_test = ppmi.read_test()
                
                #dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"train","list.txt"),isNormal=args.isNormal,isSlicing=args.isSlicing)
                #dt_test=readTestData.ReadPhysionetData(os.path.join(args.data_path,"test"), os.path.join(args.data_path,"test","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True 
                with tf.Session(config=config) as sess:
                    gan = WGAN_GRUI.WGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )
            
                    # build graph
                    gan.build_model()
            
                    # show network architecture
                    #show_all_variables()
            
                    # launch the graph in a session
                    gan.train()
                    print(" [*] Training finished!")
                    
                    gan.imputation(dt_train,True, pat_ix=True)
                    
                    print(" [*] Train dataset Imputation finished!")
                    
                    gan.imputation(dt_test,False, pat_ix=True)
                    
                    print(" [*] Test dataset Imputation finished!")
                tf.reset_default_graph()
if __name__ == '__main__':
    main()
