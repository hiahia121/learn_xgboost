#!/bin/bash


awk -F ' ' '{if ($9=="-") print $0}'  used_car_train_20200313.csv | wc -l
