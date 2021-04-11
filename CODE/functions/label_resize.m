
clc;
clear all;
close all;
srcfile=dir('C:\Program Files\MATLAB\R2018a\bin\CamVid\imagesResized\*.png');
for i=1:length(srcfile)
    filename=strcat('C:\Program Files\MATLAB\R2018a\bin\CamVid\imagesResized\',srcfile(i).name);
    I=imread(filename);
    odprz=imresize(I,[360 480]);
    imwrite(odprz,filename);



end










