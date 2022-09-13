clc;
clear all;
close all;
Dir= dir('*.png');
image= cell(numel(Dir),1);
for i=1:numel(Dir)
    image{i} = imresize(imread(Dir(i).name),[42 32]);
    threshold = graythresh(image{i});
    image{i}=imbinarize(image{i},threshold);
    image{i} = reshape(image{i},7,8,24);
    image{i} =mean(mean(image{i}));
end
csvwrite('Trainingcsv.csv',image);