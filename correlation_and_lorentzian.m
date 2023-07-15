%% Examples of lorentzian spectral plotting and 2D correlation mapping in matlab
clear
clc
clf
%% Declare variables
% c is peak position
c1 = 25;
c2 = 50;
c3 = 75;
c4 = 100;
c5 = 125;
% w is peak width
w1 = 3;
w2 = 5;
w3 = 2;
w4 = 2;
w5 = 3;
% set x interval to be from zero to 150 at 0.1 increament
x = [0:1:150];
% set zeros matrix for the 2d correlation map
data_mat = zeros(20,151);

%% loop to plot 20 spectra 
for i = 1:1:20 % for i equal to 0 to 20 with an increament of 1
    Lx1 = exp(-i/2) ./ (1+ ((x - c1)./w1).^2);
    Lx2 = 0.9^(20-i) ./ (1+ ((x - c2)./w2).^2);
    Lx3 = exp(-i/12) ./ (1+ ((x - c3)./w3).^2);
    Lx4 = 0.05*i ./ (1+ ((x - c4)./w4).^2);
    Lx5 = 0.25*exp(-i/5) ./ (1+ ((x - c5)./w5).^2);
    Lxtot = Lx1 + Lx2 + Lx3 + Lx4 + Lx5 ; % sum of lorentzians over the domain
    plot(x,imnoise(Lxtot,"gaussian",0,0.0001)) % plot by applying noise - gaussian, mean 0, and variance very small so look nicer
    hold on 
    data_mat(i,:) = Lxtot;
end
hold off
title 'noisy lorentzian with 20 spectras and 5 peaks', xlabel 'position'
ylabel 'amplitude'

%% 2d correlation plot

%initialize correlation mapping matrix
corr_map=zeros(151,151);

%given for loop on the question, set the size of matrix symmetric
for j = 1:151
    for k = 1:151
        A = data_mat(:,j); %pull vector i from A
        B = data_mat(:,k); % pull vector j from B
        corr_map(j,k) = corr2(A,B);
    end
end
%plot the result
figure,imagesc(corr_map);colormap jet;
colorbar; title correlation-2Dmap

%% end of code

%% function