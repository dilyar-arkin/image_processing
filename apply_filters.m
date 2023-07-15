%% Phys 420 - A4 written by Dilyar
%% Q1
clear,clc

% part d
w = (0:2*pi/(100-1):2*pi);
b = [0.1,0.5,0.7,0.9];
%plot the magnitude
for n = 1 : 1 : length(b);
M = abs((exp(i*w).*(1-b(n)))./(exp(i*w)-b(n)));
plot(w,M);
hold on
end
hold off;
%phase plot
figure
for n = 1 : 1 : length(b);
T = (exp(i*w).*(1-b(n)))./(exp(i*w)-b(n));
p1 = atan(imag(T)./real(T));
plot(w,p1);
hold on
end
hold off

% part e
N=100;
n = (1:1:N);
x = (sin(2*pi*n/50).*exp(-n/50)).*(sin(2*pi*n/50).*exp(-n/50));
x_noisy = imnoise(x,"gaussian",0,0.01);
figure,plot(n,x);
hold on
plot(n,x_noisy);
hold off

%part f
w = (0:2*pi/(100-1):2*pi);
%figure,plot(n,x_noisy,'LineWidth',2);

for m = 1 : 1 : length(b);
F1 = (exp(i*w).*(1-b(m)))./(exp(i*w)-b(m));
X = fft(x_noisy);
C1 = X .* F1;
c1 = ifft(C1);
plot(n,abs(c1));
hold on
end
hold off

% part g
load("TSX_Stock.mat");
n = (1:1:62);
w = (0:2*pi/(62-1):2*pi);
m = 0.7;
F2 = (exp(i*w).*m)./(exp(i*w)-m);
F2_norm = F2(:)/max(F2(:));
TSX_norm = TSX_Stock(:)/max(TSX_Stock);
Tfreq = fft(TSX_norm);
Filtered_dat = Tfreq .* F2_norm;
filt_dat = ifft(Filtered_dat);
figure,plot(n,TSX_norm','LineWidth',2)
hold on
plot(n,abs(filt_dat),'LineWidth',2);
hold off
title 'TSX_ Stock Original and Filtered (b=0.7 option selected)'

%% Q2
clear,clc
Xnoisy = load("Xnoisy.mat");
I_noisy = Xnoisy.Xnoisy;
fc = 20;
fs = 300;
[b,a] = butter(2,fc/(fs/2));
dataOut = filter(b,a,I_noisy);
figure,imshow(I_noisy);
title 'noisy'
figure,imshow(dataOut);
title 'filtered'
%%
Im = I_noisy;
% butterworth
n=5; % order
d0 = 10; % cutoff frequency
I = double(Im);
[nx ny]  = size(I);
I_fft = fftshift(fft2(I));
filtered_im = ones(nx,ny);
for i = 1 : nx-1
    for j = 1: ny-1
        dist = ((i-(nx+1))^2 + (j-(ny+1))^2)^.5;
        filtered_im(i,j) = 1/(1+(dist/d0)^(2*n));
    end
end
filtered_im = I_fft + filtered_im.*I_fft;
filtered_im = ifft2(ifftshift(filtered_im));
% end of butter

imshow(real(filtered_im));

%%

for i = 3 : 2 : 11
Im_med_filt = medfilt2(I_noisy,[i,i],'symmetric');
figure,imshow(Im_med_filt);
end

h1 = I_noisy(height(I_noisy)/2,:);
h2 = dataOut(height(dataOut)/2,:);
h3 = Im_med_filt(height(Im_med_filt)/2,:);
t1 = (1:1:length(h1));
figure,plot(t1,h1,'LineWidth',2);
hold on
plot(t1,h2,'LineWidth',2);
plot(t1,h3,'LineWidth',2);
hold off

%% Q3
clear,clc
%
Xnoisy = load("Xnoisy.mat");
XnoisyBlur = load("XnoisyBlur.mat");
Xnorm = load("Xnorm.mat");
I_noisy = Xnoisy.Xnoisy;
I_noisyBlur = XnoisyBlur.XnoisyBlur;
I_norm = Xnorm.Xnorm;
%
%figure,imshow(I_noisy);
%figure,imshow(I_noisyBlur);
%figure,imshow(I_norm);
%
h1 = I_noisyBlur(height(I_noisyBlur)/2,:);
h2 = I_noisy(height(I_noisy)/2,:);
h3 = I_norm(height(I_norm)/2,:);
t1 = (1:1:length(h1));
plot(t1,h1);
hold on
plot(t1,h2);
plot(t1,h3);
hold off
%
PSF = fspecial("gaussian",[],4);
wnr1 = deconvwnr(I_noisyBlur,PSF,1/200);
figure,imshow(wnr1);
title 'Wiener Deconvolved K = 1/200'

cnt = 0;
figure,plot(t1,h3); 
hold on
k=1;
m=0;
for n = 1 : 1 : 2000 
    wnr2 = deconvwnr(I_noisy,PSF,1/n);
    diff = I_norm - wnr2;
    mu = mean(diff,"all");
    if mu < k 
        k = mu;
        m = n;
    end
end
wnr3 = deconvwnr(I_noisy,PSF,1/n);
imshow(wnr3);
hold off
%%

%%
clear,clc
%
dat2 = load("iPhoneShake_t.csv");
% part a - plot original data vs time and plot y and z vs x
figure,plot(dat2(:,1),dat2(:,2));
hold on 
plot(dat2(:,1),dat2(:,3));
plot(dat2(:,1),dat2(:,4));
hold off
title 'original data'
figure,scatter(dat2(:,2),dat2(:,3));
title 'x vs y'
figure,scatter(dat2(:,2),dat2(:,4));
title 'x vs z '
figure,scatter(dat2(:,3),dat2(:,4));
title 'y vs z '
% comment: from orginal data, noticed that all x,y,z are priodic in nature,
% which makes sense due to what type of data this is. for plotting x vs y
% and x vs z, there is a downward trend, but it's difficult to comment on
% how these data on different axis correlate with each other.

% part b
[coeff,score,latent,tsquared,explained] = pca(dat2(:,2:4));
figure,plot(dat2(:,1),score(:,1))
hold on
plot(dat2(:,1),score(:,2))
plot(dat2(:,1),score(:,3))
title 'score1,score2,and score3 vs t'
hold off
%[U,sig,V] = svd(dat2(:,2:4),'econ');

% part c
c1 = explained(1); % 95.4%
c2 = explained(2); % 3.69%
c3 = explained(3); % 0.87%
figure,scatter3(score(:,1),score(:,2),score(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

% after projection,
% since component 1,2,and 3 are represent 99.59 % of the original data, as
% shown in 3d scatter plot, we could represent data in 3d plot, or even
% pc1 and pc2 which still represent 97.75% of the data, we could also plot
% in 2d without much loss of representation.

% part d
figure,scatter(score(:,1),score(:,2),'filled')
title 'new x vs y'
figure,scatter(score(:,1),score(:,3),'filled')
title 'new x vs z'
figure,scatter(score(:,2),score(:,3),'filled')
title 'new y vs z'

% part e
Recons = coeff * score';
Recons = Recons';
figure,scatter(Recons(:,1),Recons(:,2),'magenta','filled');
title 'reconstructed x vs yafter PCA'
figure,scatter(Recons(:,1),Recons(:,3),'green','filled');
title 'reconstructed x vs z after PCA'
figure,scatter(Recons(:,2),Recons(:,3),'red','filled');
title 'reconstructed y vs z after PCA '

% note: by comparison, we got the original datasets back in their original 
% axis. It's done by multiplying co-efficients (3x3) by inverted score
% matrix.

% part g
% by looking at the PCA component projection, it's obvious that the phone
% was shaken mainly in a single axis since PC1 have 95% representation of
% the dataset. Including some amount of contribution from y PC2 (3%), when
% pc1 and pc2 combined, the 99% confidence suggest that phone was mainly 
% shaken on the a 2d plane, with linear motion on a single axis. 
% We can not say which axis exactly because PCA result tells us the
% collective behavious of this entire dataset. 

%% Q5 
clear,clc;
[X,T] = wine_dataset;
Z = linkage(X',"complete",'euclidean');
cutoff = median([Z(end-1,3) Z(end-1,2)]);
cutoff2 = 400;
dendrogram(Z,'ColorThreshold',cutoff2)
%dendrogram(Z);
%%
clear,clc
[X,T] = wine_dataset;
X = X';

figure
scatter(X(:,1),X(:,2),'filled','black');
opts = statset('Display','iter');
[idx,C] = kmeans(X,3,'Distance','cityblock','Replicates',5,'Options',opts);
figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12,'Color','blue')
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12,'Color','green')
plot(X(idx==3,1),X(idx==3,2),'b.','MarkerSize',12,'Color','red')
plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%figure
%gscatter(X(:,1),X(:,2),idx,'bgm')
%hold on
%plot(C(:,1),C(:,2),'kx')
%legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid')

M = zeros(178,1)
M(1:59,1) = 1;
M(60:130,1) = 2;
M(131:178,1) = 3;

sc = confusionmat(M,idx)
sc1 = confusionchart(M,idx)
