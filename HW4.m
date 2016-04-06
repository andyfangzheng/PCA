%% Test 1
clear all; close all; clc;

figure(1) %track of the motion of the shining point
%camera1
load('cam1_1.mat');
vidFrames1_1=vidFrames1_1(211:370,301:360,:,:);
[m,n,p,t]=size(vidFrames1_1);
for j=1:t
    grayFrames1_1=rgb2gray(vidFrames1_1(:,:,:,j));
    [I,J] = find(grayFrames1_1>253);
    X1_1(j)=210+mean(I(:));
    Y1_1(j)=300+mean(J(:));
end
plot(1:150,X1_1(1:150),'r');
hold on;

%camera2
load('cam2_1.mat');
vidFrames2_1=vidFrames2_1(81:300,251:350,:,:);
[m,n,p,t]=size(vidFrames2_1);
for j=1:t
    grayFrames2_1=rgb2gray(vidFrames2_1(:,:,:,j));
    [I,J] = find(grayFrames2_1>250);
    X2_1(j)=80+mean(I(:));
    Y2_1(j)=250+mean(J(:));
end
plot(1:150,X2_1(11:160),'g');
hold on;

%camera3
load('cam3_1.mat');
vidFrames3_1=vidFrames3_1(231:300,251:450,:,:);
[m,n,p,t]=size(vidFrames3_1);
for j=1:t
    grayFrames3_1=rgb2gray(vidFrames3_1(:,:,:,j));
    [I,J] = find(grayFrames3_1>246);
    X3_1(j)=230+mean(I(:));
    Y3_1(j)=250+mean(J(:));
end
plot(1:150,Y3_1(1:150));
title('Original Motions of the Shining Poing')

%SVD
X(1,:)=X1_1(1:150);
X(2,:)=Y1_1(1:150);
X(3,:)=X2_1(11:160);
X(4,:)=Y2_1(11:160);
X(5,:)=X3_1(1:150);
X(6,:)=Y3_1(1:150);
[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean
Cx=(1/(n-1))*X*X'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
sig=sqrt(lambda);
V=V(:,m_arrange);
Y=V'*X; % produce the principal components projection

figure(2)
%Plot the energys
subplot(2,1,1) 
plot(1:6,sig/(sum(sig(:))),'ko');
title('Energys');
energy1=sig(1)/(sum(sig(:)));
%Plot the motions after applying the principal components projection
subplot(2,1,2)
plot(Y(1,:),'r');hold on %donimant phases ,in red
plot(Y(2,:));hold on %other phases, in blue
plot(Y(3,:));hold on
plot(Y(4,:));hold on  
plot(Y(5,:));hold on
plot(Y(6,:));
title('Motions After Projection');

%% Test 2
clear all; close all; clc;

figure(1) %track of the motion of the shining point
%camera1
load('cam1_2.mat');
[m,n,p,t]=size(vidFrames1_2);
vidFrames1_2=vidFrames1_2(211:370,301:400,:,:);
[m,n,p,t]=size(vidFrames1_2);
for j=1:t
    grayFrames1_2=rgb2gray(vidFrames1_2(:,:,:,j));
    [I,J] = find(grayFrames1_2>252);
    X1_2(j)=210+mean(I(:));
    Y1_2(j)=300+mean(J(:));
end
plot(1:250,X1_2(1:250),'r');
hold on;

%camera2
load('cam2_2.mat');
vidFrames2_2=vidFrames2_2(:,201:400,:,:);
[m,n,p,t]=size(vidFrames2_2);
for j=1:t
    grayFrames2_2=rgb2gray(vidFrames2_2(:,:,:,j));
    [I,J] = find(grayFrames2_2>249);
    X2_2(j)=mean(I(:));
    Y2_2(j)=200+mean(J(:));
end
plot(1:250,X2_2(21:270),'g');
hold on;

%camera3
load('cam3_2.mat');
vidFrames3_2=vidFrames3_2(201:300,251:450,:,:);
[m,n,p,t]=size(vidFrames3_2);
for j=1:t
    grayFrames3_2=rgb2gray(vidFrames3_2(:,:,:,j));
    [I,J] = find(grayFrames3_2>246);
    X3_2(j)=200+mean(I(:));
    Y3_2(j)=250+mean(J(:));
end
plot(1:250,Y3_2(1:250));
title('Original Motions of the Shining Poing')

%SVD
X(1,:)=X1_2(1:250);
X(2,:)=Y1_2(1:250);
X(3,:)=X2_2(21:270);
X(4,:)=Y2_2(21:270);
X(5,:)=X3_2(1:250);
X(6,:)=Y3_2(1:250);
[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean
Cx=(1/(n-1))*X*X'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
sig=sqrt(lambda);
V=V(:,m_arrange);
Y=V'*X; % produce the principal components projection

figure(2)
%Plot the energys
subplot(2,1,1) 
plot(1:6,sig/(sum(sig(:))),'ko');
title('Energys');
energy2=sig(1)/(sum(sig(:)));
%Plot the motions after applying the principal components projection
subplot(2,1,2)
plot(Y(1,:),'r');hold on %donimant phases ,in red
plot(Y(2,:));hold on %other phases, in blue
plot(Y(3,:));hold on
plot(Y(4,:));hold on  
plot(Y(5,:));hold on
plot(Y(6,:));
title('Motions After Projection');

%% Test 3
clear all; close all; clc;

figure(1) %track of the motion of the shining point
%camera1
load('cam1_3.mat');
vidFrames1_3=vidFrames1_3(231:350,251:350,:,:);
[m,n,p,t]=size(vidFrames1_3);
for j=1:t
    grayFrames1_3=rgb2gray(vidFrames1_3(:,:,:,j));
    [I,J] = find(grayFrames1_3>248);
    X1_3(j)=230+mean(I(:));
    Y1_3(j)=250+mean(J(:));
end
plot(1:150,X1_3(16:165),'r');
hold on;

%camera2
load('cam2_3.mat');
vidFrames2_3=vidFrames2_3(151:350,201:400,:,:);
[m,n,p,t]=size(vidFrames2_3);
for j=1:t
    grayFrames2_3=rgb2gray(vidFrames2_3(:,:,:,j));
    [I,J] = find(grayFrames2_3>250);
    X2_3(j)=150+mean(I(:));
    Y2_3(j)=200+mean(J(:));
end
plot(1:150,X2_3(1:150),'g');
hold on;

%camera3
load('cam3_3.mat');
vidFrames3_3=vidFrames3_3(151:350,251:400,:,:);
[m,n,p,t]=size(vidFrames3_3);
for j=1:t
    grayFrames3_3=rgb2gray(vidFrames3_3(:,:,:,j));
    [I,J] = find(grayFrames3_3>249);
    X3_3(j)=150+mean(I(:));
    Y3_3(j)=250+mean(J(:));
end
plot(1:150,Y3_3(6:155));
title('Original Motions of the Shining Poing')

%SVD
X(1,:)=X1_3(16:165);
X(2,:)=Y1_3(16:165);
X(3,:)=X2_3(1:150);
X(4,:)=Y2_3(1:150);
X(5,:)=X3_3(6:155);
X(6,:)=Y3_3(6:155);
[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean
Cx=(1/(n-1))*X*X'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
sig=sqrt(lambda);
V=V(:,m_arrange);
Y=V'*X; % produce the principal components projection

figure(2)
%Plot principle sigma square
subplot(2,1,1) 
plot(1:6,sig/(sum(sig(:))),'ko');
title('Energys');
energy3=sig(1:3)/(sum(sig(:)));
%Plot the principle component signal
subplot(2,1,2)
plot(Y(1,:),'r');hold on %three donimant phases ,in red
plot(Y(2,:),'r');hold on 
plot(Y(3,:),'r');hold on
plot(Y(4,:));hold on     %other phases, in blue
plot(Y(5,:));hold on
plot(Y(6,:));
title('Motions After Projection');

%% Test 4
clear all; close all; clc;

figure(1) %track of the motion of the shining point
%camera1
load('cam1_4.mat');
vidFrames1_4=vidFrames1_4(231:350,301:450,:,:);
[m,n,p,t]=size(vidFrames1_4);
for j=1:t
    grayFrames1_4=rgb2gray(vidFrames1_4(:,:,:,j));
    [I,J] = find(grayFrames1_4>233);
    X1_4(j)=230+mean(I(:));
    Y1_4(j)=300+mean(J(:));
end
plot(1:250,X1_4(1:250),'r');
hold on;

%camera2
load('cam2_4.mat');
vidFrames2_4=vidFrames2_4(101:300,201:450,:,:);
[m,n,p,t]=size(vidFrames2_4);
for j=1:t
    grayFrames2_4=rgb2gray(vidFrames2_4(:,:,:,j));
    [I,J] = find(grayFrames2_4>250);
    X2_4(j)=100+mean(I(:));
    Y2_4(j)=200+mean(J(:));
end
plot(1:250,X2_4(6:255),'g');
hold on;

%camera3
load('cam3_4.mat');
vidFrames3_4=vidFrames3_4(151:300,251:450,:,:);
[m,n,p,t]=size(vidFrames3_4);
for j=1:t
    grayFrames3_4=rgb2gray(vidFrames3_4(:,:,:,j));
    [I,J] = find(grayFrames3_4>234);
    X3_4(j)=150+mean(I(:));
    Y3_4(j)=250+mean(J(:));
end
plot(1:250,Y3_4(1:250));
title('Original Motions of the Shining Poing')

%SVD
X(1,:)=X1_4(1:250);
X(2,:)=Y1_4(1:250);
X(3,:)=X2_4(6:255);
X(4,:)=Y2_4(6:255);
X(5,:)=X3_4(1:250);
X(6,:)=Y3_4(1:250);
[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean
Cx=(1/(n-1))*X*X'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
sig=sqrt(lambda);
V=V(:,m_arrange);
Y=V'*X; % produce the principal components projection

figure(2)
%Plot principle sigma square
subplot(2,1,1) 
plot(1:6,sig/(sum(sig(:))),'ko');
title('Energys');
energy4=sig(1:3)/(sum(sig(:)));
%Plot the principle component signal
subplot(2,1,2)
plot(Y(1,:),'r');hold on %three donimant phases ,in red
plot(Y(2,:),'r');hold on 
plot(Y(3,:),'r');hold on
plot(Y(4,:));hold on     %other phases, in blue
plot(Y(5,:));hold on
plot(Y(6,:));
title('Motions After Projection');