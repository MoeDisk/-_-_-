%1.2.5 与图像处理相关的MATLAB函数的使用

%1.图像的文件的读/写

%imread函数用来实现图像文件的读取

A=imread('test.bmp');	%用imread函数读入图像
imshow(A);

%imwrite函数用来实现图像文件写入

imwrite(A,'test.bmp');	%可把图像文件写入matlab目录下

%iminfo函数用来查询图像文件信息

%info=iminfo('test.bmp');

%colorbar函数将颜色条添加到坐标轴对象中

RGB=imread('test.png');	%把图像读入
I=rgb2gray(RGB);	%把RGB图像转换成灰度图像
h=[1 2 1;0 0 0;-1 -2 -1];
I2=filter2(h,I);
imshow(I2,[]),colorbar('vert');	%将颜色条添加到坐标轴对象中

%warp函数将图像作为纹理进行映射，使图像显示在一个特定三维空间中。

A=imread('test.bmp');
I=rgb2gray(A);
[x,y,z]=sphere;
warp(x,y,z,I);	%用warp函数将图像作为纹理进行映射

%subimage函数实现在一个图形窗口中显示多幅图像。

RGB=imread('test.bmp');
I=rgb2gray(RGB);
subplot(1,2,1),subimage(RGB);	%subimage函数实现在一个图形窗口中显示多幅图像。
subplot(1,2,2),subimage(I);

%2.图像处理的基本操作

%imadd函数实现两幅图像的相加或者给一幅图加上一个常数

I=imread('test.bmp');
J=imadd(I,100);	%给图像增加亮度
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%imsubtract函数实现将一幅图像从另一幅图像中减去或者从一幅图像中减去一个常数。

I=imread('test.bmp');
J=imsubtract(I,100);	%给图像减少亮度
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%immultiply实现两幅图像相乘或者一幅图像的亮度缩放。

I=imread('test.bmp');
J=immultiply(I,0,5);	%图像的亮度缩放
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%imdivide函数实现两幅图像的除法或一幅图像的亮度缩放。

I=imread('test.bmp');
J=imdivide(I,0.5);	%图像亮度缩放
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%2.图像的空间域操作

%imresize函数实现图像缩放。

J=imread('test.bmp');
X1=imresize(J,2);	%对图像进行缩放
figure,imshow(J);

%imrotate函数实现图像旋转。
I=imread('test.bmp');
J=imrotate(I,45,'bilinear');
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%imcrop函数实现图像剪切。
I=imread('test.bmp');
I2=imcrop(I,[75 68 130 112]);	%对图像进行剪切
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(I2);

%3)特定域处理

%roipoly函数用于选择图像中的多边形区域。
I=imread('test.bmp');
c=[200 250 278 248 199 172];
r=[21 21 75 121 121 75];
BW=roipoly(I,c,r);
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(BW);

%roicolor函数是对RGB图像和灰度图像实现按灰度或亮度值选择区域进行处理。
a=imread('test.bmp');
I=rgb2gray(a);
BW=roicolor(I,128,225);	%按灰度值选择的区域
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(BW);

%ploy2mask函数转化指定的多边形区域为二值掩模。
x=[63 186 54 190 63];
y=[60 60 209 204 60];
bw=poly2mask(x,y,256,256);	%转化指定的多边形区域为二值掩模
imshow(bw);
hold on
plot(x,y,'b','LineWidth',2);
hold off

%roifilt2函数实现区域滤波。
a=imread('test.bmp');
I=rgb2gray(a);
c=[200 250 278 248 199 172];
r=[21 21 75 121 121 75];
BW=roipoly(I,c,r);	%roipoly函数选择图像的多边形滤波
h=fspecial('unsharp');
J=roifilt2(h,I,BW);
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%roifill函数实现对特定区域进行填充。
a=imread('test.bmp');
I=rgb2gray(a);
c=[200 250 278 248 199 172];
r=[21 21 75 121 121 75];
J=roifill(I,c,r);	%对特定区域进行填充
subplot(1,2,1),imshow(I);
subplot(1,2,2),imshow(J);

%4)图像变换
%(1)fft2函数和ifft2函数分别是计算二维的快速傅里叶变换和反变换。
f=zeros(100,100);
f(20:70,40:60)=1;
imshow(f);
F=fft2(f);	%计算二维的快速傅里叶变换
F2=log(abs(F));	%对幅值取对数
imshow(F2),colorbar;

%(2)fftshift函数实现了补零操作和改变图像显示象限。
f=zeros(100,100);
f(20:70,40:60)=1;
imshow(f);
F=fft2(f,256,256);
F2=fftshift(F);		%实现零补操作
imshow(log(abs(F2)));

%dct2函数才用基于快速傅里叶变换的算法，用于实现较大输入矩阵的离散余弦变换。
RGB=imread('test.bmp');
I=rgb2gray(RGB);
J=dct2(I);	%对I进行离散余弦变换
imshow(log(abs(J))),colorbar;
J(abs(J)<10)=0;
K=idct2(J);		%图像的二维离散余弦变换
figure,imshow(I);
figure,imshow(K,[0,255]);

%dctmtx函数用于实现较小输入矩阵的离散余弦变换。
RGB=imread('test.bmp');
I=rgb2gray(RGB);
I=im2double(I);
T=dctmtx(8);
B=blkproc(I,[8,8],'P1*x*P2',T,T');
mask=[1 1 1 1 0 0 0 0
	  1 1 1 0 0 0 0 0
	  1 1 0 0 0 0 0 0
	  1 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0];
B2=blkproc(B,[8,8],'P1.*x',mask);
I2=blkproc(B2,[8,8],'P1*x*P2',T',T);
imshow(I),figure,imshow(I2);
