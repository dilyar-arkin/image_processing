%% Assignment 3 COSC 340 written by Dilyar Arkin
clear,clc;
%% Scaling
I = imread("LennaGray.jpg");
g = imageScaling4e(I,1/6,1/6);
figure,imshow(uint8(g));

%% Rotation
clear,clc
I = imread("LennaGray.jpg");
new_Im = imageRotate4e(I,65,"full"); % enter "full" or "crop" or " " 
imshow(uint8(new_Im));
%% Functions
%% rotation function
function new_Im = imageRotate4e(I,theta,mode)
theta = theta*(pi/180); % convert radian to degree
% rotation matrix (note that minus sign changed in order to rotate c'cw)
M = [cos(theta),sin(theta),0;-sin(theta),cos(theta),0;0,0,1];
[h,w] = size(I);
Original_corner = [1,1,1;w,1,1;w,h,1;1,h,1];
Trans_corners = M * Original_corner';
minOrg = min(Trans_corners');
minx = round(minOrg(1));
miny = round(minOrg(2));
maxOrg = max(Trans_corners');
maxx = round(maxOrg(1));
maxy = round(maxOrg(2));
newW = maxx - minx;
newH = maxy - miny;
new_Im = zeros(newW,newH);

for i = minx : 1 : maxx % column
    for j = miny : 1 : maxy % row
    
        P = M^(-1) * [i;j;1]; % 
        c = round(P(1,1));
        r = round(P(2,1));

        if( r > height(I) )
            r = height(I);
        elseif(c > width(I))
            c = width(I);
        elseif(r < 1)
            r = 1;
        elseif(c<1)
            c = 1;
        elseif(I(r,c)<256 && I(r,c)>=0)
            new_Im(j-miny+1,i-minx+1) = I(r,c);
        end
    end
%pause(0.05);
end
if(mode == "full")
    return;
end
if(mode == "crop" || mode == " ")
    if(newW-w < 0.1*newW || newH - h < 0.1*newH) %small angle approximation
        return
    end
    cropped = zeros(newH,newW);
    newTopleftx = round(newH/2 - h/2);
    newToplefty = round(newW/2 - w/2);
    newLengthx = newToplefty + w;
    newLengthy = newToplefty + h;
    cropped(newToplefty:newLengthy,newTopleftx:newLengthx) = new_Im(newToplefty:newLengthy,newTopleftx:newLengthx);
    cropped = cropped(newH/2 - h/2:newH/2 + h/2,newW/2 - w/2:newW/2 + w/2);
    new_Im = cropped;
end
end
%% scaling function
function newIm = imageScaling4e(f_o,cx,cy)
    [r,c] = size(f_o);
    f = double(f_o);   
    Col = round(cx*c);
    Row = round(cy*r);    
    newIm = zeros(Row,Col);
%   scaling double for loops ( note that directly applied scaling matrix)
    for i = 1 : 1 : height(newIm)
        for j = 1 : 1 : width(newIm)
            if (round(i/cy) > height(f) || round(j/cx) > width(f) ...
                    || round(j/cx) < 1 || round(i/cy) < 1 )
                newIm(i,j) = 1;
            elseif(f(round(i/cy),round(j/cx))<256 && ...
                    f(round(i/cy),round(j/cx))>0)
                newIm(i,j) = f(round(i/cy),round(j/cx));
            else
                newIm(i,j) = 1;
            end         
        end
    end
end
