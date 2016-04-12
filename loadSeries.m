% loadSeries    Load an image series (in multi-pgm format) from disk.
%    loadSeries(fname,stInd,endInd)
%
% This function loads specified images out of a multi-pgm
% file (my own format). It takes as argument the filespec,
% the starting index (1-based) and end index.
%
% It converts both single bit images and 8-bit unsigned images
% into the 32-bit matlab floating point format, assuming that
% a 0 maps to a 0.5 and a 1 maps to 255.5.
%
% For 8-bit images, every value gets a 0.5 added to it.
function ser=loadSeries(fname,stInd,endInd)  % st and end are 1-based

fid=fopen(fname,'r');
ln1=fgetl(fid);
format=ln1(2);  % Second character.


ln2=fgetl(fid);
s='# Voxel Size: ';

if strncmp(s,ln2,14)==1
  fprintf(1,'Cant deal with size info yet.');
end

xyz=sscanf(ln2(2:size(ln2,2)),'%f');  % Strip off comment char '#'

x=xyz(1);
y=xyz(2);
diskImCount=xyz(3);

ln3=fgetl(fid);   % Has normal pgm resolution
if format=='4'
  bitsPerPix='ubit1';
else 
  bitsPerPix='ubit8';
  lastline=fgetl(fid);   % This is the last line before the binary data.
end


imsPerRow=ceil(sqrt(diskImCount)-.0000001);

rowSize=imsPerRow*x;
pixPerImRow=rowSize*y;

memImCount=min(endInd,diskImCount)-stInd+1;

ser=zeros(x,y,memImCount);

curMemIm=1;
curDiskIm=1;

while(1)
  % Read in a row of images.
  imStrip=fread(fid,[rowSize,y],bitsPerPix)';
  for i=1:imsPerRow
    if curDiskIm>=stInd & curDiskIm<=endInd & curDiskIm<=diskImCount
      ser(:,:,curMemIm)=imStrip(:,(i-1)*x+1:i*x);
      curMemIm=curMemIm+1;
    end
    curDiskIm=curDiskIm+1;
  end
  if curDiskIm>endInd | curDiskIm>diskImCount
    break;
  end
end

fclose(fid);

if bitsPerPix=='ubit1'
  for i=1:memImCount
    for j=1:x/8
      ser(:,(j-1)*8+1:j*8,i)=fliplr(ser(:,(j-1)*8+1:j*8,i));
    end
  end
  ser=ser*255;
end

ser=ser+.5;

  


