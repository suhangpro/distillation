function data = read_idx(path)
%READ_IDX Read IDX file format
% Usage: data = read_idx(path)
% 
% For details about the IDX file format, see the bottom of 
% http://yann.lecun.com/exdb/mnist/
% 
% Hang Su 2016

endian = 'b'; % try big-endian first  

[pathstr, name, ext] = fileparts(path); 
if ~isempty(ext), 
  assert(strcmp(ext,'.gz'), ['Unknown file type: ' ext]); 
  gunzip(path); 
  path = fullfile(pathstr, name); 
end
assert(logical(exist(path, 'file'))); 

fid = fopen(path, 'r'); 
magicN = fread(fid, 1, '*uint32', endian); 
if magicN>=2^16, % wrong endianness
  magicN = swapbytes(magicN); 
  endian = 'l';
end 
ndim = mod(magicN, 256); 
dtype = floor(magicN/256); 

switch dtype, 
  case hex2dec('08'), 
    dtypeMat = 'uint8'; 
  case hex2dec('09'), 
    dtypeMat = 'int8';
  case hex2dec('0B'), 
    dtypeMat = 'int16';
  case hex2dec('0C'), 
    dtypeMat = 'int32';
  case hex2dec('0D'), 
    dtypeMat = 'float32';
  case hex2dec('0E'), 
    dtypeMat = 'float64';
  otherwise, 
    error('Unknown data type: 0x%s', dec2hex(dtype,2)); 
end

sz = fread(fid, [1, ndim], 'int32', endian); 
if ndim==1, 
  sz = [sz 1];
  ndim = 2;
end

data = fread(fid, prod(sz), ['*' dtypeMat], endian); 
data = reshape(data, fliplr(sz));
data = permute(data, ndim:-1:1); 

fclose(fid); 
