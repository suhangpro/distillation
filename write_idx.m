function write_idx(data, path, endian, compress)
%WRITE_IDX Write IDX file format
% Usage: write_idx(data, path, endian='b', compress=true)
% 
% For details about the IDX file format, see the bottom of 
% http://yann.lecun.com/exdb/mnist/
%
% Hang Su 2016

switch class(data), 
  case 'uint8', 
    dtype = hex2dec('08');
    dtypeMat = 'uint8'; 
    dtypeStr = 'ubyte'; 
  case 'int8', 
    dtype = hex2dec('09');
    dtypeMat = 'int8'; 
    dtypeStr = 'byte'; 
  case 'int16', 
    dtype = hex2dec('0B');
    dtypeMat = 'int16'; 
    dtypeStr = 'short'; 
  case 'int32', 
    dtype = hex2dec('0C');
    dtypeMat = 'int32'; 
    dtypeStr = 'int'; 
  case 'single', 
    dtype = hex2dec('0D');
    dtypeMat = 'single'; 
    dtypeStr = 'float'; 
  case 'double', 
    dtype = hex2dec('0E');
    dtypeMat = 'double'; 
    dtypeStr = 'double'; 
  otherwise, 
    error('Data type not supported: %s', class(data)); 
end

if ~exist('endian', 'var') || isempty(endian), 
  endian = 'b'; 
end

if ~exist('compress', 'var') || isempty(compress), 
  compress = true; 
end

ndim = ndims(data); 
sz = size(data); 
assert(all(sz>0));
if sz(end)==1, 
  assert(ndim==2); 
  sz = sz(1:end-1);
  ndim = ndim-1;
end

if ~exist('path', 'var') || isempty(path), 
  path = sprintf('mat-idx%d-%s', ndim, dtypeStr); 
else
  [pathstr, name, ext] = fileparts(path); 
  if ~isempty(ext), 
    assert(strcmp(ext,'.gz'), 'Only .gz compression is supported'); 
    path = fullfile(pathstr, name); 
    if ~compress, error('.gz save name given but compression is disabled'); end
  end
end

if ndim>1, 
  data = permute(data, ndim:-1:1); 
end
fid = fopen(path, 'w+'); 
fwrite(fid, dtype*256+ndim, 'int32', endian); 
fwrite(fid, sz, 'int32', endian); 
fwrite(fid, data, dtypeMat, endian); 
fclose(fid); 

if compress, 
  gzip(path); 
  delete(path); 
end
