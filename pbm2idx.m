function pbm2idx(train_paths, idx_data_path, idx_label_path, test_paths)
%PBM2IDX Convert pbm data to idx format
% 
% Hang Su 2016

if ~iscell(train_paths), 
  assert(ischar(train_paths));
  train_paths = {train_paths};
end
nClasses = numel(train_paths);
if ~exist('test_paths','var') || isempty(test_paths), 
  test_paths = cell(1, nClasses); 
elseif iscell(test_paths), 
  assert(numel(test_paths) == nClasses); 
else
  assert(ischar(test_paths));
  test_paths = {test_paths}; 
end

if ~exist('idx_data_path', 'var') || isempty(idx_data_path), 
  idx_data_path = 'pbm-images-idx3-ubyte';
end

if ~exist('idx_label_path', 'var') || isempty(idx_label_path), 
  idx_label_path = 'pbm-labels-idx1-ubyte';
end

processed_data = zeros(28, 28, 0, 'uint8'); 
labels = zeros(0, 'uint8'); 
for classid = 0:nClasses-1, 
  fprintf('class #%2d ...', classid); 
  raw_data = uint8(loadSeries(train_paths{classid+1}, 1, Inf) - 0.5);
  if ~isempty(test_paths{classid+1}), 
    raw_data = cat(3, raw_data, ...
                   uint8(loadSeries(test_paths{classid+1}, 1, Inf) - 0.5));
  end
  offset = size(processed_data, 3); 
  processed_data = cat(3, processed_data, ...
                       zeros(28, 28, size(raw_data, 3), 'uint8'));
  for i=1:size(raw_data, 3), 
    processed_data(:,:,i+offset) = re_sample(raw_data(:,:,i));
  end
  labels = [labels classid*ones(1, size(raw_data, 3), 'uint8')]; 
  fprintf(' done! [%2d/%2d]\n', classid+1, nClasses); 
end

idx = randperm(length(labels));
processed_data = processed_data(:,:,idx); 
labels = labels(idx);

fprintf('Writing to idx files ...'); 
write_idx(permute(processed_data, [3,1,2]), idx_data_path);
write_idx(labels', idx_label_path); 
fprintf(' done!\n'); 

function im_out = re_sample(im_in)
% im_in: uint8 single channel image of size 128x128
% im_out: uint8 single channel image of size 28x28
[iw, ih, ic] = size(im_in);
assert(ic==1);
ow = 28; oh = 28; % output size
ow_c = 20; oh_c = 20; % content size

idxs = find(sum(im_in,2)'>0);
iy_start = idxs(1);
iy_end = idxs(end);
ih_c = iy_end - iy_start + 1;
idxs = find(sum(im_in,1)>0);
ix_start = idxs(1);
ix_end = idxs(end);
iw_c = ix_end - ix_start + 1;

sc = min(ow_c/iw_c, oh_c/ih_c); 

[xx, yy] = meshgrid(0.5:iw, 0.5:ih);
[xxo, yyo] = meshgrid(1:ow, 1:oh);
center_mass =  [sum(yy(:)'*double(im_in(:))) sum(xx(:)'*double(im_in(:)))]/sum(im_in(:));
offset = [oh ow]/2 - center_mass*sc; 
xx = xx*sc + offset(2);
yy = yy*sc + offset(1);
im_out = interp2(xx, yy, single(im_in), xxo, yyo); 
im_out = uint8(max(min(im_out,255),0)); 
