% script_reorder_mnist

images = read_idx('data/train-images-idx3-ubyte.gz');
labels = read_idx('data/train-labels-idx1-ubyte.gz'); 

train_size = 55000;
reorder_size = 27500;
num_labels = 10; 
num_images = length(labels);

num_samples = reorder_size / num_labels; 

idxs = zeros(num_labels, num_samples); 
for i=0:num_labels-1, 
    idxs(i+1, :) = find(labels(1:train_size)==i, num_samples); 
end

idx = [idxs(:); setdiff((1:num_images)', idxs(:))]; 

images = images(idx, :, :);
labels = labels(idx); 

write_idx(images, 'data/mnist-images-idx3-ubyte.gz');
write_idx(labels, 'data/mnist-labels-idx1-ubyte.gz'); 
