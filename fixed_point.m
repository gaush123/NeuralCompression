

model = '3_prototxt_solver/L2/train_val_batch_1.prototxt';
weights = '4_model_checkpoint/alexnet/alexnet9x.caffemodel.quantize';
caffe.set_mode_gpu()
net = caffe.Net(model, weights, 'test');

layers = {'fc6', 'fc7', 'fc8'};
act_fraclen = [6, 5, 4, 0]; % For 16-bit
act_fraclen32 = [19, 18, 17, 13]; % For 16-bit

w = containers.Map
act = containers.Map
bias = containers.Map

wordlen = 16;
fraclen = 15;
F = fimath('RoundingMethod','Round','ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision');
F_full = fimath('RoundingMethod','Round','ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');
F.ProductWordLength= wordlen * 2;
F.SumWordLength = wordlen;


for idx = 1:length(layers)
    F.ProductFractionLength=act_fraclen32(idx+1);
    F.SumFractionLength = act_fraclen(idx+1);

    layer = layers{idx};
    w_ = net.params(layer,1).get_data();
    max_w = max(abs(w_(:)));
    w_ = w_ / max_w;
    bias_ = net.params(layer, 2).get_data() / max_w;
    w(layer) = fi(w_, 1, wordlen, fraclen, F);
    bias(layer) = fi(bias_, 1, wordlen, fraclen, F);
end

file = fopen('fixed.log','w');

true_original = 0;
true_quantized = 0;

for times = 1:1000

net.forward_all()
ground_truth = net.blobs('label').get_data();

mid_input = net.blobs('pool5').get_data();
mid_input = reshape(mid_input, 9216, 1);
act_ = fi(mid_input, 1, wordlen, act_fraclen(1));

tic
for idx = 1:length(layers)
    layer = layers{idx};
    display(layer)

    F.ProductFractionLength=act_fraclen32(idx+1);
    F.SumFractionLength = act_fraclen(idx+1);
    act_ = fi(act_, F);

    act_ = w(layer)' * act_+ bias(layer);
    %out_ori = net.blobs(layer).get_data();
    %act_(1:10)'
    %out_ori(1:10)'
    act_(act_ < 0.0) = 0.0;
end

fc8_out = fi(act_, F_full);
final_out = net.blobs('fc8').get_data();

mean(abs(fc8_out - final_out))

idx = 1
[a, i_new] = sort(-fc8_out(:,idx));
[a, i_ori] = sort(-final_out(:,idx));
i_new(1:5)'
i_ori(1:5)'
ground_truth
display('=================================\n')

fprintf(file, '%d %d %d %d %d\n', i_new(1:5));
fprintf(file, '%d %d %d %d %d\n\n', i_ori(1:5));
true_original =true_original + length(find(i_ori(1:5)==ground_truth+1));
true_quantized =true_quantized + length(find(i_new(1:5)==ground_truth+1));
fprintf('quantized accuracy top-5: %f\n', true_quantized / times)
fprintf('original accuracy top-5: %f\n', true_original / times)

end
fprintf(file, 'quantized accuracy top-5: %f\n', true_quantized / times)
fprintf(file, 'original accuracy top-5: %f\n', true_original / times)
toc
