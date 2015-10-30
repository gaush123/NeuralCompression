
function test()
prototxt = {'3_prototxt_solver/lenet5/train_val.prototxt','3_prototxt_solver/lenet_300_100/train_val.prototxt', '3_prototxt_solver/L2/train_val.prototxt' , '3_prototxt_solver/vgg16/train_val.prototxt' };

caffemodel = '4_model_checkpoint/lenet5/lenet5.caffemodel.quantize.normalize','4_model_checkpoint/alexnet/alexnet9x.caffemodel.quantize.normalize','4_model_checkpoint/vgg16/vgg16_13x.caffemodel.quantize.normalize','4_model_checkpoint/lenet_300_100/lenet300_100_9x.caffemodel.quantize.normalize'};

caffe.set_mode_cpu()

layers_nets = {{'ip1','ip2'},{'ip1','ip2'},{'fc6','fc7','fc8'},{'fc6','fc7','fc8'}};
layers_nets_front = {{'data', 'ip1','ip2'},{'pool2', 'ip1','ip2'},{'pool5', 'fc6','fc7','fc8'},{'pool5', 'fc6','fc7','fc8'}};
nets = {'lenet_300', 'lenet5', 'alexnet', 'vgg'};

wordlen = 16;
fraclen = 15;
F = fimath('RoundingMethod','Round','ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision');
F_full = fimath('RoundingMethod','Round','ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');
F.ProductWordLength= wordlen * 2;
F.SumWordLength = wordlen;

system('mkdir data');
for net_id = 1:length(nets)
    system(strcat('mkdir data/', nets{net_id}));

    prototxt = prototxts{net_id};
    weights = caffemodels{net_id};
    net = caffe.Net(model, weights, 'test')
    layers = layers_nets{net_id};
    net.forward();
    int_length = eval_int_bits(net, layers_nets_front{net_id});
    act_fraclen = 15 - int_length;

    act_ = net.blobs(layers{1}).get_data();
    act_ = fi(act_(:), 1, wordlen, act_fraclen(1));
    for layer_id = 1:length(layers)
        dir = strcat('data/', nets{net_id}, '/', layers{layer_id}));
        system(strcat('mkdir data/', nets{net_id}, '/', layers{layer_id}));

        right_shift_bits = int_length(layer_id+1) - int_length(layer_id);
        fid = fopen(strcat('touch ',dir,'/rightshift.txt'),'w');
        fprintf(fid, '%d', right_shift_bits);
        fclose(fid);

        F.ProductFractionLength= act_fraclen(layer_id+1) + 16;
        F.SumFractionLength = act_fraclen(layer_id+1);
        act_ = fi(act_, F);
        %Dump
        fid = fopen(strcat(dir, '/act.dat'),'w');
        fwrite(fid, int_data(act_), 'integer*2');
        fclose(fid);

        layer = layers{layer_id}
        w_ = net.params(layer,1).get_data();
        bias_ = net.params(layer, 2).get_data() ;
        w = fi(w_, 1, wordlen, fraclen, F);
        bias = fi(bias_, 1, wordlen, fraclen, F);
        
        act_ = w' * act_;
        % Dump
        fid = fopen(strcat(dir, '/groundtruth.dat'),'w');
        fwrite(fid, int_data(act_), 'integer*2');
        fclose(fid);

        act_ += bias;
        act_(act_ < 0.0) = 0.0;
    end
end

end
function int_len = eval_int_bits(net, layers)
    int_len = zeros(len(layers),1);
    for idx = 1:len(layers)
        a = net.blobs(layers{idx}).get_data();
        m = max(-min(a(:)), max(a(:)));                                            
        int_len(idx) = floor(log2(m));
    end
end

function int_data = get_int_data(fidata)
    int_data = int16(fidata.data * (2 ^ fidata.FractionLength));
end
