function statistics()
model = '3_prototxt_solver/L2/train_val.prototxt';  
weights = '4_model_checkpoint/alexnet/alexnet9x.caffemodel';
caffe.set_mode_gpu()                                        
net = caffe.Net(model, weights, 'test');                    

net.forward_all()

fc5_output = net.blobs('pool5').get_data();
fc6_output = net.blobs('fc6').get_data();
fc7_output = net.blobs('fc7').get_data();
fc8_output = net.blobs('fc8').get_data();
%stat(fc5_output)
%stat(fc6_output)
stat(fc7_output)
stat(fc8_output)
end
function stat(a)
    a = a(a~=0);
    display('mean:')
    display(mean(a(:)))
    display('std:')
    display(std(a(:)))
    display('min, max:')
    display([min(a(:)), max(a(:))])
    % hist(a(:))
end

