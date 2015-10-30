function statistics()                                                          
model = '3_prototxt_solver/L2/train_val.prototxt';                             
weights = '4_model_checkpoint/alexnet/alexnet9x.caffemodel.quantize.normalize';
caffe.set_mode_cpu()                                                           
net = caffe.Net(model, weights, 'test');                                       
                                                                               
net.forward_all()                                                              
                                                                               
fc5_output = net.blobs('pool5').get_data();                                    
fc6_output = net.blobs('fc6').get_data();                                      
fc7_output = net.blobs('fc7').get_data();                                      
fc8_output = net.blobs('fc8').get_data();                                      
stat(fc5_output)                                                               
stat(fc6_output)                                                               
stat(fc7_output)                                                               
stat(fc8_output)                                                               
%asfsdffsd                                                                     
end                                                                            
function int_length = stat(a)                                                  
    a = a(a~=0);                                                               
    display('min, max:')                                                       
    display([min(a(:)), max(a(:))])                                            
                                                                               
    m = max(-min(a(:)), max(a(:)));                                            
    int_length = round(log2(m)+0.5);                                           
    display(int_length)                                                        
                                                                               
end                                                                            
