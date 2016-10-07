function output = fetch_output(handle)
    if isa(handle, 'caffe.Net')
        output = fetch_net_output(handle);
    elseif isa(handle, 'caffe.Solver')
        for i = 1 : length(handle.nets)
            output_t = fetch_net_output(handle.nets{i});
            if i == 1
                output = output_t;
            else
                for j = 1 : length(output_t)
                    output(j).value = output(j).value + output_t(j).value;
                end
            end
        end
        for j = 1 : length(output)
            output(j).value = output(j).value / length(handle.nets);
        end
    else
        error('only accept net or solver handle');
    end
end
    
function output = fetch_net_output(handle)
    outputs = handle.outputs;
    for i = 1 : length(outputs)
        output(i).name = outputs{i};
        output(i).value = handle.blob_vec(handle.name2blob_index(output(i).name)).get_data();
    end
end