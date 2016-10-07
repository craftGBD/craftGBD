cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading ResNet data...\n');
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211461&authkey=APnggEjkWW1cPEw', ...
        'ResNet_data.zip');
    fprintf('Unzipping...\n');
    unzip('ResNet_data.zip', '../ResNet-GBD');
    fprintf('Done.\n');
    delete('ResNet_data.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
