cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading BN data...\n');
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211461&authkey=APnggEjkWW1cPEw', ...
        'BN_data.zip');
    fprintf('Unzipping...\n');
    unzip('BN_data.zip', '../BN_1k');
    fprintf('Done.\n');
    delete('BN_data.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
