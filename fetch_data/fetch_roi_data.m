cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading rois...\n');
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211461&authkey=APnggEjkWW1cPEw', ...
        'rois.zip');
    fprintf('Unzipping...\n');
    unzip('rois.zip', '..');
    fprintf('Done.\n');
    delete('rois.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
