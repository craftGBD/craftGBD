cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading eval data...\n');
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211472&authkey=ANC-D8F_bKPKxy4', ...
        'data.zip');
    fprintf('Unzipping...\n');
    unzip('data.zip', '../evaluation');
    fprintf('Done.\n');
    delete('data.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

try
    fprintf('Downloading eval devkit...\n');
    urlwrite('https://onedrive.live.com/download?cid=562BEF628FB22588&resid=562BEF628FB22588%21573&authkey=AExuFXM8W3IKLpk', ...
        'devkit.zip');
    fprintf('Unzipping...\n');
    unzip('devkit.zip', '../evaluation');
    fprintf('Done.\n');
    delete('devkit.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
