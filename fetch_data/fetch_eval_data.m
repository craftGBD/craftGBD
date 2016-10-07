cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading eval data...\n');
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211461&authkey=APnggEjkWW1cPEw', ...
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
    urlwrite('https://onedrive.live.com/download?cid=2D6AAB50CBDC610F&resid=2D6AAB50CBDC610F%211461&authkey=APnggEjkWW1cPEw', ...
        'devkit.zip');
    fprintf('Unzipping...\n');
    unzip('devkit.zip', '../evaluation');
    fprintf('Done.\n');
    delete('devkit.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
