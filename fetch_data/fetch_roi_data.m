cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading rois...\n');
    urlwrite('https://onedrive.live.com/download?cid=562BEF628FB22588&resid=562BEF628FB22588%21576&authkey=AM1PFNORZu79_pM', ...
        'rois.zip');
    fprintf('Unzipping...\n');
    unzip('rois.zip', '..');
    fprintf('Done.\n');
    delete('rois.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
