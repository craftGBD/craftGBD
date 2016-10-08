cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading BN data...\n');
    urlwrite('https://onedrive.live.com/embed?cid=562BEF628FB22588&resid=562BEF628FB22588%21575&authkey=AGlMhutVZbYqPdk', ...
        'BN_data.zip');
    fprintf('Unzipping...\n');
    unzip('BN_data.zip', '../BN_1k');
    fprintf('Done.\n');
    delete('BN_data.zip');
catch
    fprintf('Error in downloading, please refer to links in this script or contact us.\n'); 
end

cd(cur_dir);
