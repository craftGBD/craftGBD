Training RPN for getting proposals in detection task in ILSVRC2016
Any questions please contact liuyuisanai@gmail.com

Please follow the tips below:
 1. You need download dependent files (our annotations / pre-trained
 models / compiled caffemex(on windows)) on cloud drive. The compiled
 caffemex only support running on windows. Please install MATLAB 2014a
 or later version and VS2013 for good compatibility.
 2. Copy annotation file "inds1314old_1500new_flip_*.mat" in ./annotation/
 3. Copy caffe/ to ./bin/
 4. Set your own configure in script_start.m, items who need to be customized
 are marked by "%todo: customize" 
 5. Run script_start.m
 6. For testing, run get_proposal_from_list.m.
 7. For evaluation, run evaluate_recall.m. The first input 'parsed' is
 produced by 'get_proposal_from_list.m', in which you need save all
 'parsed_all' in parsed(i) in each loop.


cloud drive:
https://1drv.ms/f/s!AnrDXAvZFFwEnhxORYqHDE5DPDj9
