%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Direct Bimodal Image Deblurring with Event
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
format longG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for real dataset of the EBT
dvs_contrast = 0.35;
timescale = 1e6;
image_height = 260;
image_width = 346;
steps = 256;%steps for thresholding, 256 is enough usually
rgb = true;
filtering = true;
median_range = 3; %the range of median filter, should be an odd number
half_median_range = floor(median_range / 2);
sim_data = false;
generation_mode = "time"; % time or event
generation_frame_rate = 1000;
per_n_event = 100;
high_rate_frames = true;
% time shift for delay10
t_shift = 0.00;
dataset_name = "EBT"
std_times = 3;
med_filter = true;
start_path = ".";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get list of all subfolders.
allSubFolders = genpath(start_path);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};

while true
    [singleSubFolder, remain] = strtok(remain, ':');

    if isempty(singleSubFolder)
        break;
    end

    listOfFolderNames = [listOfFolderNames singleSubFolder];
end

numberOfFolders = length(listOfFolderNames);
% Process all image files in those folders.
for k = 1:numberOfFolders
    % Get this folder and print it out.
    thisFolder = listOfFolderNames{k};
    % fprintf('Processing folder %s\n', thisFolder);
    % Get MAT files.
    filePattern = sprintf('%s/*.mat', thisFolder);
    baseFileNames = dir(filePattern);
    numberOfImageFiles = length(baseFileNames);

    % Now we have a list of all files in this folder.
    if numberOfImageFiles >= 1
        % Go through all those mat files.
        for f = 1:numberOfImageFiles
            fullFileName = fullfile(thisFolder, baseFileNames(f).name);
            [pathstr, name, ext] = fileparts(fullFileName);
            method_name_list = ["binary_imgs_deblur", "binary_imgs_high_rates", "binary_imgs_high_rates_filtered", "raw_gray"];

            for method = method_name_list
                new_dir_name = fullfile(pathstr, append(method, "_", name));

                if ~exist(new_dir_name, 'dir')
                    mkdir(new_dir_name);
                end

            end

            load([fullFileName]);
            y_o = double(aedat.data.polarity.y);
            x_o = double(aedat.data.polarity.x);
            pol_o = double(aedat.data.polarity.polarity);
            pol_o(pol_o == 0) = -1;
            t_o = double(aedat.data.polarity.timeStamp) ./ timescale;
            fprintf('     Processing mat file %s\n', fullFileName);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % preload groundtruth
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            previous_valid_irr_log = zeros(image_height, image_width);

            frame_num = size(squeeze(double(aedat.data.frame.samples(:, :, :))), 1);
            starting_frame = 1;
            ending_frame = frame_num - 1;
            events_ignore_map = zeros(image_height, image_width);
            events_ignore_map_level_1 = zeros(image_height, image_width);
            events_ignore_map_level_2 = zeros(image_height, image_width);


            gt_frame_counter = 1;
            for frame = starting_frame:ending_frame

                % here I directly read the exposure time
                frame_time = double(aedat.data.frame.timeStamp(frame)) ./ timescale;
                eventstart_inframe = double(aedat.data.frame.expStart(frame)) ./ timescale;
                eventend_inframe = double(aedat.data.frame.expEnd(frame)) ./ timescale;

                eventstart_betw_frame = double(aedat.data.frame.expStart(frame)) ./ timescale;
                eventend_betw_frame = double(aedat.data.frame.expStart(frame + 1)) ./ timescale;

                exptime = eventend_inframe - eventstart_inframe;

                % for grayscale image
                if rgb == false
                    aps_image = squeeze(double(aedat.data.frame.samples(frame, :, :)));
                    aps_image = aps_image ./ 255;
                    new_dir_name = fullfile(pathstr, append(method_name_list(end), "_", name));
                    imwrite(aps_image, append(new_dir_name, "/", int2str(eventstart_inframe * timescale), ".png"));
                    aps_image = aps_image .* 255;
                else
                    % for rgb images
                    redChannel = squeeze(double(aedat.data.frame.samples(frame, :, :, 3)));
                    greenChannel = squeeze(double(aedat.data.frame.samples(frame, :, :, 2)));
                    blueChannel = squeeze(double(aedat.data.frame.samples(frame, :, :, 1)));
                    aps_image = cat(3, redChannel, greenChannel, blueChannel) ./ 255;
                    aps_image = rgb2gray(aps_image);
                    % save the grayscale
                    new_dir_name = fullfile(pathstr, append(method_name_list(end), "_", name));
                    imwrite(aps_image, append(new_dir_name, "/", int2str(eventstart_inframe * timescale), ".png"));

                    aps_image = aps_image .* 255;
                end
                norm_aps = (aps_image - min(min(aps_image)))./(max(max(aps_image))-min(min(aps_image))); % range based normalization is problematic,

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % reconstruct the clear image
                % select event in range
                x = x_o; y = y_o; pol = pol_o; t = t_o;
                idx = (t >= eventstart_inframe) & (t <= eventend_inframe);
                y(idx ~= 1) = [];
                x(idx ~= 1) = [];
                pol(idx ~= 1) = []; %
                t(idx ~= 1) = [];

                event_size = length(t);
                event_img = zeros(image_height, image_width);
                event_binary_map = zeros(image_height, image_width);
                event_binary_flag = zeros(image_height, image_width);

                t_start = t(1);
                event_img_first_edge_pos = zeros(image_height, image_width);
                event_img_first_edge_neg = zeros(image_height, image_width);
                event_img_first_lock = zeros(image_height, image_width);

                for i = 1:event_size
                    ev_t = t(i);
                    ev_x = x(i) + 1;
                    ev_y = y(i) + 1;
                    ev_p = pol(i);

                    if ev_t >= t_start + t_shift

                        if ev_p > 0 && event_img_first_edge_pos(ev_y, ev_x) >= 0 && event_img_first_lock(ev_y, ev_x) == 0
                            event_img_first_edge_pos(ev_y, ev_x) = event_img_first_edge_pos(ev_y, ev_x) + (ev_p) * dvs_contrast;
                        end

                        if ev_p < 0 && event_img_first_edge_neg(ev_y, ev_x) <= 0 && event_img_first_lock(ev_y, ev_x) == 0
                            event_img_first_edge_neg(ev_y, ev_x) = event_img_first_edge_neg(ev_y, ev_x) + (ev_p) * dvs_contrast;
                        end

                        if ev_p < 0 && event_img_first_edge_pos(ev_y, ev_x) > 0 && event_img_first_lock(ev_y, ev_x) == 0
                            event_img_first_lock(ev_y, ev_x) = 1;
                        end

                        if ev_p > 0 && event_img_first_edge_neg(ev_y, ev_x) < 0 && event_img_first_lock(ev_y, ev_x) == 0
                            event_img_first_lock(ev_y, ev_x) = 1;
                        end

                    end

                end

                first_edge_map = abs(event_img_first_edge_pos + event_img_first_edge_neg);

                % reject value larger than 2 sigma
                edge_mean = mean(nonzeros(first_edge_map),'all');
                edge_std = std(nonzeros(first_edge_map),1,'all');
                edge_median = median(nonzeros(first_edge_map), 'all');
                edge_mad = mad(nonzeros(first_edge_map), 0, 'all');
                if med_filter
                    first_edge_map = medfilt2(first_edge_map);
                else
                    first_edge_map(first_edge_map > (std_times * edge_std + edge_mean)) = 0;

                end
                first_edge_map(event_img_first_edge_neg < 0) = -first_edge_map(event_img_first_edge_neg< 0); 


                % reverse reconstruction: positive edges comes from small valeu, thus pos = max(pos) - pos.
                log_max_pos = max(max(first_edge_map));
                log_max_neg = max(max(first_edge_map(first_edge_map<0)));
                

                log_max = max(max(abs(first_edge_map)));

                recon_first_edge_map = first_edge_map;
                recon_first_edge_map(recon_first_edge_map>0) = log_max_pos - recon_first_edge_map(recon_first_edge_map>0);
                recon_first_edge_map(recon_first_edge_map<0) = log_max_neg - recon_first_edge_map(recon_first_edge_map<0);
                recon_first_edge_map = exp(recon_first_edge_map);
                norm_recon_first_edge_map = (recon_first_edge_map - min(min(recon_first_edge_map))) ./ (max(max(recon_first_edge_map))-min(min(recon_first_edge_map)));


                % image fusion to for threshold estimation
                fuse_image_noevent = norm_aps;
                fuse_image_noevent(first_edge_map ~= 0) = norm_recon_first_edge_map(first_edge_map~=0); %crop out pixels in the aps, where edges are not zeros.
                fuse_image = fuse_image_noevent;


                % threshold estimation using maximum between class variance (ours)---------------------------------------------
                [COUNTS, X] = imhist(fuse_image, steps);
                % Total number of pixels
                total = size(fuse_image, 1) * size(fuse_image, 2);
                sumVal = 0;

                for p = 1:steps
                    sumVal = sumVal + ((p - 1) * COUNTS(p) / total);
                end
                varMax = 0;
                threshold = 0;
                omega_1 = 0;
                omega_2 = 0;
                mu_1 = 0;
                mu_2 = 0;
                mu_k = 0;

                for p = 1:steps
                    omega_1 = omega_1 + COUNTS(p) / total;
                    omega_2 = 1 - omega_1;
                    mu_k = mu_k + (p - 1) * (COUNTS(p) / total);
                    mu_1 = mu_k / omega_1;
                    mu_2 = (sumVal - mu_k) / (omega_2);
                    currentVar = omega_1 * mu_1 ^ 2 + omega_2 * mu_2 ^ 2;
                    % Check if new maximum found
                    if (currentVar > varMax)
                        varMax = currentVar;
                        threshold = p - 1;
                    end
                end
                varMax
                edge_threshold_norm = threshold / steps

                edge_max = log_max; 

                edge_threshold = edge_threshold_norm * edge_max; %thresh in log space

                % stage 2 threshold the image
                norm_aps(norm_aps >= edge_threshold_norm) = 1;
                norm_aps(norm_aps < edge_threshold_norm) = 0;
                event_binary_map = norm_aps;


                % stage 1, threshold the events
                %% % % % % % % % % % % % % % % % % % % % % % % % % % % %  let try to capture the first large edge.
                event_size = length(t);
                event_img_large_edge = zeros(image_height, image_width);
                event_img_large_edge_lock = zeros(image_height, image_width);
                for i = 1:event_size
                    ev_t = t(i);
                    ev_x = x(i) + 1;
                    ev_y = y(i) + 1;
                    ev_p = pol(i);
                    
                    if ev_t >= t_start + t_shift
                        % reset differenet edge
                        if ev_p > 0 && event_img_large_edge(ev_y, ev_x) < 0 && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge(ev_y, ev_x) = 0;
                        end
                        if ev_p < 0 && event_img_large_edge(ev_y, ev_x) > 0 && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge(ev_y, ev_x) = 0;
                        end
                        % edge accmulation
                        if ev_p > 0 && event_img_large_edge(ev_y, ev_x) >= 0 && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge(ev_y, ev_x) = event_img_large_edge(ev_y, ev_x) + (ev_p) * dvs_contrast;
                        end
                        if ev_p < 0 && event_img_large_edge(ev_y, ev_x) <= 0 && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge(ev_y, ev_x) = event_img_large_edge(ev_y, ev_x) + (ev_p) * dvs_contrast;
                        end
                        % edge detection
                        if event_img_large_edge(ev_y, ev_x) <= -edge_threshold && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge_lock(ev_y, ev_x) = 1;
                        end
                        if event_img_large_edge(ev_y, ev_x) >= edge_threshold && event_img_large_edge_lock(ev_y, ev_x) == 0
                            event_img_large_edge_lock(ev_y, ev_x) = 1;
                        end

                    end
                end
                

                event_binary_map(event_img_large_edge >= edge_threshold) = 0;
                event_binary_map(event_img_large_edge <= -edge_threshold) = 1;
                %% % % % % % % % % % % % % % % % % % % % % % % % % % % %

                new_dir_name = fullfile(pathstr, append(method_name_list(1), "_", name));
                imwrite(event_binary_map, append(new_dir_name, "/", int2str(eventstart_inframe * timescale), ".png"));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if high_rate_frames
                    % generate the following images using direct integration
                    % event -> intensity plus event -> flip the binary image
                    x_list = x_o; y_list = y_o; pol_list = pol_o; t_list = t_o;
                    idx = (t_list >= eventstart_betw_frame) & (t_list <= eventend_betw_frame);
                    % idx = (t>=eventstart_betw_frame+t_shift)&(t<=eventend_betw_frame+t_shift);

                    y_list(idx ~= 1) = [];
                    x_list(idx ~= 1) = [];
                    pol_list(idx ~= 1) = [];
                    t_list(idx ~= 1) = [];
                    event_size = length(t_list);
                    end_time = t_list(end);

                    inc_map = zeros(image_height, image_width);
                    inc_lock = zeros(image_height, image_width);
                    dec_map = zeros(image_height, image_width);
                    dec_lock = zeros(image_height, image_width);
                    set_reset_img = zeros(image_height, image_width);
                    event_binary_map_filtered = medfilt2(event_binary_map);
                    counter = 0;
                    % event_size
                    % break
                    prev_ev_t = t_list(1);
                    if (length(t_list) > 0)
                        for i = 1:event_size
                            ev_t = t_list(i);
                            ev_x = x_list(i) + 1;
                            ev_y = y_list(i) + 1;
                            ev_p = pol_list(i);

                            if ev_p == 1 && inc_lock(ev_y, ev_x) == 0
                                inc_map(ev_y, ev_x) = inc_map(ev_y, ev_x) + ev_p * dvs_contrast;
                            end

                            if ev_p == -1 && dec_lock(ev_y, ev_x) == 0
                                dec_map(ev_y, ev_x) = dec_map(ev_y, ev_x) + ev_p * dvs_contrast;
                            end

                            % if inc_map(ev_y, ev_x) >= lock_thresh && sum(sum(event_binary_map(ev_y-1:ev_y+1, ev_x-1:ev_x+1))~=0)
                            % if inc_map(ev_y, ev_x) >= lock_thresh
                            if inc_map(ev_y, ev_x) >= edge_threshold
                                inc_lock(ev_y, ev_x) = 1;
                                dec_lock(ev_y, ev_x) = 0;
                                inc_map(ev_y, ev_x) = 0;
                                event_binary_map(ev_y, ev_x) = 1;
                                counter = counter + 1;

                                if filtering
                                    % asy median filtering
                                    for x = -half_median_range:half_median_range

                                        for y = -half_median_range:half_median_range
                                            median_x = ev_x + x;
                                            median_y = ev_y + y;

                                            if median_x >= 1 && median_x <= image_width && median_y >= 1 && median_y <= image_height
                                                conv_x_start = median_x - half_median_range;
                                                conv_x_end = median_x + half_median_range;
                                                conv_y_start = median_y - half_median_range;
                                                conv_y_end = median_y + half_median_range;
                                                % boundary is filled by zero, wont affect the judgement
                                                if conv_x_start < 1
                                                    conv_x_start = 1;
                                                end

                                                if conv_x_end > image_width
                                                    conv_x_end = image_width;
                                                end

                                                if conv_y_start < 1
                                                    conv_y_start = 1;
                                                end

                                                if conv_y_end > image_height
                                                    conv_y_end = image_height;
                                                end

                                                summation = sum(sum(event_binary_map(conv_y_start:conv_y_end, conv_x_start:conv_x_end)));

                                                if summation >= floor((median_range * median_range) / 2) + 1
                                                    event_binary_map_filtered(median_y, median_x) = 1;
                                                else
                                                    event_binary_map_filtered(median_y, median_x) = 0;
                                                end

                                            end

                                        end

                                    end

                                end

                                if generation_mode == "event"

                                    if mod(i, per_n_event) == 0
                                        % figure(8)
                                        % imshow(event_binary_map)

                                        dir_name_filtered = fullfile(pathstr, append(method_name_list(3), "_", name));
                                        imwrite(event_binary_map_filtered, append(dir_name_filtered, "/", int2str(ev_t * timescale), ".png"));
                                        dir_name = fullfile(pathstr, append(method_name_list(2), "_", name));
                                        imwrite(event_binary_map, append(dir_name, "/", int2str(ev_t * timescale), ".png"));
                                    end

                                elseif generation_mode == "time"

                                    if ev_t > prev_ev_t + double(1 / generation_frame_rate)
                                        % figure(8)
                                        % imshow(event_binary_map)
                                        % figure(9)
                                        % imshow(medfilt2(event_binary_map,[median_range,median_range]))
                                        % figure(10)
                                        % imshow((event_binary_map_filtered))
                                        prev_ev_t = ev_t;

                                        dir_name_filtered = fullfile(pathstr, append(method_name_list(3), "_", name));
                                        imwrite(event_binary_map_filtered, append(dir_name_filtered, "/", int2str(ev_t * timescale), ".png"));
                                        dir_name = fullfile(pathstr, append(method_name_list(2), "_", name));
                                        imwrite(event_binary_map, append(dir_name, "/", int2str(ev_t * timescale), ".png"));

                                    end
                                 end

                            end

                            if ev_y > 2 && ev_y < image_height - 1 && ev_x > 2 && ev_x < image_width - 1
                                if dec_map(ev_y, ev_x) <= - (edge_threshold)
                                    dec_lock(ev_y, ev_x) = 1;
                                    inc_lock(ev_y, ev_x) = 0;
                                    dec_map(ev_y, ev_x) = 0;
                                    event_binary_map(ev_y, ev_x) = 0;
                                    counter = counter + 1;

                                    if filtering
                                        % asy median filtering
                                        for x = -half_median_range:half_median_range

                                            for y = -half_median_range:half_median_range
                                                median_x = ev_x + x;
                                                median_y = ev_y + y;

                                                if median_x >= 1 && median_x <= image_width && median_y >= 1 && median_y <= image_height
                                                    conv_x_start = median_x - half_median_range;
                                                    conv_x_end = median_x + half_median_range;
                                                    conv_y_start = median_y - half_median_range;
                                                    conv_y_end = median_y + half_median_range;
                                                    % boundary is filled by zero, wont affect the judgement
                                                    if conv_x_start < 1
                                                        conv_x_start = 1;
                                                    end

                                                    if conv_x_end > image_width
                                                        conv_x_end = image_width;
                                                    end

                                                    if conv_y_start < 1
                                                        conv_y_start = 1;
                                                    end

                                                    if conv_y_end > image_height
                                                        conv_y_end = image_height;
                                                    end

                                                    summation = sum(sum(event_binary_map(conv_y_start:conv_y_end, conv_x_start:conv_x_end)));

                                                    if summation >= floor((median_range * median_range) / 2) + 1
                                                        event_binary_map_filtered(median_y, median_x) = 1;
                                                    else
                                                        event_binary_map_filtered(median_y, median_x) = 0;
                                                    end

                                                end

                                            end

                                        end

                                    end

                                    if generation_mode == "event"

                                        if mod(i, per_n_event) == 0
                                            % figure(8)
                                            % imshow(event_binary_map)

                                            dir_name_filtered = fullfile(pathstr, append(method_name_list(3), "_", name));
                                            imwrite(event_binary_map_filtered, append(dir_name_filtered, "/", int2str(ev_t * timescale), ".png"));
                                            dir_name = fullfile(pathstr, append(method_name_list(2), "_", name));
                                            imwrite(event_binary_map, append(dir_name, "/", int2str(ev_t * timescale), ".png"));

                                        end

                                    elseif generation_mode == "time"

                                        if ev_t > prev_ev_t + double(1 / generation_frame_rate)

                                            dir_name_filtered = fullfile(pathstr, append(method_name_list(3), "_", name));
                                            imwrite(event_binary_map_filtered, append(dir_name_filtered, "/", int2str(ev_t * timescale), ".png"));
                                            dir_name = fullfile(pathstr, append(method_name_list(2), "_", name));
                                            imwrite(event_binary_map, append(dir_name, "/", int2str(ev_t * timescale), ".png"));

                                            prev_ev_t = ev_t;
                                        end
                                    end

                                end

                            end

                        end

                    end
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            clearvars aedat
        end

    end

end
