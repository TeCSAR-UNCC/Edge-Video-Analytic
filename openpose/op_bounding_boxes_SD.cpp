// Command-line user intraface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <opencv2/core/core.hpp>

struct UserDatum : public op::Datum
{
	std::vector<cv::Rect> personRectangle;
	std::vector<cv::Mat> roi;
};

// This worker will just invert the image
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr)
    {
        try
        {
            if (datumsPtr != nullptr && !datumsPtr->empty()) {
                // Show in command line the resulting pose keypoints for body, face and hands
                timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
                op::log(std::to_string(ts.tv_nsec));
                op::log("\nKeypoints:" + std::to_string(datumsPtr->size()));
                // Accesing each element of the keypoints
                auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;
                op::log("Person pose keypoints:");
                auto thresholdRectangle = 0.1f;
				std::cout << datumsPtr->at(0)->scaleInputToOutput << std::endl;
                for (auto person = 0 ; person < poseKeypoints.getSize(0); person++)
                {
					auto rectBuffer = op::getKeypointsRectangle(poseKeypoints,person, thresholdRectangle); //Gets the rectangle information from the keypoints
					cv::Rect rec(rectBuffer.x,rectBuffer.y,rectBuffer.width,rectBuffer.height); //Converts from openpose rectangle to opencv rectangle
					datumsPtr->at(0)->personRectangle.push_back(rec); //Inserts the rectangle into the datums vector: personRectangle.
                }
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

// This worker will just read and return all the jpg files in a directory
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    void initializationOnThread() {}

    void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr)
    {
        try
        {
            if (datumsPtr != nullptr && !datumsPtr->empty()) {
                for (auto person = 0 ; person < datumsPtr->at(0)->personRectangle.size(); person++)
                {
					//A complicated if statement to make sure that a random detection is not made
					if((datumsPtr->at(0)->personRectangle[person].width > 0) && (datumsPtr->at(0)->personRectangle[person].height > 0) && (datumsPtr->at(0)->personRectangle[person].x < datumsPtr->at(0)->cvOutputData.cols) && (datumsPtr->at(0)->personRectangle[person].y < datumsPtr->at(0)->cvOutputData.rows)) {
						
						//Function to make sure the region of interest is inside the ouput image. [Warning!] Didn't take into account of a stray detection being made in the lower right corner, that's why the complicated if statment is needed above
						op::keepRoiInside(datumsPtr->at(0)->personRectangle[person],datumsPtr->at(0)->cvOutputData.cols,datumsPtr->at(0)->cvOutputData.rows);
						std::cout << "Person " << person << " Bounding box: [x y width height]" << std::endl; //Output the bounding box data
						std::cout << "[" << datumsPtr->at(0)->personRectangle[person].x << " " << datumsPtr->at(0)->personRectangle[person].y << " " << datumsPtr->at(0)->personRectangle[person].width << " " << datumsPtr->at(0)->personRectangle[person].height << "]" << std::endl;
						cv::rectangle(datumsPtr->at(0)->cvOutputData, datumsPtr->at(0)->personRectangle[person], cv::Scalar(0,255,0)); //Draws the bounding box around the peson of interest
						datumsPtr->at(0)->roi.push_back(datumsPtr->at(0)->cvInputData(datumsPtr->at(0)->personRectangle[person])); //Inserts the region inside the bounding box inside the vector: roi
						
						//Uncommment to show the region in the bounding box. [Warning!] Breaks easily (Kinda safe with 1 person, kinda).
						/*
						cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
						cv::imshow("Test", datumsPtr->at(0)->roi[person]);
						/**/
					}
                }
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

int generateBoundingBoxes()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);
        // // For debugging
        // // Print all logging messages
        // op::ConfigureLog::setPriorityThreshold(op::Priority::None);
        // // Print out speed values faster
        // op::Profiler::setDefaultX(100);

        // Applying user defined configuration - GFlags to program variables
        // cameraSize
        const auto cameraSize = op::flagsToPoint(FLAGS_camera_resolution, "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // producerType
        op::ProducerType producerType;
        std::string producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera, FLAGS_flir_camera, FLAGS_flir_camera_index);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
		op::WrapperT<UserDatum> opWrapperT;

        // Initializing the user custom classes
        // Processing
        auto wUserPostProcessing = std::make_shared<WUserPostProcessing>();
        // Add custom processing
        const auto workerProcessingOnNewThread = true;
        opWrapperT.setWorker(op::WorkerType::PostProcessing, wUserPostProcessing, workerProcessingOnNewThread);

/********************************************************/
	// Initializing the user custom classes
        // Processing
        auto wUserOutput = std::make_shared<WUserOutput>();
        // Add custom processing
        opWrapperT.setWorker(op::WorkerType::Output, wUserOutput, workerProcessingOnNewThread);

/********************************************************/
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            !FLAGS_body_disable, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, enableGoogleLogging};
        opWrapperT.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{};
        opWrapperT.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{};
        opWrapperT.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{};
        opWrapperT.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapperT.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_foot_json, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapperT.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapperT.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapperT.disableMultiThreading();
        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapperT.exec();

        // Measuring total time
        const auto now = std::chrono::high_resolution_clock::now();
        const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                                * 1e-9;
        const auto message = "OpenPose demo successfully finished. Total time: "
                           + std::to_string(totalTimeSec) + " seconds.";
        op::log(message, op::Priority::High);

        // Return successful message
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Running openPoseDemo
    return generateBoundingBoxes();
}
