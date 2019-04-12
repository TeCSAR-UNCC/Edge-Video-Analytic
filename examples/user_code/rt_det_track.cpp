#include "reid_core_headers.hpp"
#include "trt_helper.hpp"
#include "reid_structs.hpp"
#include "reid_constants.hpp"
#include "reid_globals.hpp"
#include "reid_comms.hpp"
#include "reid_inference.hpp"
#include "reid_matching.hpp"

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

        const auto poseMode = op::flagsToPoseMode(FLAGS_body);

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
		op::WrapperT<ReIDBBox> opWrapperT; //{op::ThreadManagerMode::Asynchronous};

        // Initializing the user custom classes
        // Processing
        auto reidinf = std::make_shared<ReIDInference>();
        // Add custom processing
        const auto workerProcessingOnNewThread = true;
        opWrapperT.setWorker(op::WorkerType::PostProcessing, reidinf, workerProcessingOnNewThread);

        /********************************************************/
	    // Initializing the user custom classes
        // Processing
        auto reidmatch = std::make_shared<ReIDMatching>(op::flagsToNodeID(FLAGS_node_id));
        // Add custom processing
        opWrapperT.setWorker(op::WorkerType::Output, reidmatch, workerProcessingOnNewThread);

        /********************************************************/
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
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
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
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
    gUseDLACore = op::flagsToDLA(FLAGS_use_dla);
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    onnxToTRTModel("mobileNet_single.onnx", (INPUT_BS), trtModelStream);
    assert(trtModelStream != nullptr);


    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gUseDLACore >= 0)
    {
        runtime->setDLACore(gUseDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);

	colors[0] = {255,0,0};
	colors[1] = {0,255,0};
	colors[2] = {0,0,255};
	colors[3] = {255,255,0};
	colors[4] = {255,0,255};
	colors[5] = {0,255,255};
	colors[6] = {177,177,0};
	colors[7] = {177,0,177};
	colors[8] = {0,177,177};
	colors[9] = {177,177,177};

    /*************************************/
    /*          ADDING COMMS             */
    /*************************************/
    IPADDR = (char *)(FLAGS_server_ip.c_str());
    pthread_t threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_mutex_init(&locks[i], NULL);
    }

    for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&threads[i], NULL, sendData, NULL);
        ++i;
        pthread_create(&threads[i], NULL, recvData, NULL);
    }
    /*************************************/

	for(int i = 0; i < NUM_OBJECTS; ++i) {
		obj_table[i].sendObject.currentCamera = CAMERA_ID;
		obj_table[i].sendObject.label = 0;
		for(int j = 0; j < OUTPUT_SIZE; ++j) {
			obj_table[i].sendObject.fv_array[j] = 0;
		}
		obj_table[i].fv = cv::Mat::zeros(1, OUTPUT_SIZE, CV_32F);
		obj_table[i].sendObject.xPos = 0;
		obj_table[i].sendObject.yPos = 0;
		obj_table[i].sendObject.height = 0;
		obj_table[i].sendObject.width = 0;
		obj_table[i].life = 0;
		obj_table[i].sentToServer = 0;
	}
    generateBoundingBoxes();
	context->destroy();
	engine->destroy();
	runtime->destroy();

	/*************************************/
	/*          ADDING COMMS             */
	/*************************************/
	for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    /*************************************/

	return 0;
}
