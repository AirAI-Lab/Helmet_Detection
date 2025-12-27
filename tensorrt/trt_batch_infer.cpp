#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <unistd.h>
#include <sstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

std::vector<char> readFile(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + filepath);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size))
        throw std::runtime_error("Failed to read file: " + filepath);
    return buffer;
}

cv::Mat preprocess(const cv::Mat &img, int input_w, int input_h)
{
    // letterbox resize: keep aspect ratio, pad with 114
    int orig_w = img.cols, orig_h = img.rows;
    float r = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    int new_w = (int)std::round(orig_w * r);
    int new_h = (int)std::round(orig_h * r);
    cv::Mat resized;
    cv::cvtColor(img, resized, cv::COLOR_BGR2RGB);
    cv::resize(resized, resized, cv::Size(new_w, new_h));
    int dw = input_w - new_w;
    int dh = input_h - new_h;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    return padded;
}

struct Detection
{
    float x1, y1, x2, y2, score;
    int class_id;
};

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cout << "Usage: " << argv[0] << " <engine.trt> <in_frames_or_video> <out_frames_dir> <input_w> <input_h> <names.txt> [--conf 0.25] [--out-video path] [--log-level 0|1|2]" << std::endl;
        return 1;
    }
    std::string engineFile = argv[1];
    std::string in_path = argv[2];
    std::string out_dir = argv[3];
    int input_w = std::stoi(argv[4]);
    int input_h = std::stoi(argv[5]);
    std::string names_path = argv[6];

    float conf_thresh = 0.25f;
    int log_level = 1; // 0=ERROR,1=INFO (default),2=DEBUG
    std::string alarm_dir = "";
    double img_fps = 30.0; // assumed fps for image directories
    bool is_stream = false;
    double max_duration_sec = 0.0;                                   // default duration for stream inputs (0 = run indefinitely)
    double out_fps = 0.0;                                            // optional forced output fps for VideoWriter
    std::string rtmp_url = "rtmp://202.96.165.88/live/allen_9_1209"; // default RTMP target
    int buffer_cap = 8192;                                           // default in-memory buffered frames capacity for stream buffering
    for (int i = 7; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--conf" && i + 1 < argc)
        {
            conf_thresh = std::stof(argv[++i]);
        }
        if (a == "--out-video" && i + 1 < argc)
        { /* handled below via var */
        }
        if ((a == "--log" || a == "--log-level") && i + 1 < argc)
        {
            log_level = std::stoi(argv[++i]);
        }
        if (a == "--alarm-dir" && i + 1 < argc)
        {
            alarm_dir = argv[++i];
        }
        if (a == "--img-fps" && i + 1 < argc)
        {
            img_fps = std::stod(argv[++i]);
        }
        if (a == "--duration" && i + 1 < argc)
        {
            max_duration_sec = std::stod(argv[++i]);
        }
        if (a == "--out-fps" && i + 1 < argc)
        {
            out_fps = std::stod(argv[++i]);
        }
        if (a == "--rtmp" && i + 1 < argc)
        {
            rtmp_url = argv[++i];
        }
        if (a == "--buffer-cap" && i + 1 < argc)
        {
            buffer_cap = std::stoi(argv[++i]);
        }
    }
    // rtsp://admin:123456@192.168.2.108:554/h265/ch1/main/av_stream
    std::vector<std::string> class_names;
    if (!names_path.empty())
    {
        std::ifstream nf(names_path);
        if (nf)
        {
            std::string line;
            while (std::getline(nf, line))
            {
                if (!line.empty())
                {
                    // this is a 'have to do' produce.
                    // your name.txt might be read end to one more length
                    // std::cout << line.length() << std::endl;
                    std::string new_line = line.substr(0, line.length() - 1);
                    class_names.push_back(new_line);
                    // std::cout << new_line.length() << std::endl;
                }
            }
        }
    }

    // detect whether input is a directory of frames or a single video file
    bool video_mode = false;
    std::vector<std::string> files;

    // 先转换为小写方便检测
    std::string input_lower = in_path;
    for (auto &c : input_lower)
        c = std::tolower(c);

    // 检查是否是网络流协议（先于文件系统检查）
    bool is_network_stream = false;
    std::set<std::string> stream_protocols = {"rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://", "hls://"};

    for (const auto &protocol : stream_protocols)
    {
        if (input_lower.rfind(protocol, 0) == 0)
        {
            is_network_stream = true;
            break;
        }
    }

    if (is_network_stream)
    {
        // 网络流输入
        video_mode = true;
        is_stream = true;
        if (log_level >= 1)
        {
            std::cout << "Detected network stream input: " << in_path << std::endl;
            std::cout << "Supported stream protocols: RTSP, RTMP, HTTP, HTTPS, UDP, TCP, HLS" << std::endl;
        }
    }
    else if (std::filesystem::is_directory(in_path))
    {
        // 目录处理（图片文件夹）
        for (const auto &p : std::filesystem::directory_iterator(in_path))
        {
            if (!p.is_regular_file())
                continue;
            std::string s = p.path().string();
            std::string low = s;
            for (auto &c : low)
                c = std::tolower(c);
            if (low.find(".png") != std::string::npos ||
                low.find(".jpg") != std::string::npos ||
                low.find(".jpeg") != std::string::npos ||
                low.find(".bmp") != std::string::npos)
                files.push_back(s);
        }
        std::sort(files.begin(), files.end());
        if (files.empty())
        {
            std::cerr << "No input frames in: " << in_path << std::endl;
            return 2;
        }
    }
    else if (std::filesystem::is_regular_file(in_path))
    {
        // 单个文件：可能是视频文件或图片文件
        if (input_lower.find(".mp4") != std::string::npos ||
            input_lower.find(".avi") != std::string::npos ||
            input_lower.find(".mov") != std::string::npos ||
            input_lower.find(".mkv") != std::string::npos ||
            input_lower.find(".flv") != std::string::npos)
        {
            video_mode = true;
        }
        // 添加图片文件支持
        else if (input_lower.find(".png") != std::string::npos ||
                 input_lower.find(".jpg") != std::string::npos ||
                 input_lower.find(".jpeg") != std::string::npos ||
                 input_lower.find(".bmp") != std::string::npos)
        {
            files.push_back(in_path);
            if (log_level >= 1)
                std::cout << "Processing single image file: " << in_path << std::endl;
        }
        else
        {
            std::cerr << "Input file format not recognized: " << in_path << std::endl;
            std::cerr << "Supported: .mp4, .avi, .mov, .mkv, .flv, .png, .jpg, .jpeg, .bmp" << std::endl;
            return 2;
        }
    }
    else
    {
        // 既不是流协议，也不是存在的文件/目录
        std::cerr << "Input path does not exist or is not accessible: " << in_path << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "1. Network streams: rtsp://, rtmp://, http://, etc." << std::endl;
        std::cerr << "2. Local video files: .mp4, .avi, .mov, .mkv" << std::endl;
        std::cerr << "3. Local image files: .png, .jpg, .jpeg, .bmp" << std::endl;
        std::cerr << "4. Image directories" << std::endl;
        return 2;
    }

    std::filesystem::create_directories(out_dir);
    if (alarm_dir.empty())
        alarm_dir = out_dir + "/alarms";
    std::filesystem::create_directories(alarm_dir);

    // load engine once
    auto engine_data = readFile(engineFile);
    IRuntime *runtime = createInferRuntime(gLogger);
    if (!runtime)
    {
        std::cerr << "Failed to create TensorRT runtime\n";
        return 3;
    }
    ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine)
    {
        std::cerr << "Failed to deserialize engine\n";
        return 4;
    }
    IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create execution context\n";
        return 5;
    }

    int nbIO = engine->getNbIOTensors();
    int inputIndex = -1;
    std::vector<int> io_indices(nbIO, -1);
    for (int i = 0; i < nbIO; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (!name)
            continue;
        TensorIOMode mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
            inputIndex = i;
        io_indices[i] = i;
        if (log_level >= 1)
            std::cout << "IO[" << i << "] name='" << name << "' mode=" << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;
    }
    if (inputIndex < 0)
    {
        std::cerr << "No input IO found\n";
        return 6;
    }

    // allocate buffers
    std::vector<void *> buffers(nbIO, nullptr);
    std::vector<size_t> elem_counts(nbIO, 0);
    for (int i = 0; i < nbIO; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (!name)
            continue;
        Dims shape = engine->getTensorShape(name);
        int64_t elem = 1;
        if (shape.nbDims > 0)
        {
            for (int d = 0; d < shape.nbDims; ++d)
            {
                int dim = shape.d[d];
                if (dim <= 0)
                {
                    elem = (i == inputIndex) ? (3LL * input_w * input_h) : 100000;
                    break;
                }
                elem *= dim;
            }
        }
        else
            elem = (i == inputIndex) ? (3LL * input_w * input_h) : 100000;
        int32_t bpc = engine->getTensorBytesPerComponent(name);
        if (bpc <= 0)
            bpc = 4;
        size_t bytes = static_cast<size_t>(elem) * static_cast<size_t>(bpc);
        cudaError_t cerr = cudaMalloc(&buffers[i], bytes);
        if (cerr != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed for " << name << " bytes=" << bytes << " " << cudaGetErrorString(cerr) << std::endl;
            return 7;
        }
        elem_counts[i] = static_cast<size_t>(elem);
        if (log_level >= 1)
            std::cout << "Allocated IO[" << i << "] '" << name << "' bytes=" << bytes << std::endl;
    }

    // find combined output index if available
    int combinedOutputIndex = -1;
    const char *combinedName = nullptr;
    for (int i = 0; i < nbIO; ++i)
    {
        const char *nm = engine->getIOTensorName(i);
        if (nm && std::string(nm) == "output")
        {
            combinedOutputIndex = i;
            combinedName = nm;
            break;
        }
    }
    if (combinedOutputIndex >= 0 && log_level >= 1)
        std::cout << "Using combined output: '" << combinedName << "' (IO index=" << combinedOutputIndex << ")\n";

    // optionally open video writer if requested via argv
    std::string out_video_path;
    for (int i = 7; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--out-video" && i + 1 < argc)
        {
            out_video_path = argv[++i];
        }
    }
    cv::VideoWriter video_writer;

    auto try_open_video_writer = [&](cv::VideoWriter &wri, const std::string &path, double fps, cv::Size size) -> bool
    {
        // Candidate codecs (FourCC) to try in order of preference
        std::vector<std::pair<std::string, int>> cands = {
            {"libx264(avc1)", cv::VideoWriter::fourcc('a', 'v', 'c', '1')},
            {"libx264(X264)", cv::VideoWriter::fourcc('X', '2', '6', '4')},
            {"H264", cv::VideoWriter::fourcc('H', '2', '6', '4')},
            {"mp4v", cv::VideoWriter::fourcc('m', 'p', '4', 'v')},
            {"MJPG", cv::VideoWriter::fourcc('M', 'J', 'P', 'G')}};
        for (auto &p : cands)
        {
            const std::string &name = p.first;
            int fourcc = p.second;
            wri.open(path, fourcc, fps, size);
            if (wri.isOpened())
            {
                if (log_level >= 1)
                    std::cout << "Opened video writer with codec: " << name << " for file: " << path << std::endl;
                return true;
            }
        }
        std::cerr << "Failed to open any video codec for: " << path << std::endl;
        return false;
    };

    // process frames (either from image list or from video capture)
    cv::VideoCapture cap;
    if (video_mode)
    {
        cap.open(in_path);
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open video: " << in_path << std::endl;
            return 8;
        }
    }

    double video_fps = img_fps;
    if (video_mode)
    {
        double vfps = cap.get(cv::CAP_PROP_FPS);
        if (vfps > 1.0)
            video_fps = vfps;
    }

    // buffer mode: for stream inputs with a specified duration and no forced out_fps,
    // collect all annotated frames in memory (or disk-backed) and write the output video after capture
    bool buffer_mode = (video_mode && is_stream && max_duration_sec > 0.0 && out_fps <= 0.0 && !out_video_path.empty());
    // Disk-backed buffer: when buffer_mode is true we write frames to a temp dir instead of keeping them in memory.
    std::string buffer_dir;
    size_t buffer_count = 0;
    if (buffer_mode)
    {
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        std::ostringstream ss;
        ss << out_dir << "/buffer_tmp_" << now << "_" << getpid();
        buffer_dir = ss.str();
        std::filesystem::create_directories(buffer_dir);
        if (log_level >= 1)
            std::cout << "Using disk-backed buffer dir: " << buffer_dir << std::endl;
    }

    // alarm control: save at most 1 frame per second when alarm occurs
    double last_alarm_time = -1e9;
    std::set<std::string> alarm_names = {"no_vest", "head"};

    size_t frame_idx = 0;
    // prebuffer to estimate real capture fps for streams when --out-fps not specified
    std::vector<cv::Mat> prebuf_frames;
    auto prebuf_start = std::chrono::steady_clock::time_point();
    bool prebufing = false;
    const size_t prebuf_target = 30;   // frames
    const double prebuf_seconds = 1.0; // seconds
    auto stream_start_time = std::chrono::steady_clock::time_point();
    bool stream_started = false;
    FILE *ffmpeg_pipe = nullptr;
    for (;;)
    {
        cv::Mat frame;
        if (video_mode)
        {
            if (!cap.read(frame))
                break; // end of video
        }
        else
        {
            if (frame_idx >= files.size())
                break;
            std::string fpath = files[frame_idx];
            frame = cv::imread(fpath);
            if (frame.empty())
            {
                std::cerr << "Failed read " << fpath << "\n";
                frame_idx++;
                continue;
            }
        }
        // if this is a stream URL, enforce max duration
        if (video_mode && is_stream)
        {
            if (!stream_started)
            {
                stream_start_time = std::chrono::steady_clock::now();
                stream_started = true;
            }
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - stream_start_time).count();
            // only stop when a positive duration was requested; otherwise run indefinitely
            if (max_duration_sec > 0.0 && elapsed >= max_duration_sec)
            {
                if (log_level >= 1)
                    std::cout << "Stream duration reached: " << elapsed << "s, stopping." << std::endl;
                break;
            }
        }
        size_t fi = frame_idx;
        if (!out_video_path.empty() && !video_writer.isOpened() && !buffer_mode)
        {
            cv::Size sz(frame.cols, frame.rows);
            double fps = 29.0;
            if (video_mode)
            {
                double vfps = cap.get(cv::CAP_PROP_FPS);
                if (vfps > 1.0)
                    fps = vfps;
            }
            // override fps if user requested specific output fps
            if (out_fps > 0.0)
                fps = out_fps;
            // try open with multiple codecs
            if (!try_open_video_writer(video_writer, out_video_path, fps, sz))
            {
                if (log_level >= 0)
                    std::cerr << "Warning: failed to open any video writer for: " << out_video_path << ", will continue writing frames only." << std::endl;
            }
        }
        if (frame.empty())
        {
            frame_idx++;
            continue;
        }
        cv::Mat in = preprocess(frame, input_w, input_h);
        size_t hw = input_w * input_h;
        std::vector<cv::Mat> chs;
        cv::split(in, chs);
        size_t input_elems = 3 * input_h * input_w;
        std::vector<float> hostInput(input_elems);
        for (int c = 0; c < 3; ++c)
            memcpy(hostInput.data() + c * hw, chs[c].data, hw * sizeof(float));
        cudaMemcpy(buffers[inputIndex], hostInput.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice);

        if (!context->executeV2(buffers.data()))
        {
            std::cerr << "executeV2 failed on frame " << fi << "\n";
            break;
        }

        std::vector<Detection> dets;
        if (combinedOutputIndex >= 0)
        {
            // read shape
            Dims combShape = engine->getTensorShape(combinedName);
            int C = 0, L = 0;
            if (combShape.nbDims == 3)
            {
                C = combShape.d[1];
                L = combShape.d[2];
            }
            else if (combShape.nbDims == 2)
            {
                C = combShape.d[0];
                L = combShape.d[1];
            }
            if (C > 4 && L > 0)
            {
                size_t outElems = static_cast<size_t>(C) * static_cast<size_t>(L);
                std::vector<float> hostOutput(outElems);
                cudaMemcpy(hostOutput.data(), buffers[combinedOutputIndex], outElems * sizeof(float), cudaMemcpyDeviceToHost);
                int num_classes = C - 4;
                for (int i = 0; i < L; ++i)
                {
                    float cx = hostOutput[0 * L + i];
                    float cy = hostOutput[1 * L + i];
                    float w = hostOutput[2 * L + i];
                    float h = hostOutput[3 * L + i];
                    float best_score = -1e9f;
                    int best_class = -1;
                    for (int c = 0; c < num_classes; ++c)
                    {
                        float prob = hostOutput[(4 + c) * L + i];
                        if (prob > best_score)
                        {
                            best_score = prob;
                            best_class = c;
                        }
                    }
                    if (best_score > 1.5f || best_score < -0.5f)
                        best_score = 1.0f / (1.0f + std::exp(-best_score));
                    if (best_score < conf_thresh)
                        continue;
                    // box coordinates are in model input space (with letterbox pad). Map back to original frame coordinates.
                    // Compute letterbox params used during preprocess
                    int orig_w = frame.cols, orig_h = frame.rows;
                    float r = std::min((float)input_w / orig_w, (float)input_h / orig_h);
                    int new_w = (int)std::round(orig_w * r);
                    int new_h = (int)std::round(orig_h * r);
                    int dw = input_w - new_w;
                    int dh = input_h - new_h;
                    float pad_x = dw / 2.0f;
                    float pad_y = dh / 2.0f;
                    auto box = std::array<float, 4>{cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f};
                    float x1 = (box[0] - pad_x) / r;
                    float y1 = (box[1] - pad_y) / r;
                    float x2 = (box[2] - pad_x) / r;
                    float y2 = (box[3] - pad_y) / r;
                    Detection d{x1, y1, x2, y2, best_score, best_class};
                    dets.push_back(d);
                }
            }
        }

        // NMS
        std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b)
                  { return a.score > b.score; });
        std::vector<Detection> final_dets;
        std::vector<bool> suppressed(dets.size(), false);
        for (size_t i = 0; i < dets.size(); ++i)
        {
            if (suppressed[i])
                continue;
            final_dets.push_back(dets[i]);
            for (size_t j = i + 1; j < dets.size(); ++j)
            {
                if (suppressed[j])
                    continue;
                if (dets[i].class_id != dets[j].class_id)
                    continue;
                float inter_x1 = std::max(dets[i].x1, dets[j].x1);
                float inter_y1 = std::max(dets[i].y1, dets[j].y1);
                float inter_x2 = std::min(dets[i].x2, dets[j].x2);
                float inter_y2 = std::min(dets[i].y2, dets[j].y2);
                float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                float inter = inter_w * inter_h;
                float areaA = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
                float areaB = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
                float iou = inter / (areaA + areaB - inter + 1e-6f);
                if (iou > 0.45f)
                    suppressed[j] = true;
            }
        }

        // prepare per-frame printout similar to infer_helmet_vest.py
        if (video_mode)
        {
            double t_msec = cap.get(cv::CAP_PROP_POS_MSEC);
            if (log_level >= 1)
                std::cout << "Frame: " << fi << " time_ms: " << t_msec << std::endl;
        }
        else
        {
            if (log_level >= 1)
                std::cout << "File: " << files[fi] << std::endl;
        }

        bool alarm = false;
        double current_time_sec = 0.0;
        if (video_mode)
            current_time_sec = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        else
            current_time_sec = fi / std::max(1.0, video_fps);

        for (const auto &d : final_dets)
        {
            std::string cls_name = (d.class_id >= 0 && d.class_id < (int)class_names.size()) ? class_names[d.class_id] : std::to_string(d.class_id);
            // determine color: red for alarm classes, green otherwise
            bool is_alarm_class = (alarm_names.find(cls_name) != alarm_names.end());

            if (is_alarm_class)
                alarm = true;
            cv::Scalar color = is_alarm_class ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2), color, 2);
            char lbl[128];
            snprintf(lbl, sizeof(lbl), "%s:%.2f", cls_name.c_str(), d.score);
            cv::putText(frame, lbl, cv::Point(std::max(0, (int)d.x1), std::max(15, (int)d.y1) - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            // print detection line similar to Python
            if (log_level >= 1)
            {
                int x1 = (int)std::round(d.x1), y1 = (int)std::round(d.y1), x2 = (int)std::round(d.x2), y2 = (int)std::round(d.y2);
                std::cout << "  Class: " << cls_name << ", Conf: " << std::fixed << std::setprecision(2) << d.score << ", Box: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
            }
        }

        // if alarm detected, save one frame per second into alarm_dir
        if (alarm)
        {
            if (current_time_sec - last_alarm_time >= 1.0)
            {
                char alarm_fn[4096];
                if (video_mode)
                    snprintf(alarm_fn, sizeof(alarm_fn), "%s/alarm_t%06.0f_f%06zu.png", alarm_dir.c_str(), current_time_sec, fi + 1);
                else
                    snprintf(alarm_fn, sizeof(alarm_fn), "%s/alarm_f%06zu.png", alarm_dir.c_str(), fi + 1);
                if (!cv::imwrite(alarm_fn, frame))
                    std::cerr << "Failed to write alarm frame: " << alarm_fn << std::endl;
                else if (log_level >= 1)
                    std::cout << "Saved alarm frame: " << alarm_fn << std::endl;
                last_alarm_time = current_time_sec;
            }
            else
            {
                if (log_level >= 2)
                    std::cout << "Alarm suppressed by rate limit at t=" << current_time_sec << "s" << std::endl;
            }
        }

        // // write annotated frame to disk
        // char outfn[4096];
        // snprintf(outfn, sizeof(outfn), "%s/frame_%06zu.png", out_dir.c_str(), fi + 1);
        // if (!cv::imwrite(outfn, frame))
        //     std::cerr << "Failed to write: " << outfn << std::endl;
        // else if ((fi + 1) % 50 == 0 && log_level >= 1)
        //     std::cout << "Wrote " << (fi + 1) << " frames" << std::endl;

        //---------------------------------------------------------
        if ((fi + 1) % 50 == 0 && log_level >= 1)
            std::cout << "Wrote " << (fi + 1) << " frames" << std::endl;
        
        //---------------------------------------------------------

        // If this is a stream and an rtmp target was provided, start ffmpeg pipe on first frame

        // if (video_mode && is_stream && !rtmp_url.empty()) // 修改了，现在只要是视频模式就推流
        if (video_mode && !rtmp_url.empty())
        {
            if (!ffmpeg_pipe)
            {
                double push_fps = out_fps > 0.0 ? out_fps : (video_fps > 1.0 ? video_fps : 25.0);
                int w = frame.cols, h = frame.rows;
                char cmd[4096];
                // Use rawvideo input via stdin in bgr24 pixel format
                snprintf(cmd, sizeof(cmd), "ffmpeg -y -f rawvideo -pix_fmt bgr24 -s %dx%d -r %.2f -i - -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -f flv \"%s\"", w, h, push_fps, rtmp_url.c_str());
                if (log_level >= 1)
                    std::cout << "Starting ffmpeg push: " << cmd << std::endl;
                ffmpeg_pipe = popen(cmd, "w");
                if (!ffmpeg_pipe)
                {
                    std::cerr << "Failed to start ffmpeg for rtmp push" << std::endl;
                }
            }
            if (ffmpeg_pipe)
            {
                cv::Mat write_mat = frame;
                if (!write_mat.isContinuous())
                    write_mat = frame.clone();
                size_t bytes = write_mat.total() * write_mat.elemSize();
                size_t wrote = fwrite(write_mat.data, 1, bytes, ffmpeg_pipe);
                if (wrote != bytes)
                {
                    if (log_level >= 0)
                        std::cerr << "Warning: incomplete write to ffmpeg pipe: " << wrote << " bytes\n";
                }
                fflush(ffmpeg_pipe);
            }
        }
        // write to video if open. If buffer_mode is enabled, collect frames and write after loop.
        if (buffer_mode)
        {
            // write frame to disk-backed buffer
            char buffn[4096];
            snprintf(buffn, sizeof(buffn), "%s/buf_%08zu.png", buffer_dir.c_str(), buffer_count);
            if (!cv::imwrite(buffn, frame))
            {
                std::cerr << "Failed to write buffer frame: " << buffn << std::endl;
            }
            else
            {
                buffer_count++;
            }
        }
        else
        {
            if (video_writer.isOpened())
                video_writer.write(frame);
        }

        frame_idx++;
    }

    // If we used disk-backed buffer_mode, assemble buffered PNGs into the output video now
    if (buffer_mode && buffer_count > 0)
    {
        double fps = std::max(1.0, static_cast<double>(buffer_count) / std::max(1e-6, max_duration_sec));
        if (log_level >= 1)
            std::cout << "Flushing " << buffer_count << " buffered frames from " << buffer_dir << " to video at fps=" << fps << std::endl;
        // Use ffmpeg to assemble PNG sequence into mp4
        // Command: ffmpeg -y -framerate <fps> -i buffer_dir/buf_%08d.png -c:v libx264 -pix_fmt yuv420p out_video_path
        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "ffmpeg -y -framerate %.6f -i %s/buf_%%08d.png -c:v libx264 -pix_fmt yuv420p \"%s\"", fps, buffer_dir.c_str(), out_video_path.c_str());
        if (log_level >= 1)
            std::cout << "Running: " << cmd << std::endl;
        int rc = system(cmd);
        if (rc != 0)
            std::cerr << "ffmpeg returned non-zero exit code: " << rc << std::endl;
        // cleanup buffer files
        try
        {
            for (const auto &p : std::filesystem::directory_iterator(buffer_dir))
                std::filesystem::remove(p.path());
            std::filesystem::remove(buffer_dir);
        }
        catch (...)
        {
            if (log_level >= 1)
                std::cerr << "Warning: failed to fully remove buffer dir: " << buffer_dir << std::endl;
        }
    }

    // cleanup
    for (int i = 0; i < nbIO; ++i)
        if (buffers[i])
            cudaFree(buffers[i]);
    delete context;
    delete engine;
    delete runtime;
    if (video_writer.isOpened())
        video_writer.release();
    if (cap.isOpened())
        cap.release();
    if (ffmpeg_pipe)
    {
        if (log_level >= 1)
            std::cout << "Closing ffmpeg pipe" << std::endl;
        pclose(ffmpeg_pipe);
    }
    std::cout << "Batch inference done. Frames written to: " << out_dir << std::endl;
    return 0;
}
