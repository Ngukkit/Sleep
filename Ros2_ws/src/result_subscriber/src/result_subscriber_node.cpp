#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>
#include <iostream>

class ResultSubscriber : public rclcpp::Node {
public:
    ResultSubscriber() : Node("result_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "result_topic", 10,
            [this](std_msgs::msg::String::UniquePtr msg) {
                try {
                    auto j = nlohmann::json::parse(msg->data);
                    // === result_server.cpp의 파싱/출력 코드 복붙 ===
                    // YOLO 정보
                    if (j.contains("yolo") && !j["yolo"].is_null() && !j["yolo"].empty()) {
                        std::cout << "[YOLO] Detections:" << std::endl;
                        for (const auto& det : j["yolo"]) {
                            std::cout << "  - Class: " << det.value("class_name", "N/A")
                                      << ", Confidence: " << det.value("conf", 0.0) 
                                      << std::endl;
                        }
                    }
                    // Dlib 정보
                    if (j.contains("dlib") && !j["dlib"].is_null() && !j["dlib"].empty()) {
                        auto& d = j["dlib"];
                        if (d.value("is_drowsy_ear", false))
                            std::cout << "[Dlib] 상태: DROWSY" << std::endl;
                        if (d.value("is_yawning", false))
                            std::cout << "[Dlib] 상태: YAWNING" << std::endl;
                        if (d.value("is_gaze", false))
                            std::cout << "[Dlib] 상태: LOOK AHEAD" << std::endl;
                        if (d.value("is_distracted_from_front", false))
                            std::cout << "[Dlib] 상태: DISTRACTED FROM FRONT" << std::endl;
                        if (d.value("is_head_down", false))
                            std::cout << "[Dlib] 상태: HEAD DOWN" << std::endl;
                        if (d.value("is_head_up", false))
                            std::cout << "[Dlib] 상태: HEAD UP" << std::endl;
                        if (d.value("is_dangerous_condition", false))
                            std::cout << "[Dlib] ⚠️  DANGER: Eyes Closed + Head Down!" << std::endl;
                        if (d.value("is_drowsy_ear", false) && d.value("drowsy_frame_count", 0) >= d.value("wakeup_frame_threshold", 60))
                            std::cout << "[Dlib] ⚠️  Wake UP" << std::endl;
                        if (d.value("is_distracted_from_front", false) && d.value("distracted_frame_count", 0) >= d.value("distracted_frame_threshold", 60))
                            std::cout << "[Dlib] ⚠️  Please look forward" << std::endl;
                    }
                    // MediaPipe 정보
                    if (j.contains("mediapipe") && !j["mediapipe"].is_null() && !j["mediapipe"].empty()) {
                        auto& m = j["mediapipe"];
                        int drowsy_frame_count = m.value("drowsy_frame_count", 0);
                        int wakeup_frame_threshold = m.value("wakeup_frame_threshold", 60);
                        int distracted_frame_count = m.value("distracted_frame_count", 0);
                        int distracted_frame_threshold = m.value("distracted_frame_threshold", 60);
                        std::string danger_msg = m.value("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!");
                        bool mp_danger = m.value("is_dangerous_condition", false);
                        bool mp_hands_off = m.value("is_hands_off_warning", false);
                        bool printed = false;
                        if (mp_danger) {
                            std::cout << "----------------------------------------" << std::endl;
                            std::cout << "[result_server] Client connected." << std::endl;
                            std::cout << "[MediaPipe] ⚠️  " << danger_msg << std::endl;
                            printed = true;
                        }
                        if (drowsy_frame_count >= wakeup_frame_threshold) {
                            if (!printed) {
                                std::cout << "----------------------------------------" << std::endl;
                                std::cout << "[result_server] Client connected." << std::endl;
                                printed = true;
                            }
                            std::cout << "[MediaPipe] ⚠️  Wake UP" << std::endl;
                        }
                        if (distracted_frame_count >= distracted_frame_threshold) {
                            if (!printed) {
                                std::cout << "----------------------------------------" << std::endl;
                                std::cout << "[result_server] Client connected." << std::endl;
                                printed = true;
                            }
                            std::cout << "[MediaPipe] ⚠️  Please look forward" << std::endl;
                        }
                        if (mp_hands_off) {
                            if (!printed) {
                                std::cout << "----------------------------------------" << std::endl;
                                std::cout << "[result_server] Client connected." << std::endl;
                                printed = true;
                            }
                            std::cout << "[MediaPipe] ⚠️  Please hold the steering wheel" << std::endl;
                        }
                        if (!mp_danger && drowsy_frame_count < wakeup_frame_threshold && distracted_frame_count < distracted_frame_threshold && !mp_hands_off) {
                            std::cout << "----------------------------------------" << std::endl;
                            std::cout << "[result_server] Client connected." << std::endl;
                            std::cout << "[MediaPipe] 상태: NORMAL" << std::endl;
                        }
                        if (m.value("is_drowsy", false))
                            std::cout << "[MediaPipe] 상태: DROWSY" << std::endl;
                        if (m.value("is_yawning", false))
                            std::cout << "[MediaPipe] 상태: YAWNING" << std::endl;
                        if (m.value("is_gaze", false))
                            std::cout << "[MediaPipe] 상태: LOOK AHEAD" << std::endl;
                        if (m.value("is_distracted_from_front", false))
                            std::cout << "[MediaPipe] 상태: DISTRACTED FROM FRONT" << std::endl;
                        if (m.value("is_head_down", false))
                            std::cout << "[MediaPipe] 상태: HEAD DOWN" << std::endl;
                        if (m.value("is_head_up", false))
                            std::cout << "[MediaPipe] 상태: HEAD UP" << std::endl;
                        if (m.value("is_left_hand_off", false))
                            std::cout << "[MediaPipe] 상태: LEFT HAND OFF" << std::endl;
                        if (m.value("is_right_hand_off", false))
                            std::cout << "[MediaPipe] 상태: RIGHT HAND OFF" << std::endl;
                        if (m.value("is_left_hand_off", false) && m.value("is_right_hand_off", false))
                            std::cout << "[MediaPipe] ⚠️  Please hold the steering wheel" << std::endl;
                    }
                    // OpenVINO 정보
                    if (j.contains("openvino") && !j["openvino"].is_null() && !j["openvino"].empty()) {
                        auto& o = j["openvino"];
                        if (o.contains("faces") && !o["faces"].empty()) {
                            auto& face = o["faces"][0];
                            if (face.value("is_drowsy", false))
                                std::cout << "[OpenVINO] 상태: DROWSY" << std::endl;
                            if (face.value("is_yawning", false))
                                std::cout << "[OpenVINO] 상태: YAWNING" << std::endl;
                            if (face.value("is_gaze", false))
                                std::cout << "[OpenVINO] 상태: LOOK AHEAD" << std::endl;
                            if (face.value("is_distracted", false))
                                std::cout << "[OpenVINO] 상태: DISTRACTED" << std::endl;
                            if (face.value("is_head_down", false))
                                std::cout << "[OpenVINO] 상태: HEAD DOWN" << std::endl;
                            if (face.value("is_head_up", false))
                                std::cout << "[OpenVINO] 상태: HEAD UP" << std::endl;
                            if (face.value("is_dangerous_condition", false))
                                std::cout << "[OpenVINO] ⚠️  DANGER: Eyes Closed + Head Down!" << std::endl;
                            if (face.value("is_drowsy", false) && face.value("drowsy_frame_count", 0) >= face.value("wakeup_frame_threshold", 60))
                                std::cout << "[OpenVINO] ⚠️  Wake UP" << std::endl;
                            if (face.value("is_distracted", false) && face.value("distracted_frame_count", 0) >= face.value("distracted_frame_threshold", 60))
                                std::cout << "[OpenVINO] ⚠️  Please look forward" << std::endl;
                        }
                    }
                    // 최종 상태
                    std::string status = j.value("status", "N/A");
                    std::cout << "[Status] " << status << std::endl;
                } catch (const nlohmann::json::parse_error& e) {
                    std::cout << "[ROS2] JSON parse error: " << e.what() << std::endl;
                }
            }
        );
    }
private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ResultSubscriber>());
    rclcpp::shutdown();
    return 0;
}
