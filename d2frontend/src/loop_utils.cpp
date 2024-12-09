#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_detector.h>
#include <d2frontend/utils.h>
#include <spdlog/spdlog.h>

#include <d2common/utils.hpp>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std::chrono;
using namespace D2Common;
using D2Common::Utility::TicToc;

#define WIN_SIZE cv::Size(21, 21)

namespace D2FrontEnd {

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg,
                        int flag) {
  return cv::imdecode(img_msg->data, flag);
}

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg) {
  cv_bridge::CvImagePtr ptr;
  // std::cout << img_msg->encoding << std::endl;
  if (img_msg.encoding == "8UC1" || img_msg.encoding == "mono8") {
    ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
  } else if (img_msg.encoding == "16UC1" || img_msg.encoding == "mono16") {
    ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
    ptr->image.convertTo(ptr->image, CV_8UC1, 1.0 / 256.0);
  } else {
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  }
  return ptr;
}

cv_bridge::CvImagePtr getImageFromMsg(
    const sensor_msgs::ImageConstPtr &img_msg) {
  cv_bridge::CvImagePtr ptr;
  // std::cout << img_msg->encoding << std::endl;
  if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8") {
    ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
  } else if (img_msg->encoding == "16UC1" || img_msg->encoding == "mono16") {
    ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
    ptr->image.convertTo(ptr->image, CV_8UC1, 1.0 / 256.0);
  } else {
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  }
  return ptr;
}

cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p) {
  cv::Vec3b color;
  if (img.channels() == 3) {
    color = img.at<cv::Vec3b>(p);
  } else {
    auto grayscale = img.at<uchar>(p);
    color = cv::Vec3b(grayscale, grayscale, grayscale);
  }
  return color;
}

Eigen::MatrixXf load_csv_mat_eigen(std::string csv) {
  int cols = 0, rows = 0;
  std::vector<double> buff;

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(csv);
  std::string line;

  while (getline(infile, line)) {
    int temp_cols = 0;
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      buff.emplace_back(std::stod(cell));
      temp_cols++;
    }

    rows++;
    if (cols > 0) {
      assert(cols == temp_cols && "Matrix must have same cols on each rows!");
    } else {
      cols = temp_cols;
    }
  }

  infile.close();

  Eigen::MatrixXf result(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) result(i, j) = buff[cols * i + j];

  return result;
}

Eigen::VectorXf load_csv_vec_eigen(std::string csv) {
  int cols = 0, rows = 0;
  double buff[100000];

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(csv);
  while (!infile.eof()) {
    std::string line;
    getline(infile, line);

    int temp_cols = 0;
    std::stringstream stream(line);
    while (!stream.eof()) stream >> buff[cols * rows + temp_cols++];

    if (temp_cols == 0) continue;

    if (cols == 0) cols = temp_cols;

    rows++;
  }

  infile.close();

  rows--;

  // Populate matrix with numbers.
  Eigen::VectorXf result(rows, cols);
  for (int i = 0; i < rows; i++) result(i) = buff[i];

  return result;
}

cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q) {
  Eigen::Vector3d pt3d(pt.x, pt.y, 1);
  pt3d = q * pt3d;

  if (pt3d.z() < 1e-3 && pt3d.z() > 0) {
    pt3d.z() = 1e-3;
  }

  if (pt3d.z() > -1e-3 && pt3d.z() < 0) {
    pt3d.z() = -1e-3;
  }

  return cv::Point2f(pt3d.x() / pt3d.z(), pt3d.y() / pt3d.z());
}

}  // namespace D2FrontEnd