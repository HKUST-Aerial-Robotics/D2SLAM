#include "camodocal/camera_models/CylindricalCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/gpl/gpl.h"

namespace camodocal {

CylindricalCamera::Parameters::Parameters()
    : Camera::Parameters(CYLINRICALCAMERA),
      m_fx(0.0),
      m_fy(0.0),
      m_cx(0.0),
      m_cy(0.0) {}

CylindricalCamera::Parameters::Parameters(const std::string& cameraName, int w,
                                          int h, double fx, double fy,
                                          double cx, double cy)
    : Camera::Parameters(CYLINRICALCAMERA, cameraName, w, h),
      m_fx(fx),
      m_fy(fy),
      m_cx(cx),
      m_cy(cy) {}

double& CylindricalCamera::Parameters::fx(void) { return m_fx; }

double& CylindricalCamera::Parameters::fy(void) { return m_fy; }

double& CylindricalCamera::Parameters::cx(void) { return m_cx; }

double& CylindricalCamera::Parameters::cy(void) { return m_cy; }

double CylindricalCamera::Parameters::fx(void) const { return m_fx; }

double CylindricalCamera::Parameters::fy(void) const { return m_fy; }

double CylindricalCamera::Parameters::cx(void) const { return m_cx; }

double CylindricalCamera::Parameters::cy(void) const { return m_cy; }

bool CylindricalCamera::Parameters::readFromYamlFile(
    const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("PINHOLE") != 0) {
      return false;
    }
  }

  m_modelType = CYLINRICALCAMERA;
  fs["camera_name"] >> m_cameraName;
  m_imageWidth = static_cast<int>(fs["image_width"]);
  m_imageHeight = static_cast<int>(fs["image_height"]);
  auto n = fs["projection_parameters"];
  m_fx = static_cast<double>(n["fx"]);
  m_fy = static_cast<double>(n["fy"]);
  m_cx = static_cast<double>(n["cx"]);
  m_cy = static_cast<double>(n["cy"]);

  return true;
}

void CylindricalCamera::Parameters::writeToYamlFile(
    const std::string& filename) const {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "model_type"
     << "PINHOLE";
  fs << "camera_name" << m_cameraName;
  fs << "image_width" << m_imageWidth;
  fs << "image_height" << m_imageHeight;

  // projection: fx, fy, cx, cy
  fs << "projection_parameters";
  fs << "{"
     << "fx" << m_fx << "fy" << m_fy << "cx" << m_cx << "cy" << m_cy << "}";

  fs.release();
}

CylindricalCamera::Parameters& CylindricalCamera::Parameters::operator=(
    const CylindricalCamera::Parameters& other) {
  if (this != &other) {
    m_modelType = other.m_modelType;
    m_cameraName = other.m_cameraName;
    m_imageWidth = other.m_imageWidth;
    m_imageHeight = other.m_imageHeight;
    m_fx = other.m_fx;
    m_fy = other.m_fy;
    m_cx = other.m_cx;
    m_cy = other.m_cy;
  }

  return *this;
}

std::ostream& operator<<(std::ostream& out,
                         const CylindricalCamera::Parameters& params) {
  out << "Camera Parameters:" << std::endl;
  out << "    model_type "
      << "PINHOLE" << std::endl;
  out << "   camera_name " << params.m_cameraName << std::endl;
  out << "   image_width " << params.m_imageWidth << std::endl;
  out << "  image_height " << params.m_imageHeight << std::endl;

  // projection: fx, fy, cx, cy
  out << "Projection Parameters" << std::endl;
  out << "            fx " << params.m_fx << std::endl
      << "            fy " << params.m_fy << std::endl
      << "            cx " << params.m_cx << std::endl
      << "            cy " << params.m_cy << std::endl;

  return out;
}

CylindricalCamera::CylindricalCamera()
    : m_inv_K11(1.0),
      m_inv_K13(0.0),
      m_inv_K22(1.0),
      m_inv_K23(0.0),
      m_noDistortion(true) {}

CylindricalCamera::CylindricalCamera(const std::string& cameraName,
                                     int imageWidth, int imageHeight, double fx,
                                     double fy, double cx, double cy)
    : mParameters(cameraName, imageWidth, imageHeight, fx, fy, cx, cy) {
  m_noDistortion = true;
  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();

  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
}

CylindricalCamera::CylindricalCamera(
    const CylindricalCamera::Parameters& params)
    : mParameters(params) {
  m_noDistortion = true;
  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();
  K << mParameters.fx(), 0, mParameters.cx(), 0, mParameters.fy(),
      mParameters.fy(), 0, 0, 1;
}

Camera::ModelType CylindricalCamera::modelType(void) const {
  return mParameters.modelType();
}

const std::string& CylindricalCamera::cameraName(void) const {
  return mParameters.cameraName();
}

int CylindricalCamera::imageWidth(void) const {
  return mParameters.imageWidth();
}

int CylindricalCamera::imageHeight(void) const {
  return mParameters.imageHeight();
}

void CylindricalCamera::estimateIntrinsics(
    const cv::Size& boardSize,
    const std::vector<std::vector<cv::Point3f> >& objectPoints,
    const std::vector<std::vector<cv::Point2f> >& imagePoints) {
  // TODO: implement this
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void CylindricalCamera::liftSphere(const Eigen::Vector2d& p,
                                   Eigen::Vector3d& P) const {
  liftProjective(p, P);

  P.normalize();
}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void CylindricalCamera::liftProjective(const Eigen::Vector2d& p,
                                       Eigen::Vector3d& P) const {
  // Lift points to normalised plane
  double phi = m_inv_K11 * p(0) + m_inv_K13;
  double y_by_rho = m_inv_K22 * p(1) + m_inv_K23;
  // Assume z = 1
  double z = fabs(phi) > M_PI / 2 ? -1.0 : 1.0;
  // phi = atan2(X,Z); recover X
  double x = z * tan(phi);
  double rho = sqrt(x * x + z * z);
  //\rho = \sqrt{x^2 + z^2}
  double y = y_by_rho * rho;
  P = Eigen::Vector3d(x, y, z);
}

/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void CylindricalCamera::spaceToPlane(const Eigen::Vector3d& P,
                                     Eigen::Vector2d& p) const {
  auto rho = sqrt(P.x() * P.x() + P.z() * P.z());
  auto phi = atan2(P.x(), P.z());

  Eigen::Vector3d p_c_cyn{rho * phi, P.y(), rho};
  Eigen::Vector3d _p = K * p_c_cyn;
  _p /= _p.z();
  p.x() = _p.x();
  p.y() = _p.y();
}

/**
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void CylindricalCamera::undistToPlane(const Eigen::Vector2d& p_u,
                                      Eigen::Vector2d& p) const {
  Eigen::Vector3d p3d{p_u.x(), p_u.y(), 1.0};
  spaceToPlane(p3d, p);
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void CylindricalCamera::distortion(const Eigen::Vector2d& p_u,
                                   Eigen::Vector2d& d_u) const {
  d_u = p_u;
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *        and calculate Jacobian
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void CylindricalCamera::distortion(const Eigen::Vector2d& p_u,
                                   Eigen::Vector2d& d_u,
                                   Eigen::Matrix2d& J) const {
  d_u = p_u;
}

void CylindricalCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2,
                                         double fScale) const {
  // No distortion
}

cv::Mat CylindricalCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                                   float fx, float fy,
                                                   cv::Size imageSize, float cx,
                                                   float cy,
                                                   cv::Mat rmat) const {
  // No distortion
}

int CylindricalCamera::parameterCount(void) const { return 4; }

const CylindricalCamera::Parameters& CylindricalCamera::getParameters(
    void) const {
  return mParameters;
}

void CylindricalCamera::setParameters(
    const CylindricalCamera::Parameters& parameters) {
  mParameters = parameters;
  m_noDistortion = true;

  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

void CylindricalCamera::readParameters(
    const std::vector<double>& parameterVec) {
  if ((int)parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.fx() = parameterVec.at(0);
  params.fy() = parameterVec.at(1);
  params.cx() = parameterVec.at(2);
  params.cy() = parameterVec.at(3);

  setParameters(params);
}

void CylindricalCamera::writeParameters(
    std::vector<double>& parameterVec) const {
  parameterVec.resize(parameterCount());
  parameterVec.at(0) = mParameters.fx();
  parameterVec.at(1) = mParameters.fy();
  parameterVec.at(2) = mParameters.cx();
  parameterVec.at(3) = mParameters.cy();
}

void CylindricalCamera::writeParametersToYamlFile(
    const std::string& filename) const {
  mParameters.writeToYamlFile(filename);
}

std::string CylindricalCamera::parametersToString(void) const {
  std::ostringstream oss;
  oss << mParameters;

  return oss.str();
}

}  // namespace camodocal
