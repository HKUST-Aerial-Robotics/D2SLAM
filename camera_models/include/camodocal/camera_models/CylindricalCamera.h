#ifndef CYLINRICALCAMERA_H
#define CYLINRICALCAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "ceres/rotation.h"
#include "Camera.h"

namespace camodocal
{

class CylindricalCamera: public Camera
{
public:
    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();
        Parameters(const std::string& cameraName,
                   int w, int h,
                   double fx, double fy, double cx, double cy);

        double& fx(void);
        double& fy(void);
        double& cx(void);
        double& cy(void);

        double fx(void) const;
        double fy(void) const;
        double cx(void) const;
        double cy(void) const;

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<< (std::ostream& out, const Parameters& params);

    private:
        double m_fx;
        double m_fy;
        double m_cx;
        double m_cy;
    };

    CylindricalCamera();

    /**
    * \brief Constructor from the projection model parameters
    */
    CylindricalCamera(const std::string& cameraName,
                  int imageWidth, int imageHeight,
                  double fx, double fy, double cx, double cy);
    /**
    * \brief Constructor from the projection model parameters
    */
    CylindricalCamera(const Parameters& params);

    Camera::ModelType modelType(void) const;
    const std::string& cameraName(void) const;
    int imageWidth(void) const;
    int imageHeight(void) const;

    void estimateIntrinsics(const cv::Size& boardSize,
                            const std::vector< std::vector<cv::Point3f> >& objectPoints,
                            const std::vector< std::vector<cv::Point2f> >& imagePoints);

    // Lift points from the image plane to the sphere
    virtual void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Lift points from the image plane to the projective space
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Projects 3D points to the image plane (Pi function)
    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;
    //%output p

    // Projects 3D points to the image plane (Pi function)
    // and calculates jacobian
    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                      Eigen::Matrix<double,2,3>& J) const;
    //%output p
    //%output J

    void undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const;
    //%output p

    template <typename T>
    static void spaceToPlane(const T* const params,
                             const T* const q, const T* const t,
                             const Eigen::Matrix<T, 3, 1>& P,
                             Eigen::Matrix<T, 2, 1>& p);

    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,
                    Eigen::Matrix2d& J) const;

    void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0) const;
    cv::Mat initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx = -1.0f, float fy = -1.0f,
                                    cv::Size imageSize = cv::Size(0, 0),
                                    float cx = -1.0f, float cy = -1.0f,
                                    cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    int parameterCount(void) const;

    const Parameters& getParameters(void) const;
    void setParameters(const Parameters& parameters);

    void readParameters(const std::vector<double>& parameterVec);
    void writeParameters(std::vector<double>& parameterVec) const;

    void writeParametersToYamlFile(const std::string& filename) const;

    std::string parametersToString(void) const;

private:
    Parameters mParameters;
    Eigen::Matrix3d K;
    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    bool m_noDistortion;
};

typedef boost::shared_ptr<CylindricalCamera> CylindricalCameraPtr;
typedef boost::shared_ptr<const CylindricalCamera> CylindricalCameraConstPtr;

template <typename T>
void
CylindricalCamera::spaceToPlane(const T* const params,
                            const T* const q, const T* const t,
                            const Eigen::Matrix<T, 3, 1>& P,
                            Eigen::Matrix<T, 2, 1>& p)
{
    // T P_w[3];
    // P_w[0] = T(P(0));
    // P_w[1] = T(P(1));
    // P_w[2] = T(P(2));

    // // Convert quaternion from Eigen convention (x, y, z, w)
    // // to Ceres convention (w, x, y, z)
    // T q_ceres[4] = {q[3], q[0], q[1], q[2]};

    // T P_c[3];
    // ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    // P_c[0] += t[0];
    // P_c[1] += t[1];
    // P_c[2] += t[2];

    // // project 3D object point to the image plane
    // T fx = params[4];
    // T fy = params[5];
    // T cx = params[6];
    // T cy = params[7];

    // // Transform to model plane
    // Eigen::Map<Eigen::Matrix<T, 3, 1>> P_c_eigen(P_c);
    // auto rho = ceres::sqrt(P_c_eigen.x()*P_c_eigen.x() + P_c_eigen.z()*P_c_eigen.z());
    // auto phi = atan2(P_c_eigen.x(), P_c_eigen.z());
    // Eigen::Matrix<T, 3, 3> K;
    // K << T(fx), T(0), T(cx),
    //      T(0), T(fy), T(cy),
    //      T(0), T(0), T(1);
    // Eigen::Matrix<T, 3, 1> p_c_cyn{rho*phi, P_c_eigen.y(), rho};
    // Eigen::Matrix<T, 3, 1> _p = K*p_c_cyn;
    // _p/=_p.z();
    // p[0] = _p.x();
    // p[1] = _p.y();
}

}

#endif
