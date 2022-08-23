#pragma once

#include <Eigen/Eigen>
#include <swarm_msgs/Pose.h>
#include <fstream>
#include <mutex>
#include <chrono>
#include <ceres/ceres.h>

using namespace Eigen;

namespace D2Common {
namespace Utility {
typedef std::lock_guard<std::recursive_mutex> Guard;
inline Quaterniond g2R(const Vector3d &g)
{
    Vector3d ng1 = g.normalized();
    Vector3d ng2{0, 0, 1.0};
    Quaterniond q0 = Quaterniond::FromTwoVectors(ng1, ng2);
    double yaw = quat2eulers(q0).z();
    q0 = eulers2quat(Vector3d{0, 0, -yaw}) * q0;
    return q0;
}

template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> quatfromRotationVector(const Eigen::MatrixBase<Derived> &theta, double eps=1e-2)
{
    typedef typename Derived::Scalar Scalar_t;
    if (theta.norm() < eps) {
        return deltaQ(theta);
    } else {
        Eigen::Quaternion<Scalar_t> dq;
        Scalar_t angle = theta.norm();
        Scalar_t half_angle = angle / static_cast<Scalar_t>(2.0);
        auto xyz = theta / angle * sin(half_angle);
        return Eigen::Quaternion<Scalar_t>(cos(half_angle), xyz.x(), xyz.y(), xyz.z());
    }
    return Eigen::Quaternion<Scalar_t>::Identity();
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
{
    //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
    Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
    //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
    return q.template w() >= (typename Derived::Scalar)(0.0) ? q : p;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymVec3(const Eigen::MatrixBase<Derived> &v)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -v(2), v(1),
        v(2), typename Derived::Scalar(0), -v(0),
        -v(1), v(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
    return ans;
}


template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
    return ans;
}

template <typename Derived>
static Eigen::SparseMatrix<Derived> inverse(const Eigen::SparseMatrix<Derived> & A) {
    Eigen::SparseMatrix<Derived> I(A.rows(), A.cols());
    I.setIdentity();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<Derived>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cout << "Solve A LLT failed!!! A:\n" << A << std::endl;
    }
    assert(solver.info() == Eigen::Success && "LLT failed");
    Eigen::SparseMatrix<Derived> A_inv = solver.solve(I);
    return A_inv;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> recoverRotationSVD(Eigen::Matrix<T, 3, 3> M) {
    // This function compute argmin_R (||R - M||_F) sub to R, R is a rotation matrix
    auto svd = M.jacobiSvd(ComputeFullV|ComputeFullU);
    auto S = svd.matrixU();
    auto Vt = svd.matrixV().transpose();
    auto detSV = (S*Vt).determinant();
    Eigen::Matrix<T, 3, 3> R = S*Eigen::Matrix<T, 3, 1>(1, 1, detSV).asDiagonal()*Vt;
    return R;
}

template <typename Derived>
static std::pair<SparseMatrix<Derived>, Matrix<Derived, Dynamic, 1>> schurComplement(const SparseMatrix<Derived> & H, const Matrix<Derived, Dynamic, 1> & b, int keep_state_dim) {
    //Sparse schur complement
    int remove_state_dim = H.rows() - keep_state_dim;
    auto H11 = H.block(0, 0, keep_state_dim, keep_state_dim);
    auto H12 = H.block(0, keep_state_dim, keep_state_dim, remove_state_dim);
    auto H22 = H.block(keep_state_dim, keep_state_dim, remove_state_dim, remove_state_dim);
    auto H22_inv = inverse(SparseMatrix<Derived>(H22));
    SparseMatrix<Derived> A = H11 - H12 * H22_inv * SparseMatrix<Derived>(H12.transpose());
    Matrix<Derived, Dynamic, 1> bret = b.segment(0, keep_state_dim) - H12 * H22_inv * b.segment(keep_state_dim, remove_state_dim);
    return std::make_pair(A, bret);
}

template <typename Derived>
static std::pair<Matrix<Derived, Dynamic, Dynamic>, Matrix<Derived, Dynamic, 1>> schurComplement(const Matrix<Derived, Dynamic, Dynamic> & H, const Matrix<Derived, Dynamic, 1> & b, int keep_state_dim) {
    const double eps = 1e-8;
    int remove_state_dim = H.rows() - keep_state_dim;
    auto H11 = H.block(0, 0, keep_state_dim, keep_state_dim);
    auto H12 = H.block(0, keep_state_dim, keep_state_dim, remove_state_dim);
    auto H22 = H.block(keep_state_dim, keep_state_dim, remove_state_dim, remove_state_dim);

    SelfAdjointEigenSolver<Matrix<Derived, Dynamic, Dynamic>> saes(H22);
    Matrix<Derived, Dynamic, Dynamic> H22_inv = saes.eigenvectors() * Matrix<Derived, Dynamic, 1>((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    Matrix<Derived, Dynamic, Dynamic> A = H11 - H12 * H22_inv * H12.transpose();
    Matrix<Derived, Dynamic, 1> bret = b.segment(0, keep_state_dim) - H12 * H22_inv * b.segment(keep_state_dim, remove_state_dim);
    return std::make_pair(A, bret);
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 4> jacTan2q(const Eigen::QuaternionBase<Derived> &q) {
    //Convert the quaternion from the tangent space to the real quaternion
    Eigen::Matrix<typename Derived::Scalar, 3, 4> ans;
    ans << q.w(), -q.z(), q.y(), -q.x(),
        q.z(), q.w(), -q.x(), -q.y(),
        -q.y(), q.x(), q.w(), -q.z();
    return ans;
}

template <typename Derived>
static void writeMatrixtoFile(const std::string & path, const MatrixBase<Derived> & matrix) {
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
    std::fstream f;
    f.open(path, std::ios::out);
    f << matrix.format(CSVFormat) << std::endl;
    f.close();
}

template <typename Derived>
void removeRows(Matrix<Derived, Dynamic, Dynamic>& matrix, unsigned int rowToRemove, unsigned int count)
{
    unsigned int numRows = matrix.rows()-count;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

template <typename Derived>
void removeCols(Matrix<Derived, Dynamic, Dynamic>& matrix, unsigned int colToRemove, unsigned int count)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-count;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

template <typename Derived>
void removeRows(Matrix<Derived, Dynamic, 1>& matrix, unsigned int rowToRemove, unsigned int count)
{
    unsigned int numRows = matrix.rows()-count;

    if( rowToRemove < numRows )
        matrix.segment(rowToRemove,numRows-rowToRemove) = matrix.tail(numRows-rowToRemove);

    matrix.conservativeResize(numRows,1);
}

template <typename Derived>
Quaternion<Derived> averageQuaterions(std::vector<Quaternion<Derived>> quats) {
    Matrix<Derived, 4, 4> M = Matrix4d::Zero();
    if (quats.size() == 1) {
        return quats[0];
    }
    for (auto & q : quats) {
       Vector4d v = q.coeffs(); 
       M += v*v.transpose();
    }
    SelfAdjointEigenSolver<Matrix<Derived, 4, 4>> solver;
    solver.compute(M);
    Matrix<Derived, 4, 1> eigenvector = solver.eigenvectors().rightCols(1);
    Quaternion<Derived> q(eigenvector(3), eigenvector(0), eigenvector(1), eigenvector(2));
    return q;
}

template<typename T>
inline void yawRotateVec(T yaw, 
        const Eigen::Matrix<T, 3, 1> & vec, 
        Eigen::Matrix<T, 3, 1> & ret) {
    ret(0) = cos(yaw) * vec(0) - sin(yaw)*vec(1);
    ret(1) = sin(yaw) * vec(0) + cos(yaw)*vec(1);
    ret(2) = vec(2);
}

template<typename T>
inline Matrix<T, 3, 3> yawRotMat(T yaw) {
    Matrix<T, 3, 3> R;
    T cyaw = ceres::cos(yaw);
    T syaw = ceres::sin(yaw);
    R << cyaw, - syaw, ((T) 0), 
        syaw, cyaw, ((T) 0),
        ((T) 0), ((T) 0), ((T) 1);
    return R;
}

template <typename T>
inline T NormalizeAngle(const T& angle_radians) {
  // Use ceres::floor because it is specialized for double and Jet types.
  T two_pi(2.0 * M_PI);
  return angle_radians -
         two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}

template<typename T>
inline void deltaPose4D(Eigen::Matrix<T, 4, 1> posea, 
        Eigen::Matrix<T, 4, 1> poseb, Eigen::Matrix<T, 4, 1> & dpose) {
    dpose(3) = Utility::NormalizeAngle(poseb(3) - posea(3));
    Eigen::Matrix<T, 3, 1> tmp = poseb - posea;
    yawRotateVec(-posea(3), tmp, dpose.segment<3>(0));
}

template<typename T>
inline void poseError4D(const Ref<const Matrix<T, 3, 1>> & posa, T yaw_a,
        const Ref<const Matrix<T, 3, 1>> & posb, T yaw_b,
        const Ref<const Matrix<T, 4, 4>> &_sqrt_inf_mat, 
        T *error, bool norm_yaw=true) {
    Map<Matrix<T, 3, 1>> err(error);
    err = posb - posa;
    if (norm_yaw) {
        error[3] = Utility::NormalizeAngle(yaw_b - yaw_a);
    } else {
        error[3] = yaw_b - yaw_a;
    }
    Map<Matrix<T, 4, 1>> err_4d(error);
    err_4d.applyOnTheLeft(_sqrt_inf_mat);
}

class TicToc {
public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::system_clock::now();
    }

    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

}
}