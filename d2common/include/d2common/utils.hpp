#pragma once

#include <Eigen/Eigen>
#include <swarm_msgs/Pose.h>
#include <fstream>

using namespace Eigen;

namespace D2Common {
namespace Utility {
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
    assert(solver.info() == Eigen::Success && "LLT failed");
    Eigen::SparseMatrix<Derived> A_inv = solver.solve(I);
    return A_inv;
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
    for (auto & q : quats) {
       Vector4d v = q.coeffs(); 
       M += v*v.transpose();
    }
    SelfAdjointEigenSolver<Matrix<Derived, 4, 4>> solver;
    solver.compute(M);
    Matrix<Derived, 4, 1> eigenvector = solver.eigenvectors().rightCols(1);
    Quaternion<Derived> q(eigenvector(3), eigenvector(1), eigenvector(2), eigenvector(0));
    // Matrix<Derived, 1, Dynamic> eigenvalues = solver.eigenvalues();
    // for (int i = 0; i < eigenvalues.size(); i ++ ) {
    //     printf("%f, %f %f %f %f\n", solver.eigenvalues()(i), 
    //         solver.eigenvectors()(0,i), solver.eigenvectors()(1,i), solver.eigenvectors()(2,i), solver.eigenvectors()(3,i));
    // }
    // printf("return q w %f xyz %f %f %f\n", q.w(), q.x(), q.y(), q.z());
    return q;
}

}
}