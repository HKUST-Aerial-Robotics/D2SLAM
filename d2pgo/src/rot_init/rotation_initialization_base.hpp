#pragma once
#include <d2common/utils.hpp>
#include "../pgostate.hpp"
#include <swarm_msgs/relative_measurments.hpp>
#include "../d2pgo_config.h"

namespace D2PGO {

template<typename T>
class RotationInitialization {
protected:
    typedef Eigen::Matrix<T, 3, 3> Mat3;
    typedef Eigen::Matrix<T, 3, 1> Vec3;
    typedef Eigen::Triplet<T> Tpl;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecX;

    PGOState * state = nullptr;
    std::vector<Swarm::LoopEdge> loops;
    std::vector<Swarm::PosePrior> pose_priors; 
    std::map<FrameIdType, int> frame_id_to_idx;
    std::set<FrameIdType> fixed_frame_ids;
    std::set<FrameIdType> all_frames;
    RotInitConfig config;
    bool is_fixed_frame_set = false;
    int self_id;
    int eff_frame_num = 0;

    virtual void addFrameId(FrameIdType _frame_id) {
        all_frames.insert(_frame_id);
    }

    int getFrameIdx(FrameIdType _frame_id) {
        auto idx =  frame_id_to_idx.at(_frame_id);
        return idx;
    }

    bool isFixedFrame(FrameIdType _frame_id) {
        return fixed_frame_ids.find(_frame_id) != fixed_frame_ids.end();
    }

    void updateFrameIdx() {
        frame_id_to_idx.clear();
        int idx = 0;
        for (auto frame_id : all_frames) {
            if (!isFixedFrame(frame_id)) {
                frame_id_to_idx[frame_id] = idx;
                idx++;
            } else {
                frame_id_to_idx[frame_id] = -1;
            }
        }
        eff_frame_num = idx;
    }

    void fillInTripet(int i0, int j0, Matrix<T, 3, 3> M, std::vector<Tpl> & triplets) {
        for (int i = 0; i < 3; i ++) { //Row, col of relative rotation R
            for (int j = 0; j < 3; j ++) {
                triplets.emplace_back(Tpl(i0 + i, j0 + j, M(i, j)));
            }
        }
    }

    void setPriorFactorsbyFixedParam() {
        for (auto loop : loops) {
            //Set up the problem
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            if (isFixedFrame(frame_id_a) && ! isFixedFrame(frame_id_b)) {
                auto pose_fixed = state->getFramebyId(frame_id_a)->odom.pose();
                Swarm::Pose pose_b = pose_fixed*loop.relative_pose;
                pose_priors.emplace_back(Swarm::PosePrior(frame_id_b, pose_b, loop.getInfoMat()));
            } else if (!isFixedFrame(frame_id_a) && isFixedFrame(frame_id_b)) {
                auto pose_fixed = state->getFramebyId(frame_id_b)->odom.pose();
                Swarm::Pose pose_a = pose_fixed*loop.relative_pose.inverse();
                pose_priors.emplace_back(Swarm::PosePrior(frame_id_b, pose_a, loop.getInfoMat()));
            }
        }
    }

    void solveLinear() {
        D2Common::Utility::TicToc tic;
        VecX b(loops.size()*ROTMAT_SIZE + pose_priors.size()*ROTMAT_SIZE);
        if (config.enable_gravity_prior) {
            b.resize(loops.size()*ROTMAT_SIZE + pose_priors.size()*ROTMAT_SIZE + eff_frame_num*POS_SIZE);
        }
        b.setZero();
        int row_id = 0;
        std::vector<Tpl> triplet_list;
        for (auto loop : loops) {
            //Set up the problem
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            auto idx_a = getFrameIdx(frame_id_a);
            auto idx_b = getFrameIdx(frame_id_b);
            Mat3 sqrt_info = loop.getSqrtInfoMat().block<3, 3>(3, 3).template cast<T>();
            if (idx_a == idx_b) {
                printf("[RotationInitialization::solveLinear] Loop between frame %ld<->%ld idx %d<->%d is self loop\n", frame_id_a, frame_id_b, idx_a, idx_b);
                continue;
            }
            if (idx_a == -1 || idx_b == -1) {
                continue;
            }
            Mat3 R = loop.relative_pose.att().toRotationMatrix().template cast<T>();
            Mat3 Rt = R.transpose();
            for (int k = 0; k < 3; k ++) { //Row of Rotation of keyframe a
                fillInTripet(row_id + k*3, 9*idx_a + 3*k, sqrt_info*Rt, triplet_list);
                fillInTripet(row_id + k*3, 9*idx_b + 3*k, -sqrt_info, triplet_list);
            }
            row_id += 9;
        }

        for (auto prior: pose_priors) {
            auto frame_id = prior.frame_id;
            auto idx = getFrameIdx(frame_id);
            if (idx == -1) {
                continue;
            }
            Mat3 R = prior.getRotMat().template cast<T>();
            Mat3 Rt = R.transpose();
            Mat3 sqrt_R = prior.getSqrtInfoMatRot().template cast<T>();;
            // std::cout << "sqrtinfo:\n" << sqrt_R << std::endl;
            for (int k = 0; k < 3; k ++) { //Row of Rotation of keyframe a
                fillInTripet(row_id + k*3, 9*idx + 3*k, sqrt_R, triplet_list);
                b.segment(row_id + k*3, 3) = sqrt_R*Rt.col(k);     
            }
            row_id += 9;
        }
        
        if (config.enable_gravity_prior) {
            Mat3 sqrt_info = Mat3::Identity()*config.gravity_sqrt_info;
            //For each frame, add the gravity prior
            for (auto it : frame_id_to_idx) {
                auto frame_id = it.first;
                auto idx = getFrameIdx(frame_id);
                if (idx == -1) {
                    continue;
                }
                auto att_odom = state->getFramebyId(frame_id)->initial_ego_pose.att();
                Vec3 gravity_body = (att_odom.inverse()*config.gravity_direction).template cast<T>();
                //I3*r^3 = gravity_body
                const int k = 2;
                fillInTripet(row_id, 9*idx + 3*k, sqrt_info, triplet_list);
                b.segment(row_id, 3) = sqrt_info*gravity_body;           
                row_id += 3;
            }
        }

        SparseMatrix<T> A(row_id, 9*eff_frame_num);
        A.setFromTriplets(triplet_list.begin(), triplet_list.end());
        printf("Rots %ld A rows%ld cols %ld b rows %d/%ld\n", frame_id_to_idx.size(), A.rows(), A.cols(), row_id, b.rows());
        printf("[RotInit%d] Poses %ld EffPoses %d Loops %ld Priors %ld F32: %d g_prior: %d\n", self_id,
            frame_id_to_idx.size(), eff_frame_num, loops.size(), pose_priors.size(),
            typeid(T) == typeid(float), config.enable_gravity_prior);
        auto At = A.transpose();
        SparseMatrix<T> H = At*A;
        if (b.rows() > row_id) {
            b.conservativeResize(row_id);
        }
        // Solve by LLT
        // std::cout << "A" << std::endl << A << std::endl;
        // std::cout << "b" << std::endl << b.transpose() << std::endl;
        // std::cout << "H" << std::endl << H << std::endl;
        b = At*b;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            std::ofstream ofs("/tmp/A.txt");
            for (auto triplet : triplet_list) {
                ofs << triplet.row() << " " << triplet.col() << " " << triplet.value() << std::endl;
            }
            ofs.close();
            std::ofstream ofs1("/tmp/b.txt");
            ofs1 << b << std::endl;
            ofs1.close();
        }
        assert(solver.info() == Eigen::Success && "LLT failed");
        auto X = solver.solve(b);
        double dt = tic.toc();
        D2Common::Utility::TicToc tic2;
        // std::cout << "X" << std::endl << X.transpose() << std::endl;
        //Update the rotation
        recoverRotationLLT(X);
        printf("[RotInit%d] RotInit %.2fms LLT %.2fms Recover %.2fms Poses %ld EffPoses %d Loops %ld Priors %ld F32: %d g_prior: %d\n", self_id,
            tic.toc(), dt, tic2.toc(), frame_id_to_idx.size(), eff_frame_num, loops.size(), pose_priors.size(),
            typeid(T) == typeid(float), config.enable_gravity_prior);
    }

    void recoverRotationLLT(const VecX & X) {
        for (auto it : frame_id_to_idx) {
            auto frame_id = it.first;
            auto idx = getFrameIdx(frame_id);
            if (idx == -1) {
                continue;
            }
            auto frame = state->getFramebyId(frame_id);
            auto & pose = frame->odom.pose();
            Map<const Matrix<T, 3, 3, RowMajor>> M(X.data() + idx*9);
            //Note in X the rotation is stored in row major order.
            auto R = recoverRotationSVD(M);
            // std::cout << "Raw\n" << pose.att().toRotationMatrix() << std::endl;
            // std::cout << "M\n" << M << std::endl;
            // std::cout << "R\n" << R << std::endl;
            pose.att() = Quaternion<T>(R).template cast<double>();
            pose.to_vector(state->getPoseState(frame_id));
            Map<Matrix<double, 3, 3, RowMajor>> R_state(state->getRotState(frame_id));
            R_state = M.template cast<double>(); //Not essential to be rotation matrix. For ARock.
        }
    }

    static Eigen::Matrix<T, 3, 3> recoverRotationSVD(Eigen::Matrix<T, 3, 3> M) {
        auto svd = M.jacobiSvd(ComputeFullV|ComputeFullU);
        auto S = svd.matrixU();
        auto Vt = svd.matrixV().transpose();
        auto detSV = (S*Vt).determinant();
        Eigen::Matrix<T, 3, 3> R = S*Vec3(1, 1, detSV).asDiagonal()*Vt;
        return R;
    }

public:
    RotationInitialization(PGOState * _state, RotInitConfig _config):
        state(_state), config(_config), self_id(_config.self_id) {
    }

    virtual void addLoop(const Swarm::LoopEdge & loop) {
        loops.emplace_back(loop);
    }
    
    virtual void addLoops(const std::vector<Swarm::LoopEdge> & _loops) {
        printf("[RotationInitialization::addLoops] Adding %ld loops\n", _loops.size());
        loops = _loops;
    }

    void setFixedFrameId(FrameIdType _fixed_frame_id) {
        fixed_frame_ids.insert(_fixed_frame_id);
        printf("[RotationInitialization::setFixedFrameId] Fixed frame id: %ld now %ld fixed\n", _fixed_frame_id, fixed_frame_ids.size());
    }

    void solve() {
        //Chordal relaxation algorithm.
        for (auto loop : loops) {
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            addFrameId(frame_id_a);
            addFrameId(frame_id_b);
        }
        updateFrameIdx();
        setPriorFactorsbyFixedParam();
        solveLinear();
    }

    void reset() {
        loops.clear();
        pose_priors.clear(); 
        // frame_id_to_idx.clear();
        // is_fixed_frame_set = false;
    }
};

typedef RotationInitialization<double> RotationInitializationd;
typedef RotationInitialization<float> RotationInitializationf;
}