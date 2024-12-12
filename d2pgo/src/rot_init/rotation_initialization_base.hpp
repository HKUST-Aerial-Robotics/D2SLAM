#pragma once
#include <d2common/utils.hpp>
#include "../pgostate.hpp"
#include <swarm_msgs/relative_measurments.hpp>
#include "../d2pgo_config.h"

namespace D2PGO {
using D2Common::Utility::skewSymVec3;
using D2Common::Utility::recoverRotationSVD;
using D2Common::Utility::TicToc;

template <typename Derived, typename T>
inline void fillInTripet(int i0, int j0, const MatrixBase<Derived> & M, std::vector<Eigen::Triplet<T>> & triplets) {
    for (unsigned int i = 0; i < M.rows(); i ++) { //Row, col of relative rotation R
        for (int j = 0; j < M.cols(); j ++) {
            triplets.emplace_back(Eigen::Triplet<T>(i0 + i, j0 + j, M(i, j)));
        }
    }
}

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
    bool is_multi = false;

    virtual void addFrameId(FrameIdType _frame_id) {
        all_frames.insert(_frame_id);
    }

    int getFrameIdx(FrameIdType _frame_id) {
        if (frame_id_to_idx.find(_frame_id) == frame_id_to_idx.end()) {
            printf("[getFrameIdx@%d] Frame id %ld not found in frame_id_to_idx\n", self_id, _frame_id);
        }
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

    int setupRotInitProblembyLoop(int row_id, const Swarm::LoopEdge & loop, std::vector<Tpl> & triplet_list, VecX & b) {
        //Set up the problem
        auto frame_id_a = loop.keyframe_id_a;
        auto frame_id_b = loop.keyframe_id_b;
        auto idx_a = getFrameIdx(frame_id_a);
        auto idx_b = getFrameIdx(frame_id_b);
        Mat3 sqrt_info = loop.getSqrtInfoMat().block<3, 3>(3, 3).template cast<T>();
        if (idx_a == idx_b) {
            printf("[RotationInitialization::solveLinear] Loop between frame %ld<->%ld idx %d<->%d is self loop\n", frame_id_a, frame_id_b, idx_a, idx_b);
            return row_id;
        }
        if (idx_a == -1 || idx_b == -1) {
            return row_id;
        }
        sqrt_info = sqrt_info.norm() * Mat3::Identity();
        Mat3 R = loop.relative_pose.R().template cast<T>();
        Mat3 Rt = R.transpose();
        for (int k = 0; k < 3; k ++) { //Row of Rotation of keyframe a
            fillInTripet(row_id + k*POS_SIZE, ROTMAT_SIZE*idx_a + POS_SIZE*k, sqrt_info*Rt, triplet_list);
            fillInTripet(row_id + k*POS_SIZE, ROTMAT_SIZE*idx_b + POS_SIZE*k, -sqrt_info, triplet_list);
        }
        return row_id + 9;
    }

    int setupRotInitProblembyPrior(int row_id, const Swarm::PosePrior & prior, std::vector<Tpl> & triplet_list, VecX & b) {
        auto frame_id = prior.frame_id;
        auto idx = getFrameIdx(frame_id);
        if (idx == -1) {
            return row_id;
        }
        Mat3 R = prior.getRotMat().template cast<T>();
        Mat3 Rt = R.transpose();
        Mat3 sqrt_R = prior.getSqrtInfoMatRot().template cast<T>();
        sqrt_R = Mat3::Identity() * sqrt_R.norm();
        // std::cout << "sqrtinfo:\n" << sqrt_R << std::endl;
        for (int k = 0; k < 3; k ++) { //Row of Rotation of keyframe a
            fillInTripet(row_id + k*POS_SIZE, ROTMAT_SIZE*idx + POS_SIZE*k, sqrt_R, triplet_list);
            b.segment(row_id + k*POS_SIZE, POS_SIZE) = sqrt_R*Rt.col(k);     
        }
        return row_id + 9;
    }

    int setupRotInitProblembyGravityPrior(int row_id, FrameIdType frame_id, std::vector<Tpl> & triplet_list, VecX & b) {
        //I3*r^3 = gravity_body
        auto idx = getFrameIdx(frame_id);
        if (idx == -1) {
            return row_id;
        }
        auto att_odom = state->getFramebyId(frame_id)->initial_ego_pose.att();
        Vec3 gravity_body = (att_odom.inverse()*config.gravity_direction).template cast<T>();
        const int k = 2;
        Mat3 sqrt_info = Mat3::Identity()*config.gravity_sqrt_info;
        fillInTripet(row_id, ROTMAT_SIZE*idx + POS_SIZE*k, sqrt_info, triplet_list);
        b.segment(row_id, 3) = sqrt_info*gravity_body;           
        return row_id + 3;
    }

    VecX solveLinear(int row_id, int cols, const std::vector<Tpl> & triplet_list, VecX & b) {
        SparseMatrix<T> A(row_id, cols);
        A.setFromTriplets(triplet_list.begin(), triplet_list.end());
        auto At = A.transpose();
        SparseMatrix<T> H = At*A;
        if (b.rows() > row_id) {
            b.conservativeResize(row_id);
        }
        b = At*b;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            std::cout << solver.info() << std::endl;
            std::ofstream ofsh("/tmp/H" + std::to_string(self_id) + ".txt");
            for(int i = 0; i < H.outerSize(); i++)
                for(typename Eigen::SparseMatrix<T>::InnerIterator it(H,i); it; ++it)
                    ofsh << it.row() << " " << it.col() << " " << it.value() << std::endl;

            std::ofstream ofs("/tmp/A" + std::to_string(self_id) + ".txt");
            for (auto triplet : triplet_list) {
                ofs << triplet.row() << " " << triplet.col() << " " << triplet.value() << std::endl;
            }
            ofs.close();
            std::ofstream ofs1("/tmp/b" + std::to_string(self_id) + ".txt");
            ofs1 << b << std::endl;
            ofs1.close();
        }
        assert(solver.info() == Eigen::Success && "LLT failed");
        auto X = solver.solve(b);
        return X;
    }

    double solveLinearRot() {
        TicToc tic;
        VecX b(loops.size()*ROTMAT_SIZE + pose_priors.size()*ROTMAT_SIZE);
        if (config.enable_gravity_prior) {
            b.resize(loops.size()*ROTMAT_SIZE + pose_priors.size()*ROTMAT_SIZE + eff_frame_num*POS_SIZE);
        }
        b.setZero();
        int row_id = 0;
        std::vector<Tpl> triplet_list;
        for (auto loop : loops) {
            row_id = setupRotInitProblembyLoop(row_id, loop, triplet_list, b);
        }

        for (auto prior: pose_priors) {
            row_id = setupRotInitProblembyPrior(row_id, prior, triplet_list, b);
        }
        
        if (config.enable_gravity_prior) {
            //For each frame, add the gravity prior
            for (auto it : frame_id_to_idx) {
                auto frame_id = it.first;
                row_id = setupRotInitProblembyGravityPrior(row_id, frame_id, triplet_list, b);
            }
        }
        double dt_setup = tic.toc();
        TicToc tic_solve;
        auto X = solveLinear(row_id, 9*eff_frame_num, triplet_list, b);
        double dt_solve = tic_solve.toc();
        TicToc tic2;
        auto state_changes = recoverRotationLLT(X);
        printf("[RotInit%d] RotInit %.2fms setup %.2fms LLT %.2fms Recover %.2fms state_changes %.1f%% Poses %ld EffPoses %d Loops %ld Priors %ld F32: %d g_prior: %d\n", 
            self_id, tic.toc(), dt_setup, dt_solve, tic2.toc(), state_changes*100,
            frame_id_to_idx.size(), eff_frame_num, loops.size(), pose_priors.size(),
            typeid(T) == typeid(float), config.enable_gravity_prior);
        return state_changes;
    }

    int setupPose6dProblembyLoop(int row_id, const Swarm::LoopEdge & loop, std::vector<Tpl> & triplet_list, VecX & b, bool finetune_rot = true) {
        //Set up the problem
        int pose_size = POSE_EFF_SIZE;
        if (!finetune_rot) {
            pose_size = POS_SIZE;
        }
        auto frame_id_a = loop.keyframe_id_a;
        auto frame_id_b = loop.keyframe_id_b;
        auto idx_a = getFrameIdx(frame_id_a);
        auto idx_b = getFrameIdx(frame_id_b);
        if (idx_a == -1 || idx_b == -1) {
            return row_id;
        }
        Mat3 R_sqrt_info = loop.getSqrtInfoMat().block<3, 3>(3, 3).template cast<T>();
        Mat3 T_sqrt_info = loop.getSqrtInfoMat().block<3, 3>(0, 0).template cast<T>();
        Mat3 Rab = loop.relative_pose.R().template cast<T>();
        Vec3 t = loop.relative_pose.pos().template cast<T>();
        Mat3 Ra = state->getFramebyId(frame_id_a)->R().template cast<T>();
        Mat3 Rb = state->getFramebyId(frame_id_b)->R().template cast<T>();
        R_sqrt_info = Mat3::Identity() * R_sqrt_info.norm()/1.4142;
        T_sqrt_info = Mat3::Identity() * T_sqrt_info.norm();
        //Translation error.
        //For now a pose has 6 param. XYZ and theta_x, theta_y, theta_z
        //Row of Rotation of keyframe a
        fillInTripet(row_id, pose_size*idx_b, T_sqrt_info, triplet_list); //  take +T_sqrt_info*T_b
        fillInTripet(row_id, pose_size*idx_a, -T_sqrt_info, triplet_list); // take -T_sqrt_info*T_a
        b.segment(row_id, 3) = T_sqrt_info*Ra*t;  // T_sqrt_info*R*T_a->b    
        if (!finetune_rot) {
            return row_id + 3;
        }
        Mat3 Cm = Ra*(skewSymVec3(t).transpose()); //R * S(v_a) t_{a->b} => Cm * v
        fillInTripet(row_id, pose_size*idx_a + POS_SIZE, -T_sqrt_info*Cm, triplet_list); //  take Cm*T_sqrt_info*v_a
        row_id = row_id + 3;
        //Finish translation error.
        //Rotation error.
        Matrix<T, 9, 3> Cm_rb;
        Cm_rb << 0,         -Rb(0, 2),  Rb(0, 1),
                Rb(0, 2),   0,          -Rb(0, 0),
                -Rb(0, 1),  Rb(0, 0),   0,
                0,         -Rb(1, 2),  Rb(1, 1),
                Rb(1, 2),   0,          -Rb(1, 0),
                -Rb(1, 1),  Rb(1, 0),   0,
                0,         -Rb(2, 2),  Rb(2, 1),
                Rb(2, 2),   0,          -Rb(2, 0),
                -Rb(2, 1),  Rb(2, 0),   0;

        fillInTripet(row_id, pose_size*idx_b + POS_SIZE, Cm_rb*R_sqrt_info, triplet_list); //  take Cm_rb*v_b
        Matrix<T, 9, 3> Cm_ra;
        Cm_ra <<    Ra(0, 2)*Rab(1, 0) - Ra(0, 1)*Rab(2, 0), -Ra(0, 2)*Rab(0, 0) + Ra(0, 0)*Rab(2, 0), Ra(0, 1)*Rab(0, 0) - Ra(0, 0) * Rab(1, 0),
                    Ra(0, 2)*Rab(1, 1) - Ra(0, 1)*Rab(2, 1), -Ra(0, 2)*Rab(0, 1) + Ra(0, 0)*Rab(2, 1), Ra(0, 1)*Rab(0, 1) - Ra(0, 0) * Rab(1, 1),
                    Ra(0, 2)*Rab(1, 2) - Ra(0, 1)*Rab(2, 2), -Ra(0, 2)*Rab(0, 2) + Ra(0, 0)*Rab(2, 2), Ra(0, 1)*Rab(0, 2) - Ra(0, 0) * Rab(1, 2),
                    Ra(1, 2)*Rab(1, 0) - Ra(1, 1)*Rab(2, 0), -Ra(1, 2)*Rab(0, 0) + Ra(1, 0)*Rab(2, 0), Ra(1, 1)*Rab(0, 0) - Ra(1, 0) * Rab(1, 0),
                    Ra(1, 2)*Rab(1, 1) - Ra(1, 1)*Rab(2, 1), -Ra(1, 2)*Rab(0, 1) + Ra(1, 0)*Rab(2, 1), Ra(1, 1)*Rab(0, 1) - Ra(1, 0) * Rab(1, 1),
                    Ra(1, 2)*Rab(1, 2) - Ra(1, 1)*Rab(2, 2), -Ra(1, 2)*Rab(0, 2) + Ra(1, 0)*Rab(2, 2), Ra(1, 1)*Rab(0, 2) - Ra(1, 0) * Rab(1, 2),
                    Ra(2, 2)*Rab(1, 0) - Ra(2, 1)*Rab(2, 0), -Ra(2, 2)*Rab(0, 0) + Ra(2, 0)*Rab(2, 0), Ra(2, 1)*Rab(0, 0) - Ra(2, 0) * Rab(1, 0),
                    Ra(2, 2)*Rab(1, 1) - Ra(2, 1)*Rab(2, 1), -Ra(2, 2)*Rab(0, 1) + Ra(2, 0)*Rab(2, 1), Ra(2, 1)*Rab(0, 1) - Ra(2, 0) * Rab(1, 1),
                    Ra(2, 2)*Rab(1, 2) - Ra(2, 1)*Rab(2, 2), -Ra(2, 2)*Rab(0, 2) + Ra(2, 0)*Rab(2, 2), Ra(2, 1)*Rab(0, 2) - Ra(2, 0) * Rab(1, 2);
        fillInTripet(row_id, pose_size*idx_a + POS_SIZE, -Cm_ra*R_sqrt_info, triplet_list); //  take -Cm_ra*v_a
        Matrix<T, 3, 3, RowMajor> right = R_sqrt_info*(-Rb+Ra*Rab); // R_sqrt_info*(R_b - R_a*Rab)
        Map<Matrix<T, 9, 1>> right_vec(right.data()); // Flat matrix
        b.segment(row_id, 9) = right_vec;
        row_id = row_id + 9;
        return row_id;
    }

    int setupPose6DProblembyPrior(int row_id, const Swarm::PosePrior & prior, std::vector<Tpl> & triplet_list, VecX & b, bool finetune_rot=true) {
        auto frame_id = prior.frame_id;
        auto idx = getFrameIdx(frame_id);
        int pose_size = POSE_EFF_SIZE;
        if (!finetune_rot) {
            pose_size = POS_SIZE;
        }
        if (idx == -1) {
            return row_id;
        }
        Vec3 Tp = prior.T().template cast<T>();
        // printf("[setupPose6DProblembyPrior%d]Prior for %d T_prior: %.4f %.4f %.4f\n", self_id, frame_id, Tp(0), Tp(1), Tp(2));
        //Translation error.
        Mat3 sqrt_T = prior.getSqrtInfoMat().block<3, 3>(0, 0).template cast<T>();
        fillInTripet(row_id, pose_size*idx, sqrt_T, triplet_list); // take T_sqrt_info*T_a
        b.segment(row_id, 3) = sqrt_T*Tp;  // T_sqrt_info*T_p
        row_id = row_id + 3;
        if (!finetune_rot) {
            return row_id;
        }

        if (prior.is_prior_delta) {
            ROS_ERROR("Not support prior delta");
        } else {
            Mat3 Rp = prior.getRotMat().template cast<T>();
            Mat3 Ra = state->getFramebyId(frame_id)->R().template cast<T>();
            Vec3 Ta = state->getFramebyId(frame_id)->T().template cast<T>();
            Mat3 sqrt_R = prior.getSqrtInfoMatRot().template cast<T>();
            sqrt_R = Mat3::Identity() * sqrt_R.norm()/1.4142;
            sqrt_T = Mat3::Identity() * sqrt_T.norm();

            //Rotation error
            Matrix<T, 9, 3> Cm_ra;
            Cm_ra << 0,         -Ra(0, 2),  Ra(0, 1),
                    Ra(0, 2),   0,          -Ra(0, 0),
                    -Ra(0, 1),  Ra(0, 0),   0,
                    0,         -Ra(1, 2),  Ra(1, 1),
                    Ra(1, 2),   0,          -Ra(1, 0),
                    -Ra(1, 1),  Ra(1, 0),   0,
                    0,         -Ra(2, 2),  Ra(2, 1),
                    Ra(2, 2),   0,          -Ra(2, 0),
                    -Ra(2, 1),  Ra(2, 0),   0;
            fillInTripet(row_id, pose_size*idx + POS_SIZE, Cm_ra*sqrt_R, triplet_list); //  take Cm_ra*sqrt_R*R_a
            Matrix<T, 3, 3, RowMajor> Rp_with_info = sqrt_R*Rp;
            Map<Matrix<T, 9, 1>> right_vec(Rp_with_info.data()); // Flat matrix
            b.segment(row_id, 9) = right_vec; // Rp
            return row_id + 9;
        }
        return row_id;
    }

    double solveLinearPose6d(bool finetune_rot = false) {
        TicToc tic;
        int pose_size = POSE_EFF_SIZE;
        if (!finetune_rot) {
            pose_size = POS_SIZE;
        }
        VecX b(loops.size()*pose_size + pose_priors.size()*pose_size);
        std::vector<Eigen::Triplet<T>> triplet_list;
        b.setZero();
        int row_id = 0;
        for (auto loop : loops) {
            row_id = setupPose6dProblembyLoop(row_id, loop, triplet_list, b, finetune_rot);
        }
        for (auto prior: pose_priors) {
            row_id = setupPose6DProblembyPrior(row_id, prior, triplet_list, b, finetune_rot);
        }
        double dt_setup = tic.toc();
        tic.tic();
        TicToc tic_solve;
        auto X = solveLinear(row_id, pose_size*eff_frame_num, triplet_list, b);
        double dt_solve = tic_solve.toc();
        //Recover poses from X
        double changes = recoverPose6dfromLinear(X, finetune_rot);
        printf("[RotInit%d] solveLinearPose6d %.2fms setup %.2fms LLT %.2fms changes %.2f%% Poses %ld EffPoses %d Loops %ld Priors %ld F32: %d\n", self_id,
            tic.toc(), dt_setup, dt_solve, changes*100, frame_id_to_idx.size(), eff_frame_num, loops.size(), pose_priors.size(),
            typeid(T) == typeid(float));
        return changes;
    }

    double recoverRotationLLT(const VecX & X) {
        double state_changes_sum = 0;
        int count = 0;
        for (auto it : frame_id_to_idx) {
            auto frame_id = it.first;
            auto idx = getFrameIdx(frame_id);
            auto frame = state->getFramebyId(frame_id);
            if (idx == -1) {
                state->setAttitudeInit(frame_id, frame->odom.att());
                continue;
            }
            auto & pose = frame->odom.pose();
            Map<const Matrix<T, 3, 3, RowMajor>> M(X.data() + idx*ROTMAT_SIZE);
            //Note in X the rotation is stored in row major order.
            Map<Matrix<double, 3, 3, RowMajor>> R_state(CheckGetPtr(state->getRotState(frame_id)));
            auto Md = M.template cast<double>();
            double changes = (R_state-Md).norm()/R_state.norm();
            state_changes_sum += changes;
            count ++;
            R_state = M.template cast<double>(); //Not essential to be rotation matrix. For ARock.
            if (!(is_multi && frame->drone_id != state->getSelfId())) {
                auto R = recoverRotationSVD(Mat3(M));
                auto q = Quaternion<T>(R).template cast<double>();
                pose.att() = q;
                state->setAttitudeInit(frame_id, q);
                pose.to_vector(state->getPoseState(frame_id));
            }
        }
        return state_changes_sum/count;
    }

    double recoverPose6dfromLinear(const VecX & X, bool finetune_rot = true) {
        int pose_size = POSE_EFF_SIZE;
        double state_changes_sum = 0;
        int count = 0;
        if (!finetune_rot) {
            pose_size = POS_SIZE;
        }
        for (auto it : frame_id_to_idx) {
            auto frame_id = it.first;
            auto idx = getFrameIdx(frame_id);
            if (idx == -1) {
                continue;
            }
            auto frame = state->getFramebyId(frame_id);
            Matrix3d R0 = state->getAttitudeInit(frame_id).toRotationMatrix().template cast<double>();
            Vec3 t = X.segment(idx*pose_size, POS_SIZE);
            Map<Vector6d> perturb_state(CheckGetPtr(state->getPerturbState(frame_id)));
            double changes = (t.template cast<double>()-perturb_state.segment(0, POS_SIZE)).norm()/t.norm();
            perturb_state.segment(0, POS_SIZE) = t.template cast<double>();
            state_changes_sum += changes;
            count ++;
            if (finetune_rot) {
                Vec3 delta = X.segment(idx*POSE_EFF_SIZE+POS_SIZE, POS_SIZE);
                perturb_state.segment(POS_SIZE, POS_SIZE) = delta.template cast<double>();
            }
        }
        return state_changes_sum/count;
    }


public:
    RotationInitialization(PGOState * _state, RotInitConfig _config):
        state(_state), config(_config), self_id(_config.self_id) {
    }

    virtual void addLoop(const Swarm::LoopEdge & loop) {
        loops.emplace_back(loop);
    }
    
    virtual void addLoops(const std::vector<Swarm::LoopEdge> & _loops) {
        // printf("[RotationInitialization::addLoops] Adding %ld loops\n", _loops.size());
        loops = _loops;
    }

    void setFixedFrameId(FrameIdType _fixed_frame_id) {
        fixed_frame_ids.insert(_fixed_frame_id);
        // printf("[RotationInitialization::setFixedFrameId] Fixed frame id: %ld now %ld fixed\n", _fixed_frame_id, fixed_frame_ids.size());
    }

    SolverReport solve() {
        //Chordal relaxation algorithm.
        SolverReport report;
        TicToc tic;
        for (auto loop : loops) {
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            addFrameId(frame_id_a);
            addFrameId(frame_id_b);
        }
        updateFrameIdx();
        setPriorFactorsbyFixedParam();
        report.state_changes = solveLinearRot();
        report.total_time = tic.toc();
        report.total_iterations = 1;
        return report;   
    }

    SolverReport solveLinear6D() {
        SolverReport report;
        pose_priors.clear(); 
        setPriorFactorsbyFixedParam();
        for (int i = 0; i < config.pose6d_iterations; i ++) {
            solveLinearPose6d();
        }
        return report;
    }

    void reset() {
        loops.clear();
        pose_priors.clear(); 
    }
};

typedef RotationInitialization<double> RotationInitializationd;
typedef RotationInitialization<float> RotationInitializationf;
}