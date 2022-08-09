#pragma once
#include <d2common/utils.hpp>
#include "pgostate.hpp"
#include <swarm_msgs/relative_measurments.hpp>

namespace D2PGO {
//Chordal relaxation algorithm.
template<typename T>
class RotationInitialization {
typedef Eigen::Matrix<T, 3, 3> Mat3;
typedef Eigen::Matrix<T, 3, 1> Vec3;
typedef Eigen::Triplet<T> Tpl;

protected:
    PGOState & state;
    std::vector<Swarm::LoopEdge> loops;
    FrameIdType fixed_frame_id;
    std::map<FrameIdType, int> frame_id_to_idx;
    bool is_fixed_frame_set = false;

public:
    RotationInitialization(PGOState & _state):
        state(_state) {
    }

    void addLoop(const Swarm::LoopEdge & loop) {
        loops.emplace_back(loop);
    }
    
    void addLoops(const std::vector<Swarm::LoopEdge> & _loops) {
        printf("[RotationInitialization::addLoops] Adding %d loops\n", _loops.size());
        loops = _loops;
    }

    void setFixedFrameId(FrameIdType _fixed_frame_id) {
        fixed_frame_id = _fixed_frame_id;
        frame_id_to_idx[fixed_frame_id] = -1;
    }

    void addFrameId(FrameIdType _frame_id) {
        if (!is_fixed_frame_set) {
            setFixedFrameId(_frame_id);
            is_fixed_frame_set = true;
        } else {
            frame_id_to_idx[_frame_id] = frame_id_to_idx.size() - 2;
        }
    }

    void solve() {
        //Chordal relaxation algorithm.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        auto solver = new CeresSolver(&state, options);
        for (auto loop : loops) {
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            addFrameId(frame_id_a);
            addFrameId(frame_id_b);
            auto loop_factor = RelRotFactor9D::Create(loop);
            auto res_info = RelRot9DResInfo::create(loop_factor, 
                nullptr, loop.keyframe_id_a, loop.keyframe_id_b);
            solver->addResidual(res_info);
        }
        solver->getProblem().SetParameterBlockConstant(state.getRotState(fixed_frame_id));
        auto report = solver->solve();
        recoverRotation();
        std::cout << "Solver report: " << report.summary.FullReport() << std::endl;
        printf("[D2PGO::RotationInitialization] total frames %ld loops %ld opti_time %.1fms initial cost %.2e final cost %.2e\n", 
            frame_id_to_idx.size(), loops.size(), report.total_time*1000, report.initial_cost, report.final_cost);
    }

    void recoverRotation() {
        //Next recover the rotation matrices.
        for (auto it : frame_id_to_idx) {
            auto frame_id = it.first;
            if (it.second == -1) {
                continue;
            }
            Map<Matrix<T, 3, 3, RowMajor>> R(state.getRotState(frame_id));
            R = recoverRotationSVD(Matrix3d(R));
            state.getFramebyId(frame_id)->odom.pose().att() = Quaterniond(R);
        }
    }

protected:
    static Eigen::Matrix<T, 3, 3> recoverRotationSVD(Eigen::Matrix<T, 3, 3> M) {
        auto svd = M.jacobiSvd(ComputeFullV|ComputeFullU);
        auto S = svd.matrixU();
        auto Vt = svd.matrixV().transpose();
        auto detSV = (S*Vt).determinant();
        Eigen::Matrix<T, 3, 3> R = S*Vec3(1, 1, detSV).asDiagonal()*Vt;
        return R;
    }
};

typedef RotationInitialization<double> RotationInitializationd;
}