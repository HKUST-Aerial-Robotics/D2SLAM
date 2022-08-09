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
    PGOState state;
    std::vector<Swarm::LoopEdge> loops;
    FrameIdType fixed_frame_id;
    std::map<FrameIdType, int> frame_id_to_idx;
    bool is_fixed_frame_set = false;
public:
    RotationInitialization(PGOState _state):
        state(_state) {
    }

    void addLoop(const Swarm::LoopEdge & loop) {
        loops.emplace_back(loop);
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
        std::vector<Tpl> triplet_list;
        for (auto loop : loops) {
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            addFrameId(frame_id_a);
            addFrameId(frame_id_b);
        }

        VectorXd b((frame_id_to_idx.size() - 1)*9);
        b.setZero();
        auto pose_fixed = state.getFramebyId(fixed_frame_id)->odom.pose();
        Mat3 R_fix = pose_fixed.att().toRotationMatrix().template cast<T>();
        int row_id = 0;
        for (auto loop : loops) {
            //Set up the problem
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            auto idx_a = frame_id_to_idx.at(frame_id_a);
            auto idx_b = frame_id_to_idx.at(frame_id_b);
            if (idx_a == idx_b) {
                continue;
            }
            Mat3 R = loop.relative_pose.att().toRotationMatrix().template cast<T>();
            Mat3 Rt = R.transpose();
            if (idx_a == -1) {
                Mat3 Rj_t = (R_fix * R).transpose();
                for (int k = 0; k < 3; k ++) {  //Row of Rotation of keyframe b
                    int j0 = 9*idx_b + 3*k;
                    for (auto i = 0; i < 3; i ++) { //Row, col of relative rotation R
                        triplet_list.push_back(Tpl(row_id + k + i, j0 + i, 1));
                    }
                    b.segment<3>(row_id + k) = Rj_t.col(k);                    
                }
            } else if (idx_b == -1) {
                Mat3 Ri_t = (R_fix * Rt).transpose();
                for (int k = 0; k < 3; k ++) {  //Row of Rotation of keyframe b
                    int j0 = 9*idx_a + 3*k;
                    for (auto i = 0; i < 3; i ++) { //Row, col of relative rotation R
                        triplet_list.push_back(Tpl(row_id + k + i, j0 + i, 1));
                    }
                    b.segment<3>(row_id + k) = Ri_t.col(k);                    
                }
            } else {
                for (int k = 0; k < 3; k ++) { //Row of Rotation of keyframe a
                    int j0 = 9*idx_a + 3*k;
                    for (int i = 0; i < 3; i ++) { //Row of relative rotation R 
                        for (auto j = 0; j < 3; j ++) { //Column of relative rotation R
                            triplet_list.push_back(Tpl(row_id + k + i, j0 + j, Rt(i, j)));
                        }
                    }
                }
                for (int k = 0; k < 3; k ++) {  //Row of Rotation of keyframe b
                    int j0 = 9*idx_b + 3*k;
                    for (auto i = 0; i < 3; i ++) { //Row, col of relative rotation R
                        triplet_list.push_back(Tpl(row_id + k + i, j0 + i, -1));
                    }
                }
            }
            row_id += 3;
        }
    }

protected:
    static Eigen::Matrix<T, 3, 3> recoverRotationSVD(const Eigen::Matrix<T, 3, 3> & M) {
        auto svd = M.jacobiSvd(ComputeFullV|ComputeFullU);
        auto S = svd.matrixU();
        auto Vt = svd.matrixV().transpose();
        auto detSV = (S*Vt).detereminant();
        Eigen::Matrix<T, 3, 3> R = S*Vec3(1, 1, detSV).asDiagonal()*Vt;
        return R;
    }
};
}