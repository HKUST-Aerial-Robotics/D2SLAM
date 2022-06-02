#include <d2common/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"
#include "../factors/imu_factor.h"
#include "../factors/depth_factor.h"
#include "../factors/prior_factor.h"
#include "../factors/projectionTwoFrameOneCamDepthFactor.h"
#include "../factors/projectionTwoFrameOneCamFactor.h"
#include "../factors/projectionOneFrameTwoCamFactor.h"
#include "../factors/projectionTwoFrameTwoCamFactor.h"
#include "../factors/projectionTwoFrameTwoCamFactorDistrib.h"
#include "../factors/pose_local_parameterization.h"
#include <d2frontend/utils.h>
#include "marginalization/marginalization.hpp"

namespace D2VINS {

D2Estimator::D2Estimator(int drone_id):
    self_id(drone_id), state(drone_id) {}

void D2Estimator::init(ros::NodeHandle & nh) {
    state.init(params->camera_extrinsics, params->td_initial);
    ProjectionTwoFrameOneCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactorDistrib::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameOneCamDepthFactor::sqrt_info = params->focal_length / 1.5 * Matrix3d::Identity();
    ProjectionTwoFrameOneCamDepthFactor::sqrt_info(2,2) = params->depth_sqrt_inf;
    visual.init(nh, this);
    printf("[D2Estimator::init] init done estimator on drone %d\n", self_id);
    for (auto cam_id : state.getAvailableCameraIds()) {
        Swarm::Pose ext = state.getExtrinsic(cam_id);
        printf("[D2VINS::D2Estimator] extrinsic %d: %s\n", cam_id, ext.toStr().c_str());
    }
}

void D2Estimator::inputImu(IMUData data) {
    imubuf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }
    //Propagation current with last Bias.
}

bool D2Estimator::tryinitFirstPose(VisualImageDescArray & frame) {
    auto ret = imubuf.periodIMU(-1, frame.stamp + state.getTd(frame.drone_id));
    auto _imubuf = ret.first;
    if (_imubuf.size() < params->init_imu_num) {
        return false;
    }
    auto q0 = Utility::g2R(_imubuf.mean_acc());
    auto last_odom = Swarm::Odometry(frame.stamp, Swarm::Pose(q0, Vector3d::Zero()));

    //Easily use the average value as gyrobias now
    //Also the ba with average acc - g
    VINSFrame first_frame(frame, _imubuf.mean_acc() - IMUBuffer::Gravity, _imubuf.mean_gyro());
    first_frame.is_keyframe = true;
    first_frame.odom = last_odom;
    first_frame.imu_buf_index = ret.second;

    state.addFrame(frame, first_frame);
    
    printf("\033[0;32m[D2VINS::D2Estimator] Initial firstPose %ld\n", frame.frame_id);
    printf("[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_odom.toStr().c_str());
    printf("[D2VINS::D2Estimator] Gyro bias: %.3f %.3f %.3f\n", first_frame.Bg.x(), first_frame.Bg.y(), first_frame.Bg.z());
    printf("[D2VINS::D2Estimator] Acc  bias: %.3f %.3f %.3f\033[0m\n\n", first_frame.Ba.x(), first_frame.Ba.y(), first_frame.Ba.z());

    frame.pose_drone = first_frame.odom.pose();
    frame.Ba = first_frame.Ba;
    frame.Bg = first_frame.Bg;
    return true;
}

std::pair<bool, Swarm::Pose> D2Estimator::initialFramePnP(const VisualImageDescArray & frame, const Swarm::Pose & initial_pose) {
    //Only use first image for initialization.
    auto & image = frame.images[0];
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    for (auto & lm: image.landmarks) {
        auto & lm_id = lm.landmark_id;
        if (state.hasLandmark(lm_id)) {
            auto & est_lm = state.getLandmarkbyId(lm_id);
            if (est_lm.flag >= LandmarkFlag::INITIALIZED) {
                pts3d.push_back(cv::Point3f(est_lm.position.x(), est_lm.position.y(), est_lm.position.z()));
                pts2d.push_back(cv::Point2f(lm.pt3d_norm.x()/lm.pt3d_norm.z(), lm.pt3d_norm.y()/lm.pt3d_norm.z()));
            }
        }
    }

    if (pts3d.size() < params->pnp_min_inliers) {
        return std::make_pair(false, Swarm::Pose());
    }

    cv::Mat inliers;
    cv::Mat D, rvec, t;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    D2FrontEnd::PnPInitialFromCamPose(initial_pose*image.extrinsic, rvec, t);
    // bool success = cv::solvePnP(pts3d, pts2d, K, D, rvec, t, true);
    bool success = cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, t, true, params->pnp_iteratives,  3, 0.99,  inliers);
    auto pose_cam = D2FrontEnd::PnPRestoCamPose(rvec, t);
    auto pose_imu = pose_cam*image.extrinsic.inverse();
    // printf("[D2VINS::D2Estimator] PnP initial %s final %s points %d\n", pose.toStr().c_str(), pose_imu.toStr().c_str(), pts3d.size());
    return std::make_pair(success, pose_imu);
}

void D2Estimator::addFrame(VisualImageDescArray & _frame) {
    //First we init corresponding pose for with IMU
    margined_landmarks = state.clearFrame();
    auto & last_frame = state.lastFrame();
    auto ret = imubuf.periodIMU(last_frame.imu_buf_index, _frame.stamp + state.td);
    auto _imu = ret.first;
    auto index = ret.second;
    if (fabs(_imu.size()/(_frame.stamp - last_frame.stamp) - params->IMU_FREQ) > 15) {
        printf("\033[0;31m[D2VINS::D2Estimator] Local IMU error freq: %.3f  start_t %.3f/%.3f end_t %.3f/%.3f\033[0m\n", 
            _imu.size()/(_frame.stamp - last_frame.stamp),
            last_frame.stamp + state.td, _imu[0].t, _frame.stamp + state.td, _imu[_imu.size()-1].t);
    }
    VINSFrame frame(_frame, ret, last_frame);
    if (params->init_method == D2VINSConfig::INIT_POSE_IMU) {
        frame.odom = _imu.propagation(last_frame);
    } else {
        auto odom_imu = _imu.propagation(last_frame);
        auto pnp_init = initialFramePnP(_frame, last_frame.odom.pose());
        if (!pnp_init.first) {
            //Use IMU
            printf("\033[0;31m[D2VINS::D2Estimator] Initialization failed, use IMU instead.\033[0m\n");
        } else {
            odom_imu.pose() = pnp_init.second;
        }
        frame.odom = odom_imu;
    }
    frame.odom.stamp = _frame.stamp;
    state.addFrame(_frame, frame);

    //Assign IMU and initialization to VisualImageDescArray for broadcasting.
    _frame.imu_buf = _imu;
    _frame.pose_drone = frame.odom.pose();
    _frame.Ba = frame.Ba;
    _frame.Bg = frame.Bg;

    if (params->verbose || params->debug_print_states) {
        printf("[D2VINS::D2Estimator] Initialize VINSFrame with %d: %s\n", 
            params->init_method, frame.toStr().c_str());
    }
}

void D2Estimator::addRemoteImuBuf(int drone_id, const IMUBuffer & imu_) {
    if (remote_imu_bufs.find(drone_id) == remote_imu_bufs.end()) {
        remote_imu_bufs[drone_id] = imu_;
        printf("[D2Estimator::addRemoteImuBuf] Assign imu buf to drone %d cur_size %d\n", drone_id, remote_imu_bufs[drone_id].size());
    } else {
        auto & _imu_buf = remote_imu_bufs.at(drone_id);
        auto t_last = _imu_buf.t_last;
        bool add_first = true;
        for (size_t i = 0; i < imu_.size(); i++) {
            if (imu_[i].t > t_last) {
                if (add_first) {
                    if ((imu_[i].t - t_last)  > params->max_imu_time_err) {
                        printf("\033[0;31m[D2VINS::D2Estimator] Add remote imu buffer %d: dt %.2fms\033[0m\\n", drone_id, (imu_[i].t - t_last)*1000);
                    }
                    add_first = false;
                }
                _imu_buf.add(imu_[i]);
            }
        }
    }
}

void D2Estimator::addFrameRemote(const VisualImageDescArray & _frame) {
    if (params->estimation_mode == D2VINSConfig::SOLVE_ALL_MODE || params->estimation_mode == D2VINSConfig::SERVER_MODE) {
        addRemoteImuBuf(_frame.drone_id, _frame.imu_buf);
    }
    int r_drone_id = _frame.drone_id;
    VINSFrame vinsframe;
    auto _imu = _frame.imu_buf;
    if (state.size(r_drone_id) > 0 ) {
        auto last_frame = state.lastFrame(r_drone_id);
        if (params->estimation_mode == D2VINSConfig::SOLVE_ALL_MODE || params->estimation_mode == D2VINSConfig::SERVER_MODE) {
            auto & imu_buf = remote_imu_bufs.at(_frame.drone_id);
            auto ret = imu_buf.periodIMU(last_frame.imu_buf_index, _frame.stamp + state.td);
            auto _imu = ret.first;
            if (fabs(_imu.size()/(_frame.stamp - last_frame.stamp) - params->IMU_FREQ) > 15) {
                printf("\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f  start_t %.3f/%.3f end_t %.3f/%.3f\033[0m\n", 
                    _imu.size()/(_frame.stamp - last_frame.stamp), last_frame.stamp + state.td, _imu[0].t,
                    _frame.stamp + state.td, _imu[_imu.size()-1].t);
            }
            vinsframe = VINSFrame(_frame, ret, last_frame);
        } else {
            vinsframe = VINSFrame(_frame, _frame.Ba, _frame.Bg);
        }
        auto ego_last = last_frame.initial_ego_pose;
        auto ego_cur = _frame.pose_drone;
        auto pred_cur_pose = last_frame.odom.pose() * ego_last.inverse()*ego_cur;
        if (params->verbose) {
            printf("[D2VINS::D2Estimator] Initial remoteframe %ld@drone%d with ego-motion: %s\n",
                _frame.frame_id, r_drone_id, pred_cur_pose.toStr().c_str());
        }
        vinsframe.odom.pose() = pred_cur_pose;
    } else {
        //Need to init the first frame.
        vinsframe = VINSFrame(_frame, _frame.Ba, _frame.Bg);
        auto pnp_init = initialFramePnP(_frame, Swarm::Pose::Identity());
        if (!pnp_init.first) {
            //Use IMU
            if (params->verbose) {
                printf("\033[0;31m[D2VINS::D2Estimator] Initialization failed for remote %d@%d.\033[0m\n", _frame.frame_id, _frame.drone_id);
            }
        } else {
            if (params->verbose) {
                printf("\033[0;32m[D2VINS::D2Estimator] Initial first remoteframe@drone%d with PnP: %s\033[0m\n", r_drone_id, pnp_init.second.toStr().c_str());
                vinsframe.odom.pose() = pnp_init.second;
            }
        }
    }

    state.addFrame(_frame, vinsframe);
    if (params->verbose || params->debug_print_states) {
        printf("[D2VINS::D2Estimator] Add Remote VINSFrame with %d: %s IMU %d iskeyframe %d/%d\n", 
            _frame.drone_id, vinsframe.toStr().c_str(), _frame.imu_buf.size(), vinsframe.is_keyframe, _frame.is_keyframe);
    }
}

void D2Estimator::addSldWinToFrame(VisualImageDescArray & frame) {
    for (int i = 0; i < state.size(); i ++) {
        frame.sld_win_status.push_back(state.getFrame(i).frame_id);
    }

}

void D2Estimator::inputRemoteImage(VisualImageDescArray & frame) {
    state.updateSldwin(frame.drone_id, frame.sld_win_status);
    addFrameRemote(frame);
    if (params->estimation_mode == D2VINSConfig::SERVER_MODE && state.size(frame.drone_id) >= params->min_solve_frames) {
        state.clearFrame();
        solve();
    }
}

bool D2Estimator::inputImage(VisualImageDescArray & _frame) {
    //We MUST make sure this function is running by only one thread.
    //It is not thread safe.
    if(!initFirstPoseFlag) {
        printf("[D2VINS::D2Estimator] tryinitFirstPose imu buf %ld\n", imubuf.size());
        initFirstPoseFlag = tryinitFirstPose(_frame);
        return initFirstPoseFlag;
    }

    double t_imu_frame = _frame.stamp + state.td;
    while (!imubuf.available(t_imu_frame)) {
        //Wait for IMU
        usleep(2000);
        printf("[D2VINS::D2Estimator] wait for imu...\n");
    }

    addFrame(_frame);
    if (state.size() >= params->min_solve_frames && params->estimation_mode != D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
        solve();
    } else {
        //Presolve only for initialization.
        state.preSolve(remote_imu_bufs);
    }
    addSldWinToFrame(_frame);
    frame_count ++;
    return true;
}

void D2Estimator::setStateProperties(ceres::Problem & problem) {
    // ceres::EigenQuaternionManifold quat_manifold;
    // ceres::EuclideanManifold<3> euc_manifold;
    // auto pose_manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>(euc_manifold, quat_manifold);
    auto pose_local_param = new PoseLocalParameterization;
    //set LocalParameterization
    for (auto & drone_id : state.availableDrones()) {
        if (state.size(drone_id) > 0) {
            for (size_t i = 0; i < state.size(drone_id); i ++) {
                auto frame_a = state.getFrame(drone_id, i);
                problem.SetParameterization(state.getPoseState(frame_a.frame_id), pose_local_param);
            }
        }
        if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
            if (relative_frame_is_used[drone_id]) {
                if (drone_id == self_id) {
                    problem.SetParameterBlockConstant(state.getPwikState(drone_id));
                } else {
                        problem.SetParameterization(state.getPwikState(drone_id), pose_local_param);
                }
            }
        }
    }

    for (auto cam_id: used_camera_sets) {
        if (!params->estimate_extrinsic || state.size() < params->max_sld_win_size) {
            problem.SetParameterBlockConstant(state.getExtrinsicState(cam_id));
        } else {
            // problem.SetManifold(state.getExtrinsicState(cam_id), pose_manifold);
            problem.SetParameterization(state.getExtrinsicState(cam_id), pose_local_param);
        }
    }

    if (!params->estimate_td || state.size() < params->max_sld_win_size) {
        problem.SetParameterBlockConstant(state.getTdState(self_id));
    }

    if (!state.getPrior() || params->always_fixed_first_pose) {
        if (params->estimation_mode < D2VINSConfig::SERVER_MODE) {
            problem.SetParameterBlockConstant(state.getPoseState(state.firstFrame().frame_id));
        } else {
            //Set first drone pose as fixed (whatever it is).
            if (state.availableDrones().size() > 0) {
                auto drone_id = *state.availableDrones().begin();
                problem.SetParameterBlockConstant(state.getPoseState(state.firstFrame(drone_id).frame_id));
            }
        }
    }
}


bool D2Estimator::isMain() const {
    return self_id == 1; //Temp code/
}

void D2Estimator::solveinDistributedMode() {
    relative_frame_is_used.clear();
    for (auto & drone_id : state.availableDrones()) {
        relative_frame_is_used[drone_id] = false;
    }
}

void D2Estimator::solve() {
    if (marginalizer!=nullptr) {
        delete marginalizer;
    }
    marginalizer = new Marginalizer(&state);
    state.setMarginalizer(marginalizer);
    solve_count ++;
    state.preSolve(remote_imu_bufs);
    used_camera_sets.clear(); 
    if (problem != nullptr) {
        delete problem;
    }
    problem = new ceres::Problem();
    setupImuFactors(*problem);
    setupLandmarkFactors(*problem);
    setupPriorFactor(*problem);
    setStateProperties(*problem);

    ceres::Solver::Summary summary;
    // params->options.?
    ceres::Solve(params->options, problem, &summary);
    state.syncFromState();

    //Now do some statistics
    static double sum_time = 0;
    static double sum_iteration = 0;
    static double sum_cost = 0;
    sum_time += summary.total_time_in_seconds;
    sum_iteration += summary.num_successful_steps + summary.num_unsuccessful_steps;
    sum_cost += summary.final_cost;

    if (params->enable_perf_output) {
        std::cout << summary.BriefReport() << std::endl;
        printf("[D2VINS] average time %.1fms, average time of iter: %.1fms, average iteration %.3f, average cost %.3f\n", 
            sum_time*1000/solve_count, sum_time*1000/sum_iteration, sum_iteration/solve_count, sum_cost/solve_count);
    }

    if (params->estimation_mode < D2VINSConfig::SERVER_MODE) {
        auto last_odom = state.lastFrame().odom;
        printf("[D2VINS] solve_count %d landmarks %d odom %s td %.1fms opti_time %.1fms\n", solve_count, 
            current_landmark_num, last_odom.toStr().c_str(), state.td*1000, summary.total_time_in_seconds*1000);
    } else {
        printf("[D2VINS] solve_count %d landmarks %d td %.1fms opti_time %.1fms\n", solve_count, 
            current_landmark_num, state.td*1000, summary.total_time_in_seconds*1000);
    }

    // Reprogation
    for (auto drone_id : state.availableDrones()) {
        auto _imu = imubuf.back(state.lastFrame(drone_id).stamp + state.td);
        last_prop_odom[drone_id] = _imu.propagation(state.lastFrame(drone_id));
    }

    visual.postSolve();

    if (params->debug_print_states || params->debug_print_sldwin) {
        state.printSldWin();
    }

    if (summary.termination_type == ceres::FAILURE)  {
        std::cout << summary.message << std::endl;
        exit(1);
    }
}

void D2Estimator::addIMUFactor(ceres::Problem & problem, FrameIdType frame_ida, FrameIdType frame_idb, IntegrationBase* pre_integrations) {
    IMUFactor* imu_factor = new IMUFactor(pre_integrations);
    problem.AddResidualBlock(imu_factor, nullptr, 
        state.getPoseState(frame_ida), state.getSpdBiasState(frame_ida), 
        state.getPoseState(frame_idb), state.getSpdBiasState(frame_idb));
    if (params->always_fixed_first_pose) {
        //At this time we fix the first pose and ignore the margin of this imu factor to achieve better numerical stability
        return;
    }
    marginalizer->addImuResidual(imu_factor, frame_ida, frame_idb);
}

void D2Estimator::setupImuFactors(ceres::Problem & problem) {
    if (state.size() > 1) {
        for (size_t i = 0; i < state.size() - 1; i ++ ) {
            auto & frame_a = state.getFrame(i);
            auto & frame_b = state.getFrame(i + 1);
            auto pre_integrations = frame_b.pre_integrations; //Prev to current
            assert(frame_b.prev_frame_id == frame_a.frame_id && "Wrong prev frame id");
            addIMUFactor(problem, frame_a.frame_id, frame_b.frame_id, pre_integrations);
        }
    }

    // In non-distributed mode, we add IMU factor for each drone
    if (params->estimation_mode == D2VINSConfig::SOLVE_ALL_MODE || params->estimation_mode == D2VINSConfig::SERVER_MODE) {
        for (auto drone_id : state.availableDrones()) {
            if (drone_id == self_id) {
                continue;
            }
            if (state.size(drone_id) > 1) {
                for (size_t i = 0; i < state.size(drone_id) - 1; i ++ ) {
                    auto & frame_a = state.getFrame(drone_id, i);
                    auto & frame_b = state.getFrame(drone_id, i + 1);
                    auto pre_integrations = frame_b.pre_integrations; //Prev to current
                    if (pre_integrations == nullptr) {
                        printf("\033[0;31m[D2VINS] Warning: frame %ld<->%ld@drone%d pre_integrations is nullptr.\033[0m\n",
                            frame_a.frame_id, frame_b.frame_id, drone_id);
                        continue;
                    }
                    assert(frame_b.prev_frame_id == frame_a.frame_id && "Wrong prev frame id on remote");
                    addIMUFactor(problem, frame_a.frame_id, frame_b.frame_id, pre_integrations);
                }
            }
        }
    }
}

void D2Estimator::setupLandmarkFactors(ceres::Problem & problem) {
    auto lms = state.availableLandmarkMeasurements();
    current_landmark_num = lms.size();
    auto loss_function = new ceres::HuberLoss(1.0);    
    int residual_count = 0;
    std::map<FrameIdType, int> keyframe_measurements;
    if (params->verbose) {
        printf("[D2VINS::setupLandmarkFactors] %d landmarks\n", lms.size());
    }
    for (auto lm : lms) {
        auto lm_id = lm.landmark_id;
        if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
            if (lm.solver_id == -1 && lm.drone_id != self_id) {
                // This is a internal only remote landmark
                printf("[D2VINS::setupLandmarkFactors] skip remote landmark %d solver_id %ld drone_id %ld\n", lm_id, lm.solver_id, lm.drone_id);
                continue;
            }
            if (lm.solver_id > 0 && lm.solver_id != self_id) {
                printf("[D2VINS::setupLandmarkFactors] skip remote landmark %d solver_id %ld drone_id %ld\n", lm_id, lm.solver_id, lm.drone_id);
                continue;
            }
        }
        LandmarkPerFrame firstObs = lm.track[0];
        auto & firstFrame = state.getFramebyId(firstObs.frame_id);
        auto base_camera_id = firstObs.camera_id;
        auto mea0 = firstObs.measurement();
        keyframe_measurements[firstObs.frame_id] ++;
        state.getLandmarkbyId(lm_id).solver_flag = LandmarkSolverFlag::SOLVED;
        if (firstObs.depth_mea && params->fuse_dep && 
                firstObs.depth < params->max_depth_to_fuse &&
                firstObs.depth > params->min_depth_to_fuse) {
            auto f_dep = OneFrameDepth::Create(firstObs.depth);
            problem.AddResidualBlock(f_dep, loss_function, state.getLandmarkState(lm_id));
            marginalizer->addDepthResidual(f_dep, loss_function, firstObs.frame_id, lm_id);
            residual_count++;
        }
        for (auto i = 1; i < lm.track.size(); i++) {
            auto lm_per_frame = lm.track[i];
            auto mea1 = lm_per_frame.measurement();
            auto & frame1 = state.getFramebyId(lm_per_frame.frame_id);
            if (lm_per_frame.camera_id == base_camera_id) {
                ceres::CostFunction * f_td = nullptr;
                if (lm_per_frame.depth_mea && params->fuse_dep &&
                    lm_per_frame.depth < params->max_depth_to_fuse && 
                    lm_per_frame.depth > params->min_depth_to_fuse) {
                    f_td = new ProjectionTwoFrameOneCamDepthFactor(mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
                        firstObs.cur_td, lm_per_frame.cur_td, lm_per_frame.depth);
                } else {
                    f_td = new ProjectionTwoFrameOneCamFactor(mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
                        firstObs.cur_td, lm_per_frame.cur_td);
                }
                if (firstObs.frame_id == lm_per_frame.frame_id) {
                    printf("\033[0;31m[ [D2VINS::setupLandmarkFactors] Warning: landmarkid %ld frame %ld<->%ld@%ld is the same camera id %d.\033[0m\n",
                        lm_per_frame.landmark_id, firstObs.frame_id, lm_per_frame.frame_id, lm_id, base_camera_id);
                    continue;
                }
                problem.AddResidualBlock(f_td, loss_function,
                    state.getPoseState(firstObs.frame_id), 
                    state.getPoseState(lm_per_frame.frame_id), 
                    state.getExtrinsicState(firstObs.camera_id),
                    state.getLandmarkState(lm_id), state.getTdState(lm_per_frame.drone_id));
                marginalizer->addLandmarkResidual(f_td, loss_function,
                    firstObs.frame_id, lm_per_frame.frame_id, lm_id, firstObs.camera_id, true);
                residual_count++;
                keyframe_measurements[lm_per_frame.frame_id] ++;
                used_camera_sets.insert(firstObs.camera_id);
            } else {
                if (lm_per_frame.frame_id == firstObs.frame_id) {
                    auto f_td = new ProjectionOneFrameTwoCamFactor(mea0, mea1, firstObs.velocity, 
                        lm_per_frame.velocity, firstObs.cur_td, lm_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, nullptr,
                        state.getExtrinsicState(firstObs.camera_id),
                        state.getExtrinsicState(lm_per_frame.camera_id),
                        state.getLandmarkState(lm_id), state.getTdState(lm_per_frame.drone_id));
                    marginalizer->addLandmarkResidualOneFrameTwoCam(f_td, loss_function,
                        firstObs.frame_id, lm_id, firstObs.camera_id, lm_per_frame.camera_id);
                    residual_count++;
                } else {
                    if (firstFrame.drone_id != frame1.drone_id) {
                        auto f_td = new ProjectionTwoFrameTwoCamFactorDistrib(mea0, mea1, firstObs.velocity, 
                        lm_per_frame.velocity, firstObs.cur_td, lm_per_frame.cur_td);
                        problem.AddResidualBlock(f_td, loss_function,
                            state.getPoseState(firstObs.frame_id), 
                            state.getPoseState(lm_per_frame.frame_id), 
                            state.getExtrinsicState(firstObs.camera_id),
                            state.getExtrinsicState(lm_per_frame.camera_id),
                            state.getLandmarkState(lm_id), state.getTdState(lm_per_frame.drone_id), 
                            state.getPwikState(firstFrame.drone_id), state.getPwikState(frame1.drone_id));
                        relative_frame_is_used[firstFrame.drone_id] = true;
                        relative_frame_is_used[frame1.drone_id] = true;
                        marginalizer->addLandmarkResidualTwoFrameTwoCamDistrib(f_td, loss_function,
                            firstObs.frame_id, lm_per_frame.frame_id, lm_id, firstObs.camera_id, lm_per_frame.camera_id);
                        residual_count++;
                    } else {
                        auto f_td = new ProjectionTwoFrameTwoCamFactor(mea0, mea1, firstObs.velocity, 
                            lm_per_frame.velocity, firstObs.cur_td, lm_per_frame.cur_td);
                        problem.AddResidualBlock(f_td, loss_function,
                            state.getPoseState(firstObs.frame_id), 
                            state.getPoseState(lm_per_frame.frame_id), 
                            state.getExtrinsicState(firstObs.camera_id),
                            state.getExtrinsicState(lm_per_frame.camera_id),
                            state.getLandmarkState(lm_id), state.getTdState(lm_per_frame.drone_id));
                        marginalizer->addLandmarkResidualTwoFrameTwoCam(f_td, loss_function,
                            firstObs.frame_id, lm_per_frame.frame_id, lm_id, firstObs.camera_id, lm_per_frame.camera_id);
                        residual_count++;
                    }
                }
                used_camera_sets.insert(lm_per_frame.camera_id);
            }
            keyframe_measurements[lm_per_frame.frame_id] ++;
            problem.SetParameterLowerBound(state.getLandmarkState(lm_id), 0, params->min_inv_dep);
        }
    }
    //Check the measurements number of each keyframe
    for (auto it : keyframe_measurements) {
        auto frame_id = it.first;
        if (it.second < params->min_measurements_per_keyframe) {
            printf("\033[0;31m[D2VINS::D2Estimator] frame_id %ld has only %d measurements\033[0m\n Related landmarks:\n", 
                frame_id, it.second);
            std::vector<LandmarkPerId> related_landmarks = state.getRelatedLandmarks(frame_id);
            for (auto lm : related_landmarks) {
                printf("Landmark %ld tracks %ld flag %d\n", lm.landmark_id, lm.track.size(), lm.flag);
            }
            printf("====================");
        }
    }
    if (params->verbose) {
        printf("[D2VINS::setupLandmarkFactors] %d residuals\n", lms.size());
    }
}

void D2Estimator::setupPriorFactor(ceres::Problem & problem) {
    auto prior_factor = state.getPrior();
    if (prior_factor != nullptr) {
        auto pfactor = new PriorFactor(*prior_factor);
        problem.AddResidualBlock(pfactor, nullptr, prior_factor->getKeepParamsPointers());
        marginalizer->addPrior(pfactor);
    }
}

std::vector<LandmarkPerId> D2Estimator::getMarginedLandmarks() const {
    return margined_landmarks;
}

Swarm::Odometry D2Estimator::getImuPropagation() const {
    return last_prop_odom.at(self_id);
}

Swarm::Odometry D2Estimator::getOdometry() const {
    return getOdometry(self_id);
}

Swarm::Odometry D2Estimator::getOdometry(int drone_id) const {
    return state.lastFrame(drone_id).odom;
}


D2EstimatorState & D2Estimator::getState() {
    return state;
}

bool D2Estimator::isLocalFrame(FrameIdType frame_id) const {
    return state.getFramebyId(frame_id).drone_id == self_id;
}

}
